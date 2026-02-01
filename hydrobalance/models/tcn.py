"""Temporal Convolutional Network (TCN) forecaster.

English:
    A lightweight neural forecaster using causal/dilated 1D convolutions.
    This version is *multi-horizon*: the model outputs the next H steps in one
    forward pass, which usually performs better than fully-recursive rollout.

    Improvements over a minimal baseline:
      - standardization (mean/std) on training series
      - train/validation split + early stopping
      - optional gradient clipping for stability

日本語:
    因果(dilated)1D畳み込みを用いた軽量なニューラル時系列予測モデルです。
    本実装は *マルチホライズン*（次のHステップを一度に出力）で、
    再帰予測より誤差の蓄積を抑えやすい構成です。

    追加改善:
      - 標準化（平均/標準偏差）
      - 学習/検証分割 + 早期停止
      - 安定化のための勾配クリップ（任意）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn


class _Chomp1d(nn.Module):
    """Remove right-side padding to keep causality.

    EN: ensures output at time t depends only on <=t.
    JP: 出力が未来の情報に依存しないようにpadding分を削除します。
    """

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.chomp] if self.chomp > 0 else x


class _TemporalBlock(nn.Module):
    """A residual TCN block."""

    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.act(y + res)


class TCN(nn.Module):
    """Backbone + linear head.

    EN: Use the last hidden state to predict the next horizon values.
    JP: 最終時刻の表現から次Hステップを予測します。
    """

    def __init__(self, in_ch: int, channels: list[int], k: int, dropout: float, out_horizon: int):
        super().__init__()
        blocks = []
        ch_prev = in_ch
        for i, ch in enumerate(channels):
            blocks.append(_TemporalBlock(ch_prev, ch, k=k, dilation=2**i, dropout=dropout))
            ch_prev = ch
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(ch_prev, out_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.backbone(x)           # [B, ch, T]
        last = h[:, :, -1]             # [B, ch]
        return self.head(last)         # [B, H]


@dataclass(frozen=True)
class TCNSpec:
    lookback: int = 96
    channels: Tuple[int, ...] = (32, 32, 32)
    kernel_size: int = 3
    dropout: float = 0.15
    lr: float = 1e-3
    max_epochs: int = 80
    batch_size: int = 128
    valid_fraction: float = 0.15
    patience: int = 10
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42


class _ZScore:
    """Simple 1D standard scaler."""

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    @classmethod
    def fit(cls, values: np.ndarray) -> "_ZScore":
        v = np.asarray(values, dtype=float)
        return cls(np.nanmean(v), np.nanstd(v) + 1e-8)

    def transform(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return (v - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return v * self.std + self.mean


def _make_windows_multi(values: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, Y) pairs for multi-step prediction.

    EN:
        X_i = values[i-lookback : i]
        Y_i = values[i : i+horizon]

    JP:
        入力は直近lookback点、出力は次horizon点です。
    """
    xs, ys = [], []
    n = len(values)
    last_i = n - horizon
    for i in range(lookback, last_i + 1):
        xs.append(values[i - lookback : i])
        ys.append(values[i : i + horizon])
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y


def fit_predict(train_df: pd.DataFrame, horizon: int, spec: TCNSpec) -> np.ndarray:
    """Fit TCN and predict the next horizon values.

    English:
        We train a direct multi-horizon model (output dimension = horizon).
        Training uses a time-aware split and early stopping.

    日本語:
        horizon次元の直接予測モデルを学習します。
        時系列順を保った検証分割 + 早期停止で安定に学習します。
    """

    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)

    y_raw = train_df["value"].to_numpy(dtype=float)
    if len(y_raw) <= spec.lookback + horizon + 20:
        raise ValueError("Not enough history for TCN lookback/horizon.")

    scaler = _ZScore.fit(y_raw)
    y = scaler.transform(y_raw).astype(np.float32)

    Xw, Yw = _make_windows_multi(y, spec.lookback, horizon)

    # Time-aware train/valid split (no shuffle)
    n = len(Xw)
    n_valid = max(1, int(n * spec.valid_fraction))
    X_tr, X_va = Xw[:-n_valid], Xw[-n_valid:]
    Y_tr, Y_va = Yw[:-n_valid], Yw[-n_valid:]

    device = torch.device(spec.device)
    model = TCN(1, list(spec.channels), k=spec.kernel_size, dropout=spec.dropout, out_horizon=horizon).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    def batches(x: np.ndarray, y_: np.ndarray, bs: int):
        # EN: shuffle within training set for SGD.
        # JP: 学習データのみシャッフル。
        idx = rng.permutation(len(x))
        for i in range(0, len(x), bs):
            j = idx[i : i + bs]
            yield x[j], y_[j]

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(spec.max_epochs):
        model.train()
        for xb, yb in batches(X_tr, Y_tr, spec.batch_size):
            xb_t = torch.from_numpy(xb).unsqueeze(1).to(device)   # [B,1,T]
            yb_t = torch.from_numpy(yb).to(device)               # [B,H]

            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if spec.grad_clip and spec.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            xva = torch.from_numpy(X_va).unsqueeze(1).to(device)
            yva = torch.from_numpy(Y_va).to(device)
            vloss = float(loss_fn(model(xva), yva).cpu().item())

        scheduler.step(vloss)

        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= spec.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict next horizon from the most recent window
    model.eval()
    last_window = scaler.transform(y_raw[-spec.lookback:]).astype(np.float32)
    xb = torch.from_numpy(last_window).view(1, 1, -1).to(device)
    with torch.no_grad():
        pred_scaled = model(xb).cpu().numpy().reshape(-1)

    pred = scaler.inverse(pred_scaled).astype(float)
    return pred
