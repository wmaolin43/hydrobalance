# HydroBalance（日本語）

USGSの水文データ（例：水位・流量）を取得し、  
**時系列予測**と**貯水池運用の多目的最適化（貯水量 vs 発電量）**を一つのリポジトリにまとめたポートフォリオ用プロジェクトです。

- 予測モデル：Baseline / SARIMAX / LightGBM / TCN（PyTorch）
- 評価：rolling-origin backtest（時系列のリークを避ける）
- ベイズ最適化：GP + EI を自作実装（依存少なめで読みやすい）
- 運用最適化：NSGA-II（簡易実装） + TOPSIS

使い方の詳細は `README.md` を参照してください。
