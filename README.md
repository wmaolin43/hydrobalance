# HydroBalance: USGS-driven water-level forecasting + reservoir operation optimization

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A portfolio-grade project that turns **USGS hydrology time series** into:
1) **water level / streamflow forecasts** and  
2) **multi-objective reservoir operation recommendations** (storage vs. hydropower).

This repository is inspired by an **MCM/ICM-style** modeling pipeline and upgrades it into a reproducible, testable, interview-friendly codebase.

---

## Why this project exists

The lower Colorado River basin has seen dramatic declines in reservoir levels (e.g., Lake Powell / Lake Mead), which makes **water allocation** and **hydropower planning** more difficult.  
In our 2022 MCM/ICM project (Problem B), we built a forecasting + optimization framework using:

- **SARIMA** to predict water levels  
- **NSGA-II** to optimize (max storage, max power)  
- **TOPSIS** to pick a compromise point from the Pareto set  
- AHP/FAHP and regression for allocation/demand analysis

This repo keeps the *core modeling logic* but ships it as production-quality Python with modern baselines and clean evaluation.

---

## Features

| Layer | What you get | Why it matters for interviews |
|---|---|---|
| Data | USGS Water Services API client (dv/iv), tidy CSV output | Real-world API ingestion + data QA |
| Forecasting | Baselines + SARIMAX + LightGBM + TCN (PyTorch) | Compare transparent vs. strong models |
| Evaluation | Rolling-origin backtesting, RMSE/MAE/sMAPE + **MASE/Peak-RMSE** | Correct time-series evaluation (no leakage) + peak-aware checks |
| Uncertainty | Simple conformal-style intervals (residual quantiles) | Communicate risk/uncertainty in decisions |
| Bayesian Optimization | Custom GP + Expected Improvement | Shows DS fundamentals beyond “just Optuna” |
| Operations | Minimal NSGA-II + TOPSIS selection | Multi-objective optimization in practice |
| Engineering | CLI scripts, tests, packaging, clear docs | GitHub hygiene & maintainability |

---

## Repository structure

```
hydrobalance/
  data/          # USGS fetch + preprocessing
  features/      # lag features for tabular ML
  models/        # SARIMAX / GBM / TCN
  eval/          # metrics + rolling-origin backtest
  opt/           # GP-based Bayesian optimization
  ops/           # hydropower + NSGA-II + TOPSIS
scripts/
  fetch_usgs.py
  train_forecaster.py
  tune_sarimax_bo.py
  optimize_policy_nsga2.py
docs/
  project_context.md
  legacy/2204534.pdf
tests/
```

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

### 2) Fetch USGS data (example)

Pick a **site** and a **parameter code**:

- `00065` = gage height (water level)
- `00060` = discharge (streamflow)

Then fetch daily values:

```bash
python scripts/fetch_usgs.py \
  --site 09380000 \
  --param 00065 \
  --start 2010-01-01 \
  --end 2024-12-31 \
  --service dv \
  --out data/raw/usgs_level.csv
```

### 3) Backtest a model

SARIMAX baseline:

```bash
python scripts/train_forecaster.py --csv data/raw/usgs_level.csv --model sarimax --horizon 30
```

LightGBM:

```bash
python scripts/train_forecaster.py --csv data/raw/usgs_level.csv --model gbm --horizon 30
```

Neural TCN:

```bash
python scripts/train_forecaster.py --csv data/raw/usgs_level.csv --model tcn --horizon 30
```

Outputs are stored under `artifacts/`.

---

## Method overview

### A) Forecasting

We treat water level / flow as a time series and evaluate with walk-forward splits.

Implemented models:

- **Baselines**: last-value, seasonal naive
- **SARIMAX**: interpretable seasonal model, strong when data is stable
- **LightGBM**: non-linear tabular model with lag/rolling + diff/EMA + calendar (sin/cos) features
- **TCN**: multi-horizon neural model with causal/dilated convolutions (PyTorch), with scaling + early stopping

### Uncertainty (prediction intervals)

After you backtest a point model, you can build a quick **conformal-style** symmetric interval using
historical absolute errors (see `hydrobalance/eval/intervals.py`). This is intentionally simple and interview-friendly.

### B) Bayesian optimization (hyperparameter tuning)

Instead of relying on a third-party library, we provide a **minimal** BO implementation:

- Surrogate: `GaussianProcessRegressor`
- Acquisition: **Expected Improvement** (EI)
- Candidate search: random sampling in bounded box (low-dimensional)

Example: tune SARIMAX orders with rolling-origin RMSE:

```bash
python scripts/tune_sarimax_bo.py --csv data/raw/usgs_level.csv --out artifacts/sarimax_bo.json
```

### C) Reservoir operation optimization (NSGA-II + TOPSIS)

We optimize a parametric release policy with two objectives:

1) **maximize storage** (reduce shortage risk)  
2) **maximize hydropower**  

We use **NSGA-II** to generate a Pareto set and **TOPSIS** to pick a compromise solution.

```bash
python scripts/optimize_policy_nsga2.py --out artifacts/policy_pareto.npz --w1 0.5 --w2 0.5
```

> The simulator used in this demo is intentionally simple. Replace it with a full mass-balance model (inflow, evaporation, downstream constraints) if you want to deploy.

---

## Inputs / Outputs

### Inputs

- USGS time series (dv/iv JSON → tidy DataFrame)
- Config: forecast horizon, backtest step, model hyperparameters
- Policy bounds for multi-objective optimization

### Outputs

- `data/raw/*.csv`: fetched time series
- `artifacts/backtest.json`: per-fold metrics + summary
- `artifacts/sarimax_bo.json`: BO trace + best SARIMAX spec
- `artifacts/policy_pareto.npz`: Pareto set + TOPSIS recommendation

---

## How this differs from typical “toy forecasting notebooks”

- ✅ correct time-series CV (rolling-origin)
- ✅ multiple models + consistent interface
- ✅ Bayesian optimization implemented (readable, dependency-light)
- ✅ multi-objective optimization (NSGA-II) + decision method (TOPSIS)
- ✅ engineering polish: CLI, packaging, tests, documentation

---

## Citation / credit

The contest report that inspired this repo is included in `docs/legacy/2204534.pdf`.

If you reuse the ideas, please cite **USGS** as the data source and cite the relevant modeling references in your own writing.

---

## Next extensions (maybe add next step)

- Add exogenous signals: precipitation, snowpack, temperature, evaporation
- Use probabilistic forecasts (quantile regression / conformal prediction)
- Replace the toy simulator with a calibrated hydrologic mass-balance model
- Build a small dashboard (Streamlit) for monitoring + forecast visualization

---

## License

MIT
