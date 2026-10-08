# CVaR Portfolio Backtest

Python research prototype for comparing Equal-Weight, Minimum-Variance, and
Conditional Value-at-Risk (CVaR) portfolios across six UCITS ETFs. It includes
data preparation, portfolio optimisation, rolling backtests, and analysis notebooks.

This README covers setup and development. Research methodology, results, and
interpretation belong in the companion
[LaTeX dissertation repository](https://github.com/ZamirLucky/rd2-etf-cvar-backtest).

## Tech stack

| Purpose | Tools |
| --- | --- |
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| Market data downloads | yfinance |
| Optimisation | SciPy (SLSQP), CVXPY (CLARABEL by default) |
| Visualisation | Matplotlib, seaborn |
| Interactive analysis | Jupyter Notebook |
| Testing | pytest |

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ZamirLucky/cvar-portfolio-backtest.git
cd cvar-portfolio-backtest
python -m venv .venv
```

Activate it in **Windows PowerShell**:

```powershell
.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project structure

```text
src/
  loads_data/       # Vendor downloads and ETF02 data stitching
  strategies/       # Equal-Weight, Minimum-Variance, and CVaR functions
  data_loader.py    # ETF CSV and FX loading
  data_prep.py      # Price alignment, EUR conversion, and log returns
  backtester.py     # Rolling backtests and CSV output
  metrics.py        # Summary statistics and performance metrics
tests/              # Automated tests
notebooks/          # Exploration, strategy checks, and analysis
data/               # Raw snapshots, processed data, and backtest results
figures/            # Notebook-generated charts
verify_setup.py     # Optional dependency, solver, and network check
```

## Running the project

Run these commands from the repository root with the environment activated.
These steps overwrite derived files in `data/processed/` and `data/results/`.

```bash
# Merge ETF histories
python -m src.data_prep

# Convert prices to EUR and calculate daily log returns
python -m src.data_prep eur

# Run all three baseline strategies
python -m src.backtester

# Generate the performance summary
python -c "from src.metrics import build_and_save_performance_summary as build; build(strategy_names=['equal_weight', 'min_variance', 'cvar_95'])"
```

To work with the notebooks, launch Jupyter from their directory:

```bash
cd notebooks
jupyter notebook
```

Notebooks `01` and `02` cover data exploration and strategy checks; `04` and `05`
generate analysis figures and sensitivity outputs. Run the pipeline above before
regenerating the analysis. There is no notebook `03`; that step is handled by the
backtester and metrics commands.

Notebook `05` reuses files in `data/results/sensitivity/`. Move the relevant cached
files aside before rerunning it after changing data, code, or configuration.

## Development

- Add or modify allocation functions in `src/strategies/`. Strategies passed to
  `run_backtest()` accept an estimation-window DataFrame and return a
  `pandas.Series` of asset weights. See `run_all_strategies()` for existing adapters.
- Use `BacktestConfig` in `src/backtester.py` to configure the estimation window,
  rebalance frequency, CVaR confidence level, and output directory. Defaults are
  252 trading days, 21 trading days, `0.95`, and `data/results/` respectively.
- Keep reusable logic in `src/` and add regression tests in `tests/` when changing
  calculations or behaviour. Use notebooks for exploration and charts.

Run the tests from the repository root:

```bash
python -m pytest -q
```

The suite uses synthetic data and covers strategies, rolling backtests, and metrics.
For an optional dependency and connectivity check,
run `python verify_setup.py`; it makes a live Yahoo Finance request.
