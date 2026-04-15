# Portfolio Construction

This repository contains Jupyter notebooks and helper code for the ESSCA MSc Finance & Data Analyst course on portfolio construction. The material moves from basic return calculations to drawdowns, risk diagnostics, Value at Risk, efficient frontiers, maximum Sharpe ratio portfolios, the global minimum variance portfolio, limits of diversification, CPPI dynamic allocation, and Monte Carlo simulation via random walk models.

The project is organized as a learning workspace rather than a packaged library. Most of the analysis lives in notebooks, while reusable functions are collected in `kit.py`.

## Repository Contents

- `1_RiskAjustedReturns.ipynb`: introductory return calculations, compounding, and basic performance measurement.
- `2_ComputeDrawDown.ipynb`: wealth index construction, running peaks, and drawdown analysis.
- `3_BuildModules.ipynb`: first step toward moving repeated logic into reusable Python functions.
- `4_DeviationsFromNormality.ipynb`: skewness, kurtosis, and normality testing.
- `5_VaR_CVaR.ipynb`: semideviation, historic VaR, Gaussian VaR, modified VaR, and CVaR.
- `6_Quiz.ipynb`: applied exercises using portfolio and return datasets.
- `7_Efficient_Frontier.ipynb`: portfolio return, volatility, covariance, and efficient frontier basics.
- `8_Asset_efficient_frontier.ipynb`: additional efficient frontier plotting and portfolio combinations.
- `9_N_Asset_Efficient_Frontier.ipynb`: long-only optimization for multi-asset portfolios.
- `10_Max_Sharp_Ratio.ipynb`: tangency portfolio / maximum Sharpe ratio optimization.
- `11_GMV.ipynb`: global minimum variance portfolio construction.
- `12_Limits_of_Diversification.ipynb`: cap-weighted total market index, diversification limits, and industry-level analysis.
- `13_CPPI.ipynb`: Constant Proportion Portfolio Insurance (CPPI) strategy — dynamic floor-based allocation between a risky and a safe asset.
- `14_Random_Walk.ipynb`: Monte Carlo simulation of portfolio paths using geometric Brownian motion / random walk models.
- `Exercise/module2.ipynb`: extra practice notebook using the shared helper module.
- `kit.py`: shared functions for data loading, statistics, risk measures, portfolio optimization, and dynamic strategies.
- `data/`: local CSV datasets used by the notebooks and helper functions.

## Learning Flow

If you want to go through the repository in order, the recommended sequence is:

1. `1_RiskAjustedReturns.ipynb`
2. `2_ComputeDrawDown.ipynb`
3. `3_BuildModules.ipynb`
4. `4_DeviationsFromNormality.ipynb`
5. `5_VaR_CVaR.ipynb`
6. `6_Quiz.ipynb`
7. `7_Efficient_Frontier.ipynb`
8. `8_Asset_efficient_frontier.ipynb`
9. `9_N_Asset_Efficient_Frontier.ipynb`
10. `10_Max_Sharp_Ratio.ipynb`
11. `11_GMV.ipynb`
12. `12_Limits_of_Diversification.ipynb`
13. `13_CPPI.ipynb`
14. `14_Random_Walk.ipynb`

This order follows the progression from return measurement → portfolio construction and optimization → dynamic portfolio strategies and simulation.

## `kit.py` Overview

The `kit.py` module is the reusable core of the repository. It includes functions in six main categories:

- **Data loading**: `get_hfi_returns()`, `get_ind_returns()`, `get_ind_size()`, `get_ind_nfirms()`, `get_total_market_index_return()`
- **Risk and distribution statistics**: `drawdown()`, `skewness()`, `kurtosis()`, `is_normal()`, `semideviation()`
- **Tail-risk measures**: `var_historic()`, `var_gaussian()`, `cvar_historic()`
- **Annualization and portfolio analytics**: `annual_return()`, `annual_volatility()`, `sharpe_ratio()`, `portfolio_return()`, `portfolio_vol()`
- **Optimization and plotting**: `minimize_vol()`, `optimal_weights()`, `msr()`, `gmv()`, `plot_ef()`, `plot_ef2()`
- **Dynamic strategies and reporting**: `run_cppi()`, `summary_stats()`

There are also helper functions such as `annual_return_per_month()` and `annual_volatility_by_month()` for notebook-specific workflows.

Example:

```python
import kit as erk

hfi = erk.get_hfi_returns()
drawdowns = erk.drawdown(hfi["Convertible Arbitrage"])

# Run CPPI backtest
result = erk.run_cppi(risky_r=hfi["Convertible Arbitrage"], m=3, floor=0.8)

# Print a summary stats table
stats = erk.summary_stats(hfi)
```

## Data Files

The `data/` folder contains the CSV inputs used throughout the notebooks, including:

- EDHEC hedge fund index returns
- Fama-French factor datasets
- 30-industry and 49-industry return series
- portfolio sorts based on firm size (number of firms and average size per industry)
- sample price series and Berkshire Hathaway data

Most helper functions assume these files are available under the local `data/` directory with the current filenames.

## Setup

There is no `requirements.txt` file in this repository, so dependencies need to be installed manually. A minimal setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy matplotlib notebook
```

Then start Jupyter from the repository root:

```bash
jupyter notebook
```

Running Jupyter from the root folder matters because `kit.py` loads datasets with relative paths such as `data/edhec-hedgefundindices.csv`.

## Project Structure

```text
.
|-- 1_RiskAjustedReturns.ipynb
|-- 2_ComputeDrawDown.ipynb
|-- 3_BuildModules.ipynb
|-- 4_DeviationsFromNormality.ipynb
|-- 5_VaR_CVaR.ipynb
|-- 6_Quiz.ipynb
|-- 7_Efficient_Frontier.ipynb
|-- 8_Asset_efficient_frontier.ipynb
|-- 9_N_Asset_Efficient_Frontier.ipynb
|-- 10_Max_Sharp_Ratio.ipynb
|-- 11_GMV.ipynb
|-- 12_Limits_of_Diversification.ipynb
|-- 13_CPPI.ipynb
|-- 14_Random_Walk.ipynb
|-- BuildOwnModules/
|-- Exercise/
|   `-- module2.ipynb
|-- data/
|   `-- *.csv
|-- kit.py
`-- README.md
```

## Notes

- Some filenames reflect the original course naming and have been kept unchanged for compatibility with the existing workspace.
- The notebooks are exploratory. Results may depend on cell execution order if a notebook has not been run from top to bottom.
- If you update `kit.py`, rerun dependent notebooks so imported functions and plotted results stay in sync.
- The CPPI notebook (`13_CPPI.ipynb`) relies on `run_cppi()` from `kit.py`. Ensure `kit.py` is importable from the same directory as the notebooks.
