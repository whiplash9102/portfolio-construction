# Portfolio Construction Toolkit

Python and Jupyter notebook project for learning quantitative portfolio construction.

This repository starts with basic return and risk measurement, then builds toward efficient frontiers, portfolio optimization, CPPI, Monte Carlo simulation, CIR interest-rate modeling, liability-driven investing, and dynamic risk budgeting.

The notebooks are grouped by topic in `notebooks/`. Reusable finance functions live in `kit.py`, and the datasets used in the exercises are included in `data/`.

## Contents

- [Quick Start](#quick-start)
- [Learning Path](#learning-path)
- [Notebook Index](#notebook-index)
- [Core Module](#core-module)
- [Sample Data](#sample-data)
- [Project Structure](#project-structure)
- [Notes](#notes)

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy scipy matplotlib notebook ipywidgets
```

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open the notebooks from the `notebooks/` folder. Each notebook includes a setup cell that finds the repository root, adds it to `sys.path`, and switches the working directory to the root so `kit.py` and `data/` load correctly.

## Learning Path

The notebooks are designed to be read in numbered order.

```text
Return and risk measurement
-> Efficient frontiers
-> Portfolio optimization
-> Diversification
-> Dynamic strategies
-> Liability-driven investing
-> Interest-rate modeling
-> Risk budgeting
-> Execution
```

## Notebook Index

### 01 - Risk Metrics

Foundational return, drawdown, distribution, and tail-risk analysis.

- [1_RiskAjustedReturns.ipynb](notebooks/01_risk_metrics/1_RiskAjustedReturns.ipynb) - Return calculations, compounding, and basic performance measurement
- [2_ComputeDrawDown.ipynb](notebooks/01_risk_metrics/2_ComputeDrawDown.ipynb) - Wealth index construction, running peaks, and drawdown analysis
- [3_BuildModules.ipynb](notebooks/01_risk_metrics/3_BuildModules.ipynb) - Refactoring repeated notebook logic into reusable Python functions
- [4_DeviationsFromNormality.ipynb](notebooks/01_risk_metrics/4_DeviationsFromNormality.ipynb) - Skewness, kurtosis, and Jarque-Bera normality testing
- [5_VaR_CVaR.ipynb](notebooks/01_risk_metrics/5_VaR_CVaR.ipynb) - Semideviation, VaR, Cornish-Fisher VaR, and CVaR

### 02 - Efficient Frontier

Core portfolio construction concepts using two-asset and multi-asset portfolios.

- [6_Test_Market.ipynb](notebooks/02_efficient_frontier/6_Test_Market.ipynb) - Applied exercises using portfolio and return datasets
- [7_Efficient_Frontier.ipynb](notebooks/02_efficient_frontier/7_Efficient_Frontier.ipynb) - Portfolio return, volatility, covariance, and efficient frontier basics
- [8_Asset_efficient_frontier.ipynb](notebooks/02_efficient_frontier/8_Asset_efficient_frontier.ipynb) - Two-asset efficient frontier plotting
- [9_N_Asset_Efficient_Frontier.ipynb](notebooks/02_efficient_frontier/9_N_Asset_Efficient_Frontier.ipynb) - Long-only mean-variance optimization

### 03 - Optimization

Optimization methods for selecting portfolio weights.

- [10_Max_Sharp_Ratio.ipynb](notebooks/03_optimization/10_Max_Sharp_Ratio.ipynb) - Tangency portfolio and maximum Sharpe ratio optimization
- [11_GMV.ipynb](notebooks/03_optimization/11_GMV.ipynb) - Global Minimum Variance portfolio construction

### 04 - Diversification

Market-cap weighting and the practical limits of diversification.

- [12_Limits_of_Diversification.ipynb](notebooks/04_diversification/12_Limits_of_Diversification.ipynb) - Cap-weighted market index construction and industry-level diversification analysis

### 05 - Dynamic Strategies

Portfolio insurance and simulation-based strategy testing.

- [13_CPPI.ipynb](notebooks/05_dynamic_strategies/13_CPPI.ipynb) - Constant Proportion Portfolio Insurance with dynamic floor-based allocation
- [14_Random_Walk.ipynb](notebooks/05_dynamic_strategies/14_Random_Walk.ipynb) - Geometric Brownian Motion simulations
- [15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb](notebooks/05_dynamic_strategies/15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb) - Interactive Monte Carlo scenario exploration

### 06 - Liability Driven Investing

Funding ratios, stochastic interest rates, bond pricing, and duration matching.

- [16_PV_Liabilities_Funding_Ratio.ipynb](notebooks/06_liability_driven/16_PV_Liabilities_Funding_Ratio.ipynb) - Present value of liabilities and funding ratio analysis
- [17_CIR_Model_Interest_Rate_Liability_Hedging.ipynb](notebooks/06_liability_driven/17_CIR_Model_Interest_Rate_Liability_Hedging.ipynb) - CIR interest-rate modeling and zero-coupon bond pricing
- [18_GHP_Construction_Duration_Matching.ipynb](notebooks/06_liability_driven/18_GHP_Construction_Duration_Matching.ipynb) - Goal-Hedging Portfolio construction and duration matching
- [19_Monte_Carlo_w_CIR.ipynb](notebooks/06_liability_driven/19_Monte_Carlo_w_CIR.ipynb) - Monte Carlo bond-price simulation under CIR rates

### 07 - Risk Budgeting

Dynamic allocation rules for managing floors, drawdowns, and terminal outcomes.

- [20_Risk_Budgeting_Strategies.ipynb](notebooks/07_risk_budgeting/20_Risk_Budgeting_Strategies.ipynb) - Fixed-mix allocation, terminal wealth, and portfolio insurance strategies
- [21_Dynamic_Risk_Budgeting.ipynb](notebooks/07_risk_budgeting/21_Dynamic_Risk_Budgeting.ipynb) - Glide paths, floor allocation, and drawdown-constrained allocation
- [22_Execution.ipynb](notebooks/07_risk_budgeting/22_Execution.ipynb) - Execution-oriented risk budgeting exercises

## Core Module

`kit.py` contains the reusable functions used throughout the notebooks.

Main function groups:

- Data loading: `get_hfi_returns()`, `get_ind_returns()`, `get_ind_size()`, `get_ind_nfirms()`, `get_total_market_index_return()`
- Risk statistics: `drawdown()`, `skewness()`, `kurtosis()`, `is_normal()`, `semideviation()`
- Tail risk: `var_historic()`, `var_gaussian()`, `cvar_historic()`
- Annualization and portfolio math: `annual_return()`, `annual_volatility()`, `sharpe_ratio()`, `portfolio_return()`, `portfolio_vol()`
- Optimization: `minimize_vol()`, `optimal_weights()`, `msr()`, `gmv()`, `plot_ef()`, `plot_ef2()`
- Simulation and dynamic strategies: `run_cppi()`, `gbm()`, `summary_stats()`
- Interest rates and bonds: `discount()`, `pv()`, `funding_ratio()`, `cir()`, `bond_cash_flows()`, `bond_price()`, `macaulay_duration()`, `match_duration()`, `bond_total_return()`
- Risk budgeting: `bt_mix()`, `fixedmix_allocator()`, `terminal_values()`, `terminal_stats()`, `glide_path_allocator()`, `floor_allocator()`, `drawdown_allocator()`

Example:

```python
import kit as erk

hfi = erk.get_hfi_returns()

drawdowns = erk.drawdown(hfi["Convertible Arbitrage"])
stats = erk.summary_stats(hfi)

cppi = erk.run_cppi(
    risky_r=hfi["Convertible Arbitrage"],
    m=3,
    floor=0.8,
)

prices = erk.gbm(
    n_years=10,
    n_scenarios=1000,
    mu=0.07,
    sigma=0.15,
)

rates, zcb_prices = erk.cir(
    n_years=10,
    n_scenarios=50,
    a=0.05,
    b=0.03,
    sigma=0.05,
)
```

## Sample Data

The `data/` directory contains the datasets used in the notebooks:

- EDHEC hedge fund index returns
- Fama-French factor datasets, monthly and daily
- 30-industry and 49-industry return series
- Industry-level firm counts and average firm sizes
- Size-sorted portfolio returns
- Sample price series and individual stock data

All data loaders in `kit.py` expect these files to remain in `data/` with their current filenames.

## Project Structure

```text
.
|-- data/
|   `-- *.csv
|-- notebooks/
|   |-- 01_risk_metrics/
|   |-- 02_efficient_frontier/
|   |-- 03_optimization/
|   |-- 04_diversification/
|   |-- 05_dynamic_strategies/
|   |-- 06_liability_driven/
|   `-- 07_risk_budgeting/
|-- kit.py
|-- README.md
`-- .gitignore
```

## Notes

- The notebooks are exploratory. If cells are run out of order, results may differ. Use **Restart & Run All** when you need reproducible output.
- If you update `kit.py`, restart the kernel in any dependent notebook.
- Python cache files, local Claude settings, Jupyter checkpoints, and `.DS_Store` files are intentionally ignored by Git.
