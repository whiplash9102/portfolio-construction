# Portfolio Construction Toolkit

A hands-on Python toolkit for quantitative portfolio analysis, from basic return calculations to dynamic allocation strategies, Monte Carlo simulation, interest-rate modeling, and liability-driven investing.

The project is organized as an interactive learning workspace: analysis lives in Jupyter notebooks under `notebooks/`, while reusable functions are collected in `kit.py`. Sample datasets ship with the repository so the notebooks can run locally.

---

## Notebooks

| # | Notebook | Folder | Topic |
|---|----------|--------|-------|
| 1 | [`1_RiskAjustedReturns.ipynb`](notebooks/01_risk_metrics/1_RiskAjustedReturns.ipynb) | `01_risk_metrics` | Return calculations, compounding, and basic performance measurement |
| 2 | [`2_ComputeDrawDown.ipynb`](notebooks/01_risk_metrics/2_ComputeDrawDown.ipynb) | `01_risk_metrics` | Wealth index construction, running peaks, and drawdown analysis |
| 3 | [`3_BuildModules.ipynb`](notebooks/01_risk_metrics/3_BuildModules.ipynb) | `01_risk_metrics` | Refactoring repeated logic into reusable Python functions |
| 4 | [`4_DeviationsFromNormality.ipynb`](notebooks/01_risk_metrics/4_DeviationsFromNormality.ipynb) | `01_risk_metrics` | Skewness, kurtosis, and normality testing with Jarque-Bera |
| 5 | [`5_VaR_CVaR.ipynb`](notebooks/01_risk_metrics/5_VaR_CVaR.ipynb) | `01_risk_metrics` | Semideviation, historical VaR, Gaussian VaR, Cornish-Fisher VaR, and CVaR |
| 6 | [`6_Test_Market.ipynb`](notebooks/02_efficient_frontier/6_Test_Market.ipynb) | `02_efficient_frontier` | Applied exercises using portfolio and return datasets |
| 7 | [`7_Efficient_Frontier.ipynb`](notebooks/02_efficient_frontier/7_Efficient_Frontier.ipynb) | `02_efficient_frontier` | Portfolio return, volatility, covariance, and efficient frontier basics |
| 8 | [`8_Asset_efficient_frontier.ipynb`](notebooks/02_efficient_frontier/8_Asset_efficient_frontier.ipynb) | `02_efficient_frontier` | Two-asset efficient frontier plotting and portfolio combinations |
| 9 | [`9_N_Asset_Efficient_Frontier.ipynb`](notebooks/02_efficient_frontier/9_N_Asset_Efficient_Frontier.ipynb) | `02_efficient_frontier` | Long-only mean-variance optimization for multi-asset portfolios |
| 10 | [`10_Max_Sharp_Ratio.ipynb`](notebooks/03_optimization/10_Max_Sharp_Ratio.ipynb) | `03_optimization` | Tangency portfolio and maximum Sharpe ratio optimization |
| 11 | [`11_GMV.ipynb`](notebooks/03_optimization/11_GMV.ipynb) | `03_optimization` | Global Minimum Variance portfolio construction |
| 12 | [`12_Limits_of_Diversification.ipynb`](notebooks/04_diversification/12_Limits_of_Diversification.ipynb) | `04_diversification` | Cap-weighted total market index, diversification limits, and industry-level analysis |
| 13 | [`13_CPPI.ipynb`](notebooks/05_dynamic_strategies/13_CPPI.ipynb) | `05_dynamic_strategies` | Constant Proportion Portfolio Insurance with dynamic floor-based allocation |
| 14 | [`14_Random_Walk.ipynb`](notebooks/05_dynamic_strategies/14_Random_Walk.ipynb) | `05_dynamic_strategies` | Monte Carlo simulation of portfolio paths using Geometric Brownian Motion |
| 15 | [`15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb`](notebooks/05_dynamic_strategies/15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb) | `05_dynamic_strategies` | Interactive widgets for Monte Carlo scenario exploration |
| 16 | [`16_PV_Liabilities_Funding_Ratio.ipynb`](notebooks/06_liability_driven/16_PV_Liabilities_Funding_Ratio.ipynb) | `06_liability_driven` | Present value of liabilities, discount factors, and funding ratio analysis |
| 17 | [`17_CIR_Model_Interest_Rate_Liability_Hedging.ipynb`](notebooks/06_liability_driven/17_CIR_Model_Interest_Rate_Liability_Hedging.ipynb) | `06_liability_driven` | Cox-Ingersoll-Ross stochastic interest-rate model, ZCB pricing, and rate visualization |
| 18 | [`18_GHP_Construction_Duration_Matching.ipynb`](notebooks/06_liability_driven/18_GHP_Construction_Duration_Matching.ipynb) | `06_liability_driven` | Goal-Hedging Portfolio construction with bond pricing, Macaulay duration, and duration matching |
| 19 | [`19_Monte_Carlo_w_CIR.ipynb`](notebooks/06_liability_driven/19_Monte_Carlo_w_CIR.ipynb) | `06_liability_driven` | Monte Carlo simulation of bond prices under CIR stochastic interest rates |
| 20 | [`20_Risk_Budgeting_Strategies.ipynb`](notebooks/07_risk_budgeting/20_Risk_Budgeting_Strategies.ipynb) | `07_risk_budgeting` | Fixed-mix, terminal wealth, and portfolio insurance strategy analysis |
| 21 | [`21_Dynamic_Risk_Budgeting.ipynb`](notebooks/07_risk_budgeting/21_Dynamic_Risk_Budgeting.ipynb) | `07_risk_budgeting` | Glide path, floor allocator, and drawdown-constrained dynamic allocation |
| 22 | [`22_Execution.ipynb`](notebooks/07_risk_budgeting/22_Execution.ipynb) | `07_risk_budgeting` | Execution-oriented risk budgeting exercises |

---

## Recommended Learning Flow

The notebooks follow a natural progression:

```text
Return Measurement -> Risk Diagnostics -> Efficient Frontiers -> Optimization -> Diversification -> Dynamic Strategies -> Liability Management -> Interest Rate Modeling -> Duration Matching & LDI -> Risk Budgeting -> Execution
```

Work through them in numbered order for the smoothest experience.

---

## `kit.py` - The Core Module

All reusable logic is centralized in `kit.py`. Functions fall into these categories:

### Data Loading

| Function | Purpose |
|----------|---------|
| `get_hfi_returns()` | Load hedge-fund-style index returns |
| `get_ind_returns()` | Load 30-industry monthly value-weighted returns |
| `get_ind_size()` | Load average firm size per industry |
| `get_ind_nfirms()` | Load number of firms per industry |
| `get_total_market_index_return()` | Compute a cap-weighted total market index from industry data |

### Risk & Distribution Statistics

| Function | Purpose |
|----------|---------|
| `drawdown()` | Wealth index, peak tracking, and drawdown series |
| `skewness()` / `kurtosis()` | Higher-moment descriptive statistics |
| `is_normal()` | Jarque-Bera normality test |
| `semideviation()` | Downside semideviation |

### Tail-Risk Measures

| Function | Purpose |
|----------|---------|
| `var_historic()` | Historical Value at Risk |
| `var_gaussian()` | Parametric Gaussian VaR with optional Cornish-Fisher modification |
| `cvar_historic()` | Conditional VaR / Expected Shortfall |

### Annualization & Portfolio Analytics

| Function | Purpose |
|----------|---------|
| `annual_return_per_month()` / `annual_volatility_by_month()` | Monthly return and volatility annualization helpers |
| `annual_return()` / `annual_volatility()` | Compounded annual return and annualized volatility |
| `sharpe_ratio()` | Annualized Sharpe ratio |
| `portfolio_return()` / `portfolio_vol()` | Weighted portfolio return and volatility |

### Optimization & Visualization

| Function | Purpose |
|----------|---------|
| `minimize_vol()` | Minimize volatility for a target return using SLSQP |
| `optimal_weights()` | Generate efficient frontier weight vectors |
| `msr()` | Maximum Sharpe Ratio portfolio weights |
| `gmv()` | Global Minimum Variance portfolio weights |
| `plot_ef()` / `plot_ef2()` | Plot N-asset or 2-asset efficient frontiers with optional overlays |

### Dynamic Strategies & Simulation

| Function | Purpose |
|----------|---------|
| `run_cppi()` | Backtest CPPI with configurable multiplier, floor, and optional drawdown floor |
| `gbm()` | Geometric Brownian Motion Monte Carlo price/return simulator |
| `summary_stats()` | Summary table with return, volatility, Sharpe, drawdown, skew, kurtosis, VaR, and CVaR |

### Interest Rate Modeling, Bond Pricing & LDI

| Function | Purpose |
|----------|---------|
| `discount()` | Price a pure discount bond paying $1 at time `t` given rate `r` |
| `pv()` | Present value of time-indexed cash flows |
| `funding_ratio()` | Ratio of present value of assets to present value of liabilities |
| `inst_to_ann()` / `ann_to_inst()` | Convert between instantaneous and annualized interest rates |
| `cir()` | CIR stochastic interest-rate simulator returning annualized rates and zero-coupon bond prices |
| `bond_cash_flows()` | Generate coupon bond cash flow schedules |
| `bond_price()` | Price a coupon bond from cash flows and discount rates |
| `macaulay_duration()` | Compute Macaulay duration from cash flows and discount rate |
| `match_duration()` | Compute short/long bond weights that match a target duration |
| `bond_total_return()` | Compute total return from monthly bond prices and coupon reinvestment |

### Risk Budgeting & Dynamic Allocation

| Function | Purpose |
|----------|---------|
| `bt_mix()` | Backtest a mix between two return streams using an allocator |
| `fixedmix_allocator()` | Produce constant portfolio weights across scenarios |
| `terminal_values()` / `terminal_stats()` | Analyze terminal wealth, shortfall, and cap statistics |
| `glide_path_allocator()` | Simulate target-date-style allocation paths |
| `floor_allocator()` | Allocate between PSP and GHP while targeting a floor value |
| `drawdown_allocator()` | Allocate between PSP and GHP with a drawdown constraint |

### Quick Example

```python
import kit as erk

# Load sample hedge fund returns
hfi = erk.get_hfi_returns()

# Drawdown analysis
dd = erk.drawdown(hfi["Convertible Arbitrage"])

# CPPI backtest
result = erk.run_cppi(risky_r=hfi["Convertible Arbitrage"], m=3, floor=0.8)

# Summary statistics table
stats = erk.summary_stats(hfi)

# Monte Carlo simulation
prices = erk.gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15)

# CIR interest-rate simulation
rates, zcb_prices = erk.cir(n_years=10, n_scenarios=50, a=0.05, b=0.03, sigma=0.05)

# Bond pricing and duration
cf = erk.bond_cash_flows(maturity=3, principal=100, coupon_rate=0.05, coupon_per_year=2)
price = erk.bond_price(maturity=3, principal=100, coupon_rate=0.05, discount_rate=0.04)
duration = erk.macaulay_duration(cf, discount_rate=0.04 / 2)

# Dynamic allocation backtest
psp_r = erk.gbm(n_years=5, n_scenarios=100, mu=0.07, sigma=0.15, prices=False)
ghp_r = erk.gbm(n_years=5, n_scenarios=100, mu=0.03, sigma=0.05, prices=False)
mix = erk.bt_mix(psp_r, ghp_r, allocator=erk.fixedmix_allocator, w1=0.6)
```

---

## Sample Data

The `data/` directory ships with mock and publicly available datasets used throughout the notebooks:

- Hedge fund index returns
- Fama-French factor datasets, monthly and daily
- 30-industry and 49-industry return series, equal- and value-weighted
- Industry-level firm counts and average firm sizes
- Size-sorted portfolio returns
- Sample price series and individual stock data

All data loaders in `kit.py` expect these files under the local `data/` directory with their current filenames.

---

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scipy matplotlib notebook ipywidgets
```

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Then open notebooks from the `notebooks/` folder. Each notebook includes a setup cell that locates the repository root, adds it to `sys.path`, and switches the working directory to the root so `kit.py` and `data/` resolve correctly.

---

## Project Structure

```text
.
├── data/
│   └── *.csv
├── notebooks/
│   ├── 01_risk_metrics/
│   │   ├── 1_RiskAjustedReturns.ipynb
│   │   ├── 2_ComputeDrawDown.ipynb
│   │   ├── 3_BuildModules.ipynb
│   │   ├── 4_DeviationsFromNormality.ipynb
│   │   └── 5_VaR_CVaR.ipynb
│   ├── 02_efficient_frontier/
│   │   ├── 6_Test_Market.ipynb
│   │   ├── 7_Efficient_Frontier.ipynb
│   │   ├── 8_Asset_efficient_frontier.ipynb
│   │   └── 9_N_Asset_Efficient_Frontier.ipynb
│   ├── 03_optimization/
│   │   ├── 10_Max_Sharp_Ratio.ipynb
│   │   └── 11_GMV.ipynb
│   ├── 04_diversification/
│   │   └── 12_Limits_of_Diversification.ipynb
│   ├── 05_dynamic_strategies/
│   │   ├── 13_CPPI.ipynb
│   │   ├── 14_Random_Walk.ipynb
│   │   └── 15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb
│   ├── 06_liability_driven/
│   │   ├── 16_PV_Liabilities_Funding_Ratio.ipynb
│   │   ├── 17_CIR_Model_Interest_Rate_Liability_Hedging.ipynb
│   │   ├── 18_GHP_Construction_Duration_Matching.ipynb
│   │   └── 19_Monte_Carlo_w_CIR.ipynb
│   └── 07_risk_budgeting/
│       ├── 20_Risk_Budgeting_Strategies.ipynb
│       ├── 21_Dynamic_Risk_Budgeting.ipynb
│       └── 22_Execution.ipynb
├── kit.py
├── README.md
└── .gitignore
```

---

## Notes

- The notebooks are exploratory. If cells are run out of order, results may differ; use **Restart & Run All** for reproducible output.
- If you update `kit.py`, restart the kernel in any dependent notebook so the changes take effect.
- The CPPI, Monte Carlo, CIR, GHP, LDI, and risk-budgeting notebooks rely heavily on functions from `kit.py`.
- Python cache files, local Claude settings, Jupyter checkpoints, and `.DS_Store` files are intentionally ignored by Git.
