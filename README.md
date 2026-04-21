# Portfolio Construction Toolkit

A hands-on Python toolkit for quantitative portfolio analysis — from basic return calculations all the way to dynamic allocation strategies and Monte Carlo simulation.

The project is organized as an **interactive learning workspace**: analysis lives in Jupyter notebooks, while all reusable functions are collected in a single helper module (`kit.py`). Sample datasets ship with the repository so every notebook runs out of the box.

---

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `1_RiskAjustedReturns.ipynb` | Return calculations, compounding, and basic performance measurement |
| 2 | `2_ComputeDrawDown.ipynb` | Wealth index construction, running peaks, and drawdown analysis |
| 3 | `3_BuildModules.ipynb` | Refactoring repeated logic into reusable Python functions |
| 4 | `4_DeviationsFromNormality.ipynb` | Skewness, kurtosis, and normality testing (Jarque-Bera) |
| 5 | `5_VaR_CVaR.ipynb` | Semideviation, historic VaR, Gaussian VaR, modified (Cornish-Fisher) VaR, and CVaR |
| 6 | `6_Test_Market.ipynb` | Applied exercises using portfolio and return datasets |
| 7 | `7_Efficient_Frontier.ipynb` | Portfolio return, volatility, covariance, and efficient frontier basics |
| 8 | `8_Asset_efficient_frontier.ipynb` | Two-asset efficient frontier plotting and portfolio combinations |
| 9 | `9_N_Asset_Efficient_Frontier.ipynb` | Long-only mean-variance optimization for multi-asset portfolios |
| 10 | `10_Max_Sharp_Ratio.ipynb` | Tangency portfolio / maximum Sharpe ratio optimization |
| 11 | `11_GMV.ipynb` | Global Minimum Variance portfolio construction |
| 12 | `12_Limits_of_Diversification.ipynb` | Cap-weighted total market index, diversification limits, and industry-level analysis |
| 13 | `13_CPPI.ipynb` | Constant Proportion Portfolio Insurance — dynamic floor-based allocation between a risky and a safe asset |
| 14 | `14_Random_Walk.ipynb` | Monte Carlo simulation of portfolio paths using Geometric Brownian Motion |
| 15 | `15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb` | Interactive widgets for Monte Carlo scenario exploration |
| 16 | `16_PV_Liabilities_Funding_Ratio.ipynb` | Present value of liabilities and funding ratio analysis |

Additional material lives in `Exercise/module2.ipynb`.

---

## Recommended Learning Flow

The notebooks follow a natural progression:

```
Return Measurement → Risk Diagnostics → Portfolio Construction & Optimization → Dynamic Strategies → Simulation & Liability Management
```

Work through them in numbered order for the smoothest experience.

---

## `kit.py` — The Core Module

All reusable logic is centralized in `kit.py`. Functions fall into six categories:

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
| `semideviation()` | Downside (negative) semideviation |

### Tail-Risk Measures
| Function | Purpose |
|----------|---------|
| `var_historic()` | Historical Value at Risk |
| `var_gaussian()` | Parametric Gaussian VaR (with optional Cornish-Fisher modification) |
| `cvar_historic()` | Conditional VaR (Expected Shortfall) |

### Annualization & Portfolio Analytics
| Function | Purpose |
|----------|---------|
| `annual_return()` / `annual_volatility()` | Compounded annual return and annualized volatility |
| `sharpe_ratio()` | Annualized Sharpe ratio |
| `portfolio_return()` / `portfolio_vol()` | Weighted portfolio return and volatility |

### Optimization & Visualization
| Function | Purpose |
|----------|---------|
| `minimize_vol()` | Minimize volatility for a given target return (SLSQP) |
| `optimal_weights()` | Generate efficient frontier weight vectors |
| `msr()` | Maximum Sharpe Ratio (tangency) portfolio weights |
| `gmv()` | Global Minimum Variance portfolio weights |
| `plot_ef()` / `plot_ef2()` | Plot N-asset or 2-asset efficient frontiers with optional CML, EW, and GMV overlays |

### Dynamic Strategies & Simulation
| Function | Purpose |
|----------|---------|
| `run_cppi()` | Backtest the CPPI strategy with configurable multiplier, floor, and optional drawdown-based floor |
| `gbm()` | Geometric Brownian Motion Monte Carlo price/return simulator |
| `summary_stats()` | One-call summary table: annualized return, volatility, Sharpe, max drawdown, skew, kurtosis, VaR, CVaR |

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

# Monte Carlo simulation (10 years, 1 000 paths)
prices = erk.gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15)
```

---

## Sample Data

The `data/` directory ships with mock and publicly available datasets used throughout the notebooks:

- Hedge fund index returns (monthly)
- Fama-French factor datasets (monthly & daily)
- 30-industry and 49-industry return series (equal- and value-weighted)
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

Then launch Jupyter **from the repository root** (so `kit.py` can resolve its relative `data/` paths):

```bash
jupyter notebook
```

---

## Project Structure

```text
.
├── 1_RiskAjustedReturns.ipynb
├── 2_ComputeDrawDown.ipynb
├── 3_BuildModules.ipynb
├── 4_DeviationsFromNormality.ipynb
├── 5_VaR_CVaR.ipynb
├── 6_Test_Market.ipynb
├── 7_Efficient_Frontier.ipynb
├── 8_Asset_efficient_frontier.ipynb
├── 9_N_Asset_Efficient_Frontier.ipynb
├── 10_Max_Sharp_Ratio.ipynb
├── 11_GMV.ipynb
├── 12_Limits_of_Diversification.ipynb
├── 13_CPPI.ipynb
├── 14_Random_Walk.ipynb
├── 15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb
├── 16_PV_Liabilities_Funding_Ratio.ipynb
├── BuildOwnModules/
├── Exercise/
│   └── module2.ipynb
├── data/
│   └── *.csv
├── kit.py
└── README.md
```

---

## Notes

- The notebooks are exploratory. If cells are run out of order, results may differ — always **Restart & Run All** for reproducible output.
- If you update `kit.py`, restart the kernel in any dependent notebook so the changes take effect.
- The CPPI notebook (`13_CPPI`) and Monte Carlo notebook (`15_Interactive_Plotting...`) rely on functions from `kit.py`. Make sure the module is importable from the same directory.
