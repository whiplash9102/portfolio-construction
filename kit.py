import pandas as pd
import scipy.stats
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
import math


def drawdown(return_series: pd.Series):
    """
    Take a time series of asset return
    Computed and return a Dataframe that consist:
    the wealth index
    the previous peak
    drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peak = wealth_index.cummax()
    drawdown = (wealth_index / previous_peak) - 1
    return pd.DataFrame(
        {"Wealth": wealth_index, "Peaks": previous_peak, "Drawdown": drawdown}
    )


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund index returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0)

    hfi = hfi / 100
    hfi.index = pd.to_datetime(hfi.index).to_period("M")
    return hfi


def get_ind_size():
    """
    Load and format the size of each stock in each industry
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    """ """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_returns():
    """
    Load and format the retunr of each stock in each industry
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_total_market_index_return():
    """
    Compute the cap-weighted total market index return across all 30 industries.

    The total market cap for each industry is estimated as:
        market_cap = number_of_firms × average_firm_size

    These market caps are then used as weights to compute a value-weighted
    (cap-weighted) blended return across all industries each month.

    Returns
    -------
    pd.Series
        Monthly total market index returns, indexed by Period (monthly).
    """
    ind_returns = get_ind_returns()  # monthly VW returns per industry
    ind_size = get_ind_size()  # avg firm size per industry (in $M)
    ind_nfirms = get_ind_nfirms()  # number of firms per industry

    # Total market cap per industry each month
    ind_mktcap = ind_size * ind_nfirms

    # Align on the common date range
    common_idx = ind_returns.index.intersection(ind_mktcap.index)
    ind_returns = ind_returns.loc[common_idx]
    ind_mktcap = ind_mktcap.loc[common_idx]

    # Cap-weights: each industry's share of total market cap that month
    total_mktcap = ind_mktcap.sum(axis=1)  # pd.Series
    cap_weights = ind_mktcap.divide(total_mktcap, axis=0)  # DataFrame

    # Weighted return = sum(weight_i * return_i) across industries
    total_market_return = (cap_weights * ind_returns).sum(axis=1)
    total_market_return.name = "Total Market"
    return total_market_return


def skewness(r):
    """
    Calculate the skewness of dataset
    """
    demean_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demean_r**3).mean()
    return exp / sigma_r**3


def kurtosis(r):
    """
    Calculate the kurtosis
    """
    demean_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demean_r**4).mean()
    return exp / sigma_r**4


def is_normal(r, level=0.01):
    """
    Apply the Jarque-Bera test to determine if a Series a normal or not
    Test is apply at the 1% level by default
    Retunr True if the hypothesis normally approved
    """

    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def semideviation(r):
    """
    Return the semideviation aka negative semideviation of r
    r must be a Series of DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def var_historic(r, level=5):
    """
    VaR historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expect r to be Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Retun the parametric Gaussian VaR of a Series or DataFrame
    If modified is True, then the modified VaR is returned,
    Using the Cornish-Fisher modification
    """

    z = norm.ppf(level / 100)
    if modified:
        # modify Z score based on the sknewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Compute conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        var = var_historic(r, level=level)
        beyond_r = r <= -var
        return -r[beyond_r].mean()
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def annual_return_per_month(r, period=12):
    """
    Calculate the annual return
    """
    interval = r.shape[0]
    return_per_interval = (1 + r).prod() ** (1 / interval) - 1
    return (return_per_interval + 1) ** period - 1


def annual_volatility_by_month(r):
    return r.std() * np.sqrt(12)


def annual_return(r, period):
    """
    Calculate the compounded annual return
    """
    compounded_growth = (1 + r).prod()
    n_period = r.shape[0]
    return compounded_growth ** (period / n_period) - 1


def annual_volatility(r, period):
    """ "
    Annualize volatity of a set of returns
    """
    return r.std() * (period**0.5)


def sharpe_ratio(r, risk_free_rate, period):
    """
    Calculate the annualized sharp ratio of set of retunrs
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / period) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annual_return(excess_ret, period)
    annual_vol = annual_volatility(r, period)
    return ann_ex_ret / annual_vol


def portfolio_return(weights, er):
    """
    Calculate the return of a portfolio given weights and expected returns
    """
    return weights.T @ er


def portfolio_vol(weights, cov):
    """
    Calculate the volatility of a portfolio given weights and covariance matrix
    """
    return (weights.T @ cov @ weights) ** 0.5


def plot_ef2(npoints, er, cov, style=".-"):
    """
    Plot 2 asset with efficient frontier
    """
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only pot 2 assets frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, npoints)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)


# def discount(t, r):
#     """
#     Compute the price of pure discount bond that pay a dollar a time t, given interest rate r
#     """
#     return (1 + r) ** (-t)


def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t, given an interest rate r per
    period. Returns a |t| x |r| Series or DataFrame. r can be a float, Series, or DataFrame.
    Returns a DataFrame indexed by t.
    """
    discounts = pd.DataFrame([(r + 1) ** (-i) for i in t], index=t)
    return discounts


# # def pv(l, r):
# #     """
# #     Compute the present value of a sequence of liabilities
# #     l is indexexed by the time, and the values are the amount of each liability
# #     retun present value of the sequences
# #     """

# #     dates = l.index
# #     discounts = discount(dates, r)
#     return (discounts * l).sum()


def pv(flows, r):
    """
    Compute the present value of a sequence of flows given by the time as an index and amount. r can be
    a scalar, or a Series or DataFrame with the number of rows matching the number of rows in flows.
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis="rows").sum()


def minimize_vol(target_return, er, cov):
    """Minimize portfolio volatility for a given target return."""
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n

    min_ret = float(er.min())
    max_ret = float(er.max())
    if not (min_ret <= target_return <= max_ret):
        raise ValueError(
            f"target_return={target_return:.6f} is outside feasible long-only range "
            f"[{min_ret:.6f}, {max_ret:.6f}]"
        )

    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er),
    }

    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    result = minimize(
        portfolio_vol,
        init_guess,
        args=(cov,),
        method="SLSQP",
        options={"disp": False},
        constraints=(return_is_target, weights_sum_to_1),
        bounds=bounds,
    )

    if not result.success:
        raise ValueError(f"SLSQP failed: {result.message}")

    return result.x


def optimal_weights(n_points, er, cov):
    """
    --> List of weight to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate, er, cov):
    """Return the weights of the portfolio that gives the maximum Sharpe ratio"""

    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n

    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """Negative Sharpe ratio (because we minimize)"""
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(riskfree_rate, er, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=(weights_sum_to_1,),
        options={"disp": False},
    )

    return result.x


def gmv(cov):
    """
    Returns the weight of the Global Minimum Vol Portfolio
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def plot_ef(
    npoints,
    er,
    cov,
    style=".-",
    show_cml=False,
    riskfree_rate=0.1,
    show_ew=False,
    show_gmv=False,
):
    """Plot N-asset efficient frontier"""
    weights = optimal_weights(npoints, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display gmv
        ax.plot([vol_gmv], [r_gmv], marker="o", color="midnightblue", markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display Ew
        ax.plot([vol_ew], [r_ew], marker="o", color="black", markersize=10)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add cml
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", linestyle="dashed")
    return ax


def run_cppi(
    risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawDown=None
):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary of containing: Asset Value History, Risky Asset Weight History, Cushion History
    """

    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = risky_r.to_frame("R")

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate / 12  # fill all rows with the risk-free rate

    # set up the tracking DataFrame
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawDown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawDown)

        cushion = (account_value - floor_value) / account_value
        risky_weight = m * cushion
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)

        safe_weight = 1 - risky_weight
        risky_alloc = account_value * risky_weight
        safe_alloc = account_value * safe_weight

        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (
            1 + safe_r.iloc[step]
        )
        account_history.iloc[step] = account_value
        risky_w_history.iloc[step] = risky_weight
        cushion_history.iloc[step] = cushion

    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Weight": risky_w_history,
        "Cushion": cushion_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r,
    }

    return backtest_result


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains the annualized return, volatility, Sharpe ratio,
    max drawdown, and Sharpe ratio of the given returns.
    """
    ann_r = r.aggregate(annual_return, period=12)
    ann_vol = r.aggregate(annual_volatility, period=12)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate=riskfree_rate, period=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    kurt = r.aggregate(lambda r: kurtosis(r))
    skew = r.aggregate(lambda r: skewness(r))
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)

    return pd.DataFrame(
        {
            "Annualized Return": ann_r,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd,
            "Kurtosis": kurt,
            "Skew": skew,
            "Sharp Ratio": ann_sr,
            "CVaR 5% (Gaussian) ": cf_var5,
            "Historic CVaR 5%": hist_cvar5,
        }
    )


def gbm(
    n_years=10,
    n_scenarios=1000,
    mu=0.07,
    sigma=0.15,
    steps_per_year=12,
    s_0=100.0,
    prices=True,
):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(
        loc=(1 + mu) ** dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios)
    )
    rets_plus_1[0] = 1
    ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
    return ret_val


def inst_to_ann(r):
    """
    Convert short rate to annualized rate
    """
    return np.expm1(r)


def ann_to_inst(r):
    """
    Convert annualized rate to short rate
    """
    return np.log1p(r)


def funding_ratio(assets, liabilities, r):
    """
    Compute the funding ratio, defined as the ratio of the market value of assets to the market value of liabilities.
    """
    return pv(assets, r) / pv(liabilities, r)


def cir(
    n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None
):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1 / steps_per_year
    num_steps = int(n_years * steps_per_year) + 1  # because n_years might be a float

    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2 * sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = (
            (2 * h * math.exp((h + a) * ttm / 2))
            / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        ) ** (2 * a * b / sigma**2)
        _B = (2 * (math.exp(h * ttm) - 1)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        _P = _A * np.exp(-_B * r)
        return _P

    prices[0] = price(n_years, r_0)
    ####

    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years - step * dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupon_per_year=12):
    """
    Return a series of cashflows generated by a bond, indexed by coupon payment
    """

    n_coupons = round(maturity * coupon_per_year)
    coupon_payment = (principal * coupon_rate) / coupon_per_year
    coupon_times = np.arange(1, n_coupons + 1)
    cashflow = pd.Series(data=coupon_payment, index=coupon_times)
    cashflow.iloc[-1] += principal
    return cashflow


def bond_price(
    maturity, principal=100, coupon_rate=0.03, coupon_per_year=12, discount_rate=0.03
):
    """
    Return the price of a bond given the maturity, principal, coupon rate, coupon per year, and discount rate
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(
                maturity - t / coupon_per_year,
                principal,
                coupon_rate,
                coupon_per_year,
                discount_rate.loc[t],
            )
        return prices
    else:
        if maturity <= 0:
            return principal + principal * coupon_rate / coupon_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupon_per_year)
        return pv(cash_flows, discount_rate / coupon_per_year)


def mauclay_duration(cf, discount_rates):
    """
    Compute the Macaulay duration of a bond, given the cash flows and the discount rates.
    """
    dcf = discount(cf.index, r=discount_rates) * cf
    weights = dcf / dcf.sum()
    return np.average(cf.index, weights=weights)


def match_duration(cf_t, cf_s, cf_l, discount_rates):
    """
    Compute the weights for matching the duration of the portfolio to the duration of the liability.
    """

    d_t = mauclay_duration(cf_t, discount_rates)
    d_s = mauclay_duration(cf_s, discount_rates)
    d_l = mauclay_duration(cf_l, discount_rates)
    return (d_l - d_t) / (d_l - d_s)


def bond_total_return(monthly_prices, principal, coupon_rate, coupon_per_year):
    """
    Computes the total return of a bond based on monthly bond prices and coupon
    payments. Assumes that dividends are paid out at the end of the period and that
    they are reinvested in the bond.
    """
    coupons = pd.DataFrame(
        data=0, index=monthly_prices.index, columns=monthly_prices.columns
    )
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(
        12 / coupon_per_year, t_max, int(coupon_per_year * t_max / 12), dtype=int
    )
    coupons.iloc[pay_date] = (principal * coupon_rate) / coupon_per_year
    total_return = (monthly_prices + coupons) / monthly_prices.shift() - 1
    return total_return.dropna()


def bt_mix(r1, r2, allocator, **kwargs):
    """
    Run a back test by allocating a two sets of return r1 and r2 are TxN dataframes where T is the time step index and N is the number of scenarios.
    allocator is a function that take two sets of returns and allocators specifc parameters, and produces an allocation to the first portfolio as a Tx1 dataFrame
    Return a TxN DataFrameof the resulting N Portfolio scenarios
    """

    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator return weights that don''t match r1")
    r_mix = weights * r1 + (1 - weights) * r2
    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocation between the PSP and GHP across N scenarios
    PSP and GHP are TxN DataFrame that represent the returns of the PSP and GHP such that:
      each column is a scenario
      each row is the price of a timestep
    Return an TxN dataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)


def terminal_values(rets):
    """
    Return the final value of a dollor at the end of return period
    """
    return (rets + 1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Compute the terminal statistics of a series of returns
    """
    terminal_wealth = terminal_values(rets)

    breach = terminal_wealth < floor
    p_breach = breach.mean() if breach.sum() > 0 else 0

    reach = terminal_wealth >= cap

    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else 0
    e_surplus = (terminal_wealth[reach] - cap).mean() if reach.sum() > 0 else 0

    stats = {
        "mean": terminal_wealth.mean(),
        "stdev": terminal_wealth.std(),
        "min": terminal_wealth.min(),
        "max": terminal_wealth.max(),
        "p_breach": p_breach,
        "e_short": e_short,
        "e_surplus": e_surplus,
    }

    sum_stats = pd.DataFrame.from_dict(stats, orient="index", columns=[name])

    return sum_stats


def glide_path_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulate a target date-fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_cols = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path] * n_cols, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns

    return paths
