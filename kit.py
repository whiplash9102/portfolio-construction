import pandas as pd
import scipy.stats
from scipy.stats import norm
import numpy as np


def drawdown(return_series : pd.Series):
    """
    Take a time series of asset return
    Computed and return a Dataframe that consist:
    the wealth index
    the previous peak
    drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peak = wealth_index.cummax()
    drawdown = (wealth_index/previous_peak) - 1
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peak,
        "Drawdown": drawdown
    })

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund index returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates =True)
    
    hfi = hfi / 100
    hfi.index= hfi.index.to_period("M")
    return hfi

def get_ind_returns():
    """
    Load and format the retunr of each stock in each industry
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0,parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def skewness(r):
    """
    Calculate the skewness of dataset
    """
    demean_r= r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demean_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Calculate the kurtosis
    """
    demean_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demean_r**4).mean()
    return exp/sigma_r**4

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

def var_gaussian(r, level=5, modified = False):
    """
    Retun the parametric Gaussian VaR of a Series or DataFrame
    If modified is True, then the modified VaR is returned,
    Using the Cornish-Fisher modification
    """

    z  = norm.ppf(level/100)
    if modified:
        # modify Z score based on the sknewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z= (z + 
            (z**2 - 1)*s/6 +
            (z**3 - 3*z)*(k-3)/24-
            (2*z**3-5*z)*(s**2)/36
        )
    return -(r.mean() + z*r.std(ddof=0))

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
    return_per_interval = (1+r).prod()**(1/interval) - 1
    return (return_per_interval+1)**period - 1 

def annual_volatility_by_month(r):
    return r.std()*np.sqrt(12)

def annual_return(r, period):
    """
    Calculate the compounded annual return
    """
    compounded_growth = (1+r).prod()
    n_period = r.shape[0]
    return compounded_growth**(period/n_period) - 1


def annual_volatility(r, period):
    """"
    Annualize volatity of a set of returns 
    """
    return r.std()*(period**0.5)

def sharpe_ratio(r, risk_free_rate, period):
    """
    Calculate the annualized sharp ratio of set of retunrs
    """
    rf_per_period = (1+risk_free_rate)**(1/period) - 1
    excess_ret = r- rf_per_period
    ann_ex_ret = annual_return(excess_ret, period)
    annual_vol = annual_volatility(r, period)
    return ann_ex_ret/annual_vol

def portfolio_return(weights, er):
    """
    Calculate the return of a portfolio given weights and expected returns
    """
    return weights.T @ er


def portfolio_vol(weights, cov):
    """
    Calculate the volatility of a portfolio given weights and covariance matrix
    """
    return (weights.T @ cov @ weights) **0.5