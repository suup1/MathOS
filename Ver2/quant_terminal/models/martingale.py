"""
models/martingale.py
Martingale Property Testing — Quant Research Terminal

Tests whether a price series satisfies the martingale property:
    E[ X(t+1) | F(t) ] = X(t)

Includes:
  - Mean return test (basic martingale check)
  - Augmented Dickey-Fuller test (unit root / random walk)
  - Variance ratio test (Lo-MacKinlay)
  - Simulation of ideal martingale paths
"""

import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────
#  1. BASIC MARTINGALE TEST
# ──────────────────────────────────────────────
def test_martingale_basic(prices: pd.Series) -> dict:
    """
    Test martingale property via mean return ≈ 0.
    A martingale has zero expected return (drift).

    Returns dict with mean, std, t-stat, p-value, and verdict.
    """
    returns = prices.pct_change().dropna()
    n       = len(returns)
    mean    = returns.mean()
    std     = returns.std()
    se      = std / np.sqrt(n)
    t_stat  = mean / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    verdict = "MARTINGALE (fail to reject H0)" if p_value > 0.05 else "NOT MARTINGALE (reject H0)"

    return {
        "mean_return" : round(mean, 6),
        "std_return"  : round(std, 6),
        "t_stat"      : round(t_stat, 4),
        "p_value"     : round(p_value, 4),
        "n"           : n,
        "verdict"     : verdict
    }


# ──────────────────────────────────────────────
#  2. AUGMENTED DICKEY-FULLER TEST
# ──────────────────────────────────────────────
def test_adf(prices: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for unit root (random walk).
    Random walk (unit root present) is consistent with martingale.

    Requires: statsmodels
    """
    from statsmodels.tsa.stattools import adfuller

    result  = adfuller(prices.dropna(), autolag="AIC")
    verdict = (
        "UNIT ROOT PRESENT → consistent with random walk / martingale"
        if result[1] > 0.05
        else "UNIT ROOT REJECTED → price is mean-reverting"
    )

    return {
        "adf_stat"   : round(result[0], 4),
        "p_value"    : round(result[1], 4),
        "lags_used"  : result[2],
        "critical_1%": result[4]["1%"],
        "critical_5%": result[4]["5%"],
        "verdict"    : verdict
    }


# ──────────────────────────────────────────────
#  3. VARIANCE RATIO TEST (Lo-MacKinlay, 1988)
# ──────────────────────────────────────────────
def variance_ratio_test(prices: pd.Series, q: int = 4) -> dict:
    """
    Lo-MacKinlay Variance Ratio Test.
    Under martingale: VR(q) = 1.

    q : holding period (default = 4)
    """
    log_prices = np.log(prices.dropna().values)
    n          = len(log_prices)

    # 1-period returns
    ret1 = np.diff(log_prices)
    mu   = ret1.mean()

    # q-period returns
    retq = log_prices[q:] - log_prices[:-q]

    # Variances
    var1 = np.sum((ret1 - mu) ** 2) / (n - 2)
    varq = np.sum((retq - q * mu) ** 2) / (n - q - 1)

    vr      = varq / (q * var1)
    z_stat  = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    verdict = (
        "FAIL TO REJECT martingale (VR ≈ 1)"
        if p_value > 0.05
        else "REJECT martingale (VR ≠ 1)"
    )

    return {
        "q"          : q,
        "VR"         : round(vr, 4),
        "z_stat"     : round(z_stat, 4),
        "p_value"    : round(p_value, 4),
        "verdict"    : verdict
    }


# ──────────────────────────────────────────────
#  4. SIMULATE MARTINGALE PATHS (Monte Carlo)
# ──────────────────────────────────────────────
def simulate_martingale(
    S0: float     = 100.0,
    sigma: float  = 0.02,
    n_steps: int  = 252,
    n_paths: int  = 10,
    seed: int     = 42
) -> pd.DataFrame:
    """
    Simulate Geometric Brownian Motion with zero drift (martingale).
    S(t+1) = S(t) * exp(sigma * Z)  where Z ~ N(0,1)

    Returns DataFrame of shape (n_steps, n_paths).
    """
    rng  = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = sigma * shocks - 0.5 * sigma ** 2  # Ito correction for zero drift
    log_prices  = np.cumsum(log_returns, axis=0)
    prices      = S0 * np.exp(log_prices)
    prices      = np.vstack([np.full(n_paths, S0), prices])

    cols = [f"Path_{i+1}" for i in range(n_paths)]
    df   = pd.DataFrame(prices, columns=cols)
    df.index.name = "Step"
    return df


# ──────────────────────────────────────────────
#  5. FULL REPORT
# ──────────────────────────────────────────────
def run_martingale_report(prices: pd.Series, symbol: str = "ASSET") -> dict:
    """
    Run complete martingale analysis on a price series.
    Returns structured dict for display in GUI.
    """
    report = {
        "symbol"         : symbol,
        "n_observations" : len(prices),
        "basic_test"     : test_martingale_basic(prices),
        "adf_test"       : test_adf(prices),
        "variance_ratio" : variance_ratio_test(prices),
        "simulated_paths": simulate_martingale(S0=prices.iloc[0])
    }
    return report


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    # Simulate a random walk price series
    rw = pd.Series(100 + np.cumsum(np.random.randn(500) * 1.5))
    print("=== Martingale Analysis ===")
    report = run_martingale_report(rw, symbol="SIMULATED")
    for key, val in report.items():
        if key != "simulated_paths":
            print(f"\n[{key}]")
            print(val)
