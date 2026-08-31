"""
models/black_scholes.py
Black-Scholes Option Pricing Engine — Quant Research Terminal

Includes:
  - Call and Put pricing
  - Option Greeks (Delta, Gamma, Theta, Vega, Rho)
  - Implied Volatility solver (Newton-Raphson)
  - 3D surface data for volatility surface plots
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ──────────────────────────────────────────────
#  CORE: d1 / d2 CALCULATIONS
# ──────────────────────────────────────────────
def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """
    Compute d1 and d2 for Black-Scholes formula.
    S     : Spot price
    K     : Strike price
    T     : Time to expiry (years)
    r     : Risk-free rate (annualized, decimal)
    sigma : Volatility (annualized, decimal)
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ──────────────────────────────────────────────
#  PRICING
# ──────────────────────────────────────────────
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call option price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put option price (Put-Call Parity)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price_option(S, K, T, r, sigma, option_type="call") -> float:
    """Price either call or put."""
    if option_type.lower() == "call":
        return black_scholes_call(S, K, T, r, sigma)
    return black_scholes_put(S, K, T, r, sigma)


# ──────────────────────────────────────────────
#  GREEKS
# ──────────────────────────────────────────────
def compute_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type="call") -> dict:
    """
    Compute all five option Greeks.

    Returns
    -------
    dict with keys: delta, gamma, theta, vega, rho
    """
    d1, d2   = _d1_d2(S, K, T, r, sigma)
    disc     = np.exp(-r * T)
    phi_d1   = norm.pdf(d1)   # Standard normal PDF at d1
    sqrt_T   = np.sqrt(T)

    # Delta
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * phi_d1 * sigma) / (2 * sqrt_T)
            - r * K * disc * norm.cdf(d2)
        ) / 365
        rho = K * T * disc * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * phi_d1 * sigma) / (2 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
        ) / 365
        rho = -K * T * disc * norm.cdf(-d2) / 100

    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega  = S * phi_d1 * sqrt_T / 100  # Per 1% change in vol

    return {
        "delta" : round(delta, 4),
        "gamma" : round(gamma, 6),
        "theta" : round(theta, 4),
        "vega"  : round(vega, 4),
        "rho"   : round(rho, 4)
    }


# ──────────────────────────────────────────────
#  IMPLIED VOLATILITY (Brent's Method)
# ──────────────────────────────────────────────
def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6
) -> float:
    """
    Solve for implied volatility from observed market price.
    Uses Brent's method — robust and always converges.

    Returns IV as decimal (e.g., 0.25 = 25% vol).
    Returns NaN if no solution found.
    """
    pricing_fn = black_scholes_call if option_type.lower() == "call" else black_scholes_put

    try:
        iv = brentq(
            lambda sigma: pricing_fn(S, K, T, r, sigma) - market_price,
            a=1e-4,
            b=10.0,
            xtol=tol
        )
        return round(iv, 6)
    except ValueError:
        return float("nan")


# ──────────────────────────────────────────────
#  VOLATILITY SURFACE DATA
# ──────────────────────────────────────────────
def generate_vol_surface(
    S: float     = 100.0,
    r: float     = 0.05,
    sigma: float = 0.20,
    option_type: str = "call"
) -> dict:
    """
    Generate data for a 3D vol surface (strike × expiry → price).

    Returns dict with:
      strikes  : 1D array
      expiries : 1D array
      prices   : 2D array (len(strikes) × len(expiries))
    """
    strikes  = np.linspace(S * 0.7, S * 1.3, 30)
    expiries = np.linspace(1/52, 2.0, 30)   # 1 week to 2 years
    K_grid, T_grid = np.meshgrid(strikes, expiries)

    prices = np.vectorize(price_option)(S, K_grid, T_grid, r, sigma, option_type)

    return {
        "strikes"  : strikes,
        "expiries" : expiries,
        "K_grid"   : K_grid,
        "T_grid"   : T_grid,
        "prices"   : prices
    }


# ──────────────────────────────────────────────
#  FULL PRICING REPORT
# ──────────────────────────────────────────────
def run_bs_report(S, K, T, r, sigma) -> dict:
    """
    Return complete Black-Scholes analysis for both call and put.
    """
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price  = black_scholes_put(S, K, T, r, sigma)
    d1, d2     = _d1_d2(S, K, T, r, sigma)
    call_greeks = compute_greeks(S, K, T, r, sigma, "call")
    put_greeks  = compute_greeks(S, K, T, r, sigma, "put")

    moneyness = "ATM" if abs(S - K) < 0.5 else ("ITM" if S > K else "OTM")

    return {
        "inputs": {
            "S": S, "K": K, "T": round(T, 4),
            "r": r, "sigma": sigma, "moneyness": moneyness
        },
        "call": {
            "price"  : round(call_price, 4),
            "greeks" : call_greeks,
            "d1"     : round(d1, 4),
            "d2"     : round(d2, 4)
        },
        "put": {
            "price"  : round(put_price, 4),
            "greeks" : put_greeks,
            "d1"     : round(d1, 4),
            "d2"     : round(d2, 4)
        },
        "put_call_parity_check": round(
            call_price - put_price - (S - K * np.exp(-r * T)), 6
        )  # Should ≈ 0
    }


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import json
    S, K, T, r, sigma = 150.0, 155.0, 0.25, 0.05, 0.22
    report = run_bs_report(S, K, T, r, sigma)
    print("=== Black-Scholes Report ===")
    print(json.dumps(report, indent=2))

    print("\n=== Implied Volatility Test ===")
    mkt_price = black_scholes_call(S, K, T, r, 0.22)
    iv = implied_volatility(mkt_price, S, K, T, r, "call")
    print(f"Market Price: {mkt_price:.4f} | Recovered IV: {iv:.4f} (expected: 0.22)")
