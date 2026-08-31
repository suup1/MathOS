"""
models/regime.py
Market Regime Detection — Quant Research Terminal

Uses Hidden Markov Models (HMM) to identify latent market regimes
(e.g., Bull / Bear / High Volatility) from return/volatility data.

Includes:
  - 2-state and 3-state HMM fitting
  - Regime labeling (Bull / Bear)
  - Regime statistics (mean, vol per state)
  - Forward regime probability (current state)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
#  FEATURE ENGINEERING
# ──────────────────────────────────────────────
def build_features(prices: pd.Series) -> pd.DataFrame:
    """
    Build feature matrix for HMM from price series.

    Features:
      - log_return       : Daily log return
      - rolling_vol_5    : 5-day realized volatility
      - rolling_vol_21   : 21-day realized volatility
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    df = pd.DataFrame({"log_return": log_returns})
    df["rolling_vol_5"]  = df["log_return"].rolling(5).std()
    df["rolling_vol_21"] = df["log_return"].rolling(21).std()
    df.dropna(inplace=True)

    return df


# ──────────────────────────────────────────────
#  FIT HMM
# ──────────────────────────────────────────────
def fit_hmm(
    prices: pd.Series,
    n_states: int = 2,
    n_iter: int   = 1000,
    features: list = None
) -> dict:
    """
    Fit a Gaussian HMM to price-derived features.

    Parameters
    ----------
    prices   : Raw price series
    n_states : Number of hidden states (2 = Bull/Bear, 3 = Bull/Bear/Crisis)
    n_iter   : Max EM iterations
    features : List of feature column names to use (default: all)

    Returns
    -------
    dict with model, states, regime_labels, dates, statistics
    """
    feat_df  = build_features(prices)
    feat_cols = features or ["log_return", "rolling_vol_5"]
    X         = feat_df[feat_cols].values

    model = GaussianHMM(
        n_components    = n_states,
        covariance_type = "full",
        n_iter          = n_iter,
        random_state    = 42
    )
    model.fit(X)
    hidden_states = model.predict(X)

    # ── Label states by their mean log return (Bull = highest mean)
    state_means = [X[hidden_states == s, 0].mean() for s in range(n_states)]
    sorted_states = np.argsort(state_means)  # ascending

    if n_states == 2:
        label_map = {sorted_states[0]: "Bear", sorted_states[1]: "Bull"}
    else:
        label_map = {
            sorted_states[0]: "Crisis",
            sorted_states[1]: "Neutral",
            sorted_states[2]: "Bull"
        }

    regime_labels = pd.Series(
        [label_map[s] for s in hidden_states],
        index=feat_df.index
    )

    # ── Statistics per state
    stats = {}
    for s in range(n_states):
        mask       = hidden_states == s
        rets       = X[mask, 0]
        stats[label_map[s]] = {
            "mean_daily_return" : round(rets.mean() * 100, 4),    # %
            "daily_vol"         : round(rets.std() * 100, 4),     # %
            "annualized_vol"    : round(rets.std() * np.sqrt(252) * 100, 2),
            "n_days"            : int(mask.sum()),
            "pct_of_time"       : round(mask.sum() / len(mask) * 100, 1)
        }

    # ── Transition matrix
    trans_mat = pd.DataFrame(
        np.exp(model.transmat_) if hasattr(model, '_transmat_') else model.transmat_,
        index   = [label_map[i] for i in range(n_states)],
        columns = [label_map[i] for i in range(n_states)]
    ).round(4)

    # ── Current state
    last_prob = model.predict_proba(X)[-1]
    current_probs = {label_map[s]: round(last_prob[s], 4) for s in range(n_states)}
    current_state = max(current_probs, key=current_probs.get)

    return {
        "model"           : model,
        "n_states"        : n_states,
        "hidden_states"   : hidden_states,
        "regime_labels"   : regime_labels,
        "dates"           : feat_df.index,
        "prices_aligned"  : prices.loc[feat_df.index],
        "log_returns"     : feat_df["log_return"],
        "stats"           : stats,
        "transition_matrix": trans_mat,
        "current_state"   : current_state,
        "current_probs"   : current_probs,
        "label_map"       : label_map,
        "score"           : round(model.score(X), 2)   # Log-likelihood
    }


# ──────────────────────────────────────────────
#  REGIME COLOR MAP (for plotting)
# ──────────────────────────────────────────────
REGIME_COLORS = {
    "Bull"    : "#00C896",  # Green
    "Bear"    : "#FF4F5A",  # Red
    "Crisis"  : "#FF8C00",  # Orange
    "Neutral" : "#6B8EAD",  # Steel blue
}


def get_regime_spans(regime_labels: pd.Series) -> list:
    """
    Convert regime label series to list of (start, end, regime) spans.
    Used for shading background in price charts.
    """
    spans = []
    prev_regime = None
    start       = None

    for date, regime in regime_labels.items():
        if regime != prev_regime:
            if prev_regime is not None:
                spans.append((start, date, prev_regime))
            start       = date
            prev_regime = regime

    if start is not None:
        spans.append((start, regime_labels.index[-1], prev_regime))

    return spans


# ──────────────────────────────────────────────
#  ROLLING REGIME DETECTION
# ──────────────────────────────────────────────
def rolling_regime_detect(
    prices: pd.Series,
    window: int    = 126,
    n_states: int  = 2
) -> pd.Series:
    """
    Detect regime using a rolling window — captures structural changes.
    Slower but more adaptive than fitting on full history.
    Returns Series of regime labels (only available after first window).
    """
    labels = {}

    for i in range(window, len(prices)):
        window_prices = prices.iloc[i - window:i]
        try:
            result = fit_hmm(window_prices, n_states=n_states)
            labels[prices.index[i]] = result["current_state"]
        except Exception:
            labels[prices.index[i]] = "Unknown"

    return pd.Series(labels)


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a regime-switching price series
    np.random.seed(7)
    n = 500
    bull_returns = np.random.normal(0.0008, 0.010, n // 2)
    bear_returns = np.random.normal(-0.0005, 0.020, n // 2)
    all_returns  = np.concatenate([bull_returns, bear_returns])
    prices       = pd.Series(100 * np.exp(np.cumsum(all_returns)))

    print("=== Regime Detection ===")
    result = fit_hmm(prices, n_states=2)

    print(f"Current Regime  : {result['current_state']}")
    print(f"Current Probs   : {result['current_probs']}")
    print(f"Log-Likelihood  : {result['score']}")
    print("\nRegime Statistics:")
    for regime, s in result["stats"].items():
        print(f"  [{regime}] Mean: {s['mean_daily_return']}% | Vol: {s['annualized_vol']}% | Days: {s['n_days']}")
    print("\nTransition Matrix:")
    print(result["transition_matrix"])
