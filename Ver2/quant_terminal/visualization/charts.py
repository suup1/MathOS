"""
visualization/charts.py
Chart Engine — Quant Research Terminal

All matplotlib figure builders. Each function returns a Figure
that can be embedded into PySide6 via FigureCanvasQTAgg.

Charts:
  1. Price chart with volume
  2. Martingale simulation paths
  3. Black-Scholes 3D surface
  4. Regime detection overlay
  5. Sentiment overlay on price
  6. Greeks bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams.update({
    "font.family"       : "monospace",
    "text.color"        : "#E0E0E0",
    "axes.labelcolor"   : "#E0E0E0",
    "xtick.color"       : "#AAAAAA",
    "ytick.color"       : "#AAAAAA",
    "axes.edgecolor"    : "#444444",
    "axes.facecolor"    : "#1A1A2E",
    "figure.facecolor"  : "#0D0D1A",
    "grid.color"        : "#2A2A3A",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
})

ACCENT   = "#00C8FF"
POSITIVE = "#00C896"
NEGATIVE = "#FF4F5A"
NEUTRAL  = "#6B8EAD"
GOLD     = "#FFD700"


# ──────────────────────────────────────────────
#  1. PRICE + VOLUME CHART
# ──────────────────────────────────────────────
def plot_price_chart(df: pd.DataFrame, symbol: str = "ASSET") -> Figure:
    """
    OHLCV price chart with volume bars.
    df must have: close, volume, open (optional)
    """
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.suptitle(f"{symbol}  —  Price & Volume", color=ACCENT, fontsize=13, fontweight="bold", y=0.98)

    # Price line
    ax_price.plot(df.index, df["close"], color=ACCENT, linewidth=1.5, label="Close")
    ax_price.fill_between(df.index, df["close"], df["close"].min() * 0.99, alpha=0.08, color=ACCENT)

    # SMA overlays
    for window, color in [(20, GOLD), (50, POSITIVE)]:
        if len(df) >= window:
            sma = df["close"].rolling(window).mean()
            ax_price.plot(df.index, sma, linewidth=0.8, color=color,
                          linestyle="--", alpha=0.7, label=f"SMA{window}")

    ax_price.set_ylabel("Price ($)", fontsize=10)
    ax_price.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax_price.grid(True)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.2f}"))

    # Volume bars
    colors = [POSITIVE if c >= o else NEGATIVE
              for c, o in zip(df["close"], df.get("open", df["close"]))]
    ax_vol.bar(df.index, df["volume"], color=colors, alpha=0.6, width=1)
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  2. MARTINGALE SIMULATION PATHS
# ──────────────────────────────────────────────
def plot_martingale_paths(sim_df: pd.DataFrame, real_prices: pd.Series = None,
                          symbol: str = "ASSET") -> Figure:
    """
    Plot simulated GBM martingale paths, optionally overlaying real prices.
    sim_df : DataFrame from martingale.simulate_martingale()
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Martingale Simulation — Zero-Drift GBM Paths", color=ACCENT,
                 fontsize=13, fontweight="bold")

    for col in sim_df.columns:
        ax.plot(sim_df.index, sim_df[col], linewidth=0.7, alpha=0.5, color=NEUTRAL)

    # Mean path
    mean_path = sim_df.mean(axis=1)
    ax.plot(sim_df.index, mean_path, color=GOLD, linewidth=2.0, label="Mean Path", zorder=5)

    # Real prices (scaled to match start)
    if real_prices is not None:
        scale = sim_df.iloc[0, 0] / real_prices.iloc[0]
        scaled = real_prices.values[:len(sim_df)] * scale
        ax.plot(range(len(scaled)), scaled, color=ACCENT, linewidth=1.5,
                linestyle="-", label=f"Actual: {symbol}", zorder=6)

    ax.set_xlabel("Trading Days", fontsize=10)
    ax.set_ylabel("Price", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_martingale_returns(log_returns: pd.Series, symbol: str = "ASSET") -> Figure:
    """
    Distribution of log-returns with normality overlay.
    """
    from scipy.stats import norm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Martingale Return Analysis — {symbol}", color=ACCENT,
                 fontsize=13, fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(log_returns, bins=50, color=ACCENT, alpha=0.7, density=True, edgecolor="#0D0D1A")
    x = np.linspace(log_returns.min(), log_returns.max(), 300)
    ax.plot(x, norm.pdf(x, log_returns.mean(), log_returns.std()),
            color=GOLD, linewidth=2, label="Normal fit")
    ax.axvline(0, color=NEGATIVE, linewidth=1.5, linestyle="--", label="Zero drift")
    ax.set_title("Return Distribution", color="#E0E0E0")
    ax.set_xlabel("Log Return")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True)

    # Return time series
    ax2 = axes[1]
    ax2.plot(log_returns.index, log_returns, color=ACCENT, linewidth=0.7, alpha=0.8)
    ax2.axhline(0, color=GOLD, linewidth=1, linestyle="--")
    ax2.fill_between(log_returns.index, log_returns, 0,
                     where=(log_returns >= 0), color=POSITIVE, alpha=0.3)
    ax2.fill_between(log_returns.index, log_returns, 0,
                     where=(log_returns < 0), color=NEGATIVE, alpha=0.3)
    ax2.set_title("Log Return Series", color="#E0E0E0")
    ax2.set_xlabel("Date")
    ax2.grid(True)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  3. BLACK-SCHOLES SURFACE
# ──────────────────────────────────────────────
def plot_bs_surface(surface_data: dict, option_type: str = "call") -> Figure:
    """
    3D option price surface: Strike × Expiry → Price
    surface_data : from black_scholes.generate_vol_surface()
    """
    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")

    K_grid = surface_data["K_grid"]
    T_grid = surface_data["T_grid"]
    prices = surface_data["prices"]

    surf = ax.plot_surface(
        K_grid, T_grid, prices,
        cmap="plasma", edgecolor="none", alpha=0.85
    )

    ax.set_xlabel("Strike (K)", labelpad=10, fontsize=9)
    ax.set_ylabel("Expiry (Years)", labelpad=10, fontsize=9)
    ax.set_zlabel(f"{option_type.title()} Price ($)", labelpad=10, fontsize=9)
    ax.set_title(
        f"Black-Scholes {option_type.title()} Price Surface",
        color=ACCENT, fontsize=13, fontweight="bold", pad=15
    )

    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, pad=0.1,
                 label=f"{option_type.title()} Price")
    ax.grid(True)
    fig.patch.set_facecolor("#0D0D1A")
    ax.set_facecolor("#0D0D1A")
    fig.tight_layout()
    return fig


def plot_greeks(greeks: dict, option_type: str = "call") -> Figure:
    """
    Bar chart of all five option Greeks.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Option Greeks — {option_type.title()}", color=ACCENT,
                 fontsize=13, fontweight="bold")

    names  = list(greeks.keys())
    values = list(greeks.values())
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in values]

    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="#0D0D1A", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#E0E0E0")

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  4. REGIME DETECTION CHART
# ──────────────────────────────────────────────
def plot_regime_chart(regime_result: dict, symbol: str = "ASSET") -> Figure:
    """
    Price chart with colored regime backgrounds.
    regime_result : from regime.fit_hmm()
    """
    from models.regime import get_regime_spans, REGIME_COLORS

    prices  = regime_result["prices_aligned"]
    labels  = regime_result["regime_labels"]
    returns = regime_result["log_returns"]
    spans   = get_regime_spans(labels)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.suptitle(f"Market Regime Detection — {symbol}", color=ACCENT,
                 fontsize=13, fontweight="bold")

    # Shade regime backgrounds
    for start, end, regime in spans:
        color = REGIME_COLORS.get(regime, "#888888")
        ax1.axvspan(start, end, alpha=0.15, color=color)
        ax2.axvspan(start, end, alpha=0.15, color=color)

    ax1.plot(prices.index, prices.values, color=ACCENT, linewidth=1.2, zorder=5)
    ax1.set_ylabel("Price ($)", fontsize=10)
    ax1.grid(True)

    # Returns colored by regime
    regime_color_series = labels.map(REGIME_COLORS)
    ax2.bar(returns.index, returns.values, color=regime_color_series.values, alpha=0.75, width=1)
    ax2.axhline(0, color="#888888", linewidth=0.8)
    ax2.set_ylabel("Log Return", fontsize=10)
    ax2.grid(True)

    # Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.7, label=r)
               for r in REGIME_COLORS if r in labels.values]
    ax1.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.3)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  5. SENTIMENT OVERLAY
# ──────────────────────────────────────────────
def plot_sentiment_overlay(
    price_df: pd.DataFrame,
    sentiment: pd.Series,
    symbol: str = "ASSET"
) -> Figure:
    """
    Price chart with sentiment score overlaid on secondary axis.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Price vs. Sentiment — {symbol}", color=ACCENT,
                 fontsize=13, fontweight="bold")

    # Price
    ax1.plot(price_df.index, price_df["close"], color=ACCENT, linewidth=1.5, label="Price")
    ax1.set_ylabel("Price ($)", fontsize=10, color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)

    # Sentiment on secondary axis
    ax2 = ax1.twinx()
    colors = [POSITIVE if s >= 0 else NEGATIVE for s in sentiment.values]
    ax2.bar(sentiment.index, sentiment.values, color=colors, alpha=0.45, width=1.5)
    ax2.plot(sentiment.index, sentiment.rolling(3, min_periods=1).mean(),
             color=GOLD, linewidth=1.5, linestyle="--", label="Sentiment (3d MA)")
    ax2.axhline(0, color="#666666", linewidth=0.8)
    ax2.set_ylabel("Sentiment Score", fontsize=10, color=GOLD)
    ax2.tick_params(axis="y", labelcolor=GOLD)
    ax2.set_ylim(-1.2, 1.2)

    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, framealpha=0.3)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from models import martingale, black_scholes, regime

    # Test martingale sim plot
    sim = martingale.simulate_martingale()
    fig = plot_martingale_paths(sim)
    fig.savefig("test_martingale.png", dpi=100)
    print("Saved test_martingale.png")

    # Test BS surface
    surf = black_scholes.generate_vol_surface()
    fig2 = plot_bs_surface(surf)
    fig2.savefig("test_bs_surface.png", dpi=100)
    print("Saved test_bs_surface.png")
