import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    REGIME_LABELS, REGIME_COLORS, TICKERS, TICKER_NAMES,
    OUTPUT_DIR, SAVE_FIGURES,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def _save(fig: plt.Figure, name: str) -> None:
    if SAVE_FIGURES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
def plot_regime_timeline(
    regime_series: pd.Series,
    macro_raw: pd.DataFrame,
) -> plt.Figure:
    """
    Three-panel chart: yield curve + VIX + unemployment, coloured by regime.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle("Macroeconomic Regime Timeline  (2000 – present)", fontsize=14, fontweight="bold", y=1.01)

    panel_cfg = [
        ("yield_curve",  "Yield Curve 10Y–2Y (%)", "black"),
        ("vix",          "VIX",                    "#7b2d8b"),
        ("unemployment", "Unemployment Rate (%)",   "#e07b00"),
    ]

    for ax, (col, ylabel, color) in zip(axes, panel_cfg):
        _shade_regimes(ax, regime_series)
        series = macro_raw[col].reindex(macro_raw.index)
        ax.plot(series.index, series.values, color=color, linewidth=1.3, label=ylabel)
        if col == "yield_curve":
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Date")

    patches = [
        mpatches.Patch(facecolor=REGIME_COLORS[r], alpha=0.35, label=f"Regime {r}: {REGIME_LABELS[r]}")
        for r in range(3)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.03), frameon=False)

    plt.tight_layout()
    _save(fig, "regime_timeline")
    return fig


def _shade_regimes(ax: plt.Axes, regime_series: pd.Series) -> None:
    """Fill axis background with regime colours."""
    prev_r, prev_date = None, None
    for date, r in regime_series.items():
        if r != prev_r:
            if prev_r is not None:
                ax.axvspan(prev_date, date, alpha=0.20, color=REGIME_COLORS[prev_r], linewidth=0)
            prev_r, prev_date = r, date
    if prev_r is not None:
        ax.axvspan(prev_date, regime_series.index[-1], alpha=0.20, color=REGIME_COLORS[prev_r], linewidth=0)


# ──────────────────────────────────────────────────────────────────────────────
def plot_cumulative_returns_dual(results_hard: dict, results_prob: dict) -> plt.Figure:
    """
    Cumulative return chart showing both hard-label and probabilistic portfolios
    against SPY and equal-weight benchmarks.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    strategies = {
        "Probabilistic Portfolio": (results_prob["portfolio"], "#1f77b4",  2.2, "-"),
        "Hard-label Portfolio":    (results_hard["portfolio"], "#9467bd",  1.5, "--"),
        "S&P 500 (SPY)":           (results_prob["spy"],       "#ff7f0e",  1.6, "-"),
        "Equal Weight":            (results_prob["equal_weight"], "#2ca02c", 1.4, "-"),
    }

    for name, (ret, color, lw, ls) in strategies.items():
        cum = (1.0 + ret.dropna()).cumprod()
        ax.plot(cum.index, cum.values, label=name, color=color, linewidth=lw, linestyle=ls)

    ax.set_title("Cumulative Return  —  Probabilistic vs Hard-label vs Benchmarks",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.4)

    plt.tight_layout()
    _save(fig, "cumulative_returns")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_drawdowns_dual(results_hard: dict, results_prob: dict) -> plt.Figure:
    """Drawdown chart for all four strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    strategies = {
        "Probabilistic Portfolio": (results_prob["portfolio"], "#1f77b4",  1.8, "-"),
        "Hard-label Portfolio":    (results_hard["portfolio"], "#9467bd",  1.4, "--"),
        "S&P 500 (SPY)":           (results_prob["spy"],       "#ff7f0e",  1.5, "-"),
        "Equal Weight":            (results_prob["equal_weight"], "#2ca02c", 1.3, "-"),
    }

    for name, (ret, color, lw, ls) in strategies.items():
        cum = (1.0 + ret.dropna()).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1.0) * 100
        ax.plot(dd.index, dd.values, label=name, color=color, linewidth=lw, linestyle=ls)

    ax.set_title("Drawdown Comparison  —  Probabilistic vs Hard-label vs Benchmarks",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, "drawdown_comparison")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_returns_by_regime(
    equity_returns: pd.DataFrame,
    regime_series: pd.Series,
) -> plt.Figure:
    """Grouped bar chart: annualised return per sector per regime."""
    common = equity_returns.index.intersection(regime_series.index)
    rets = equity_returns.loc[common]
    regs = regime_series.loc[common]

    regime_data = {}
    for r in range(3):
        mask = regs == r
        annual = rets[mask].mean() * 12 * 100
        label = f"Regime {r}\n({REGIME_LABELS[r]})\nn={mask.sum()}"
        regime_data[label] = annual

    df_plot = pd.DataFrame(regime_data, index=[TICKER_NAMES.get(t, t) for t in TICKERS])

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df_plot))
    width = 0.25

    for i, (col, color) in enumerate(zip(df_plot.columns, REGIME_COLORS.values())):
        ax.bar(x + i * width, df_plot[col], width, label=col,
               color=color, alpha=0.82, edgecolor="white", linewidth=0.5)

    ax.set_title("Annualised Sector Returns by Regime", fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(df_plot.index, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    _save(fig, "returns_by_regime")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_weight_evolution(weights_df: pd.DataFrame, regimes: pd.Series) -> plt.Figure:
    """Stacked area chart of portfolio weights over time, with regime shading."""
    fig, ax = plt.subplots(figsize=(15, 6))

    _shade_regimes(ax, regimes)
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(TICKERS)))
    ax.stackplot(weights_df.index, weights_df.T.values, labels=TICKERS, colors=colors, alpha=0.85)

    ax.set_title("Portfolio Weight Evolution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    _save(fig, "weight_evolution")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_regime_heatmap(equity_returns: pd.DataFrame, regime_series: pd.Series) -> plt.Figure:
    """
    Heatmap: mean annualised return for each (sector, regime) cell.
    """
    common = equity_returns.index.intersection(regime_series.index)
    rets = equity_returns.loc[common]
    regs = regime_series.loc[common]

    table = pd.DataFrame(index=TICKERS, columns=[REGIME_LABELS[r] for r in range(3)], dtype=float)
    for r in range(3):
        mask = regs == r
        table[REGIME_LABELS[r]] = rets[mask].mean() * 12 * 100

    table.index = [TICKER_NAMES.get(t, t) for t in TICKERS]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        table.astype(float),
        annot=True, fmt=".1f", center=0,
        cmap="RdYlGn", linewidths=0.4,
        cbar_kws={"label": "Annualised Return (%)"},
        ax=ax,
    )
    ax.set_title("Regime–Sector Return Heatmap (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "regime_sector_heatmap")
    return fig
