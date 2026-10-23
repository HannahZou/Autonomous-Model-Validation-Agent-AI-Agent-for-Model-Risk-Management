"""Static charts for the validation report (PNG, light print surface).

Palette: validated categorical slots blue / orange and the reserved status
red for VaR exceptions, which always appear with a legend label. One y-axis
per chart; recessive grid; text in ink colours, never series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"


def _figure(width: float = 6.5, height: float = 3.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title: str, path: Path) -> Path:
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def credit_calibration(deciles: list[dict], path: Path) -> Path:
    df = pd.DataFrame(deciles)
    fig, ax = _figure()
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, df["predicted_rate"], w, color=BLUE, label="Predicted PD")
    ax.bar(x + w / 2 + 0.01, df["observed_rate"], w, color=ORANGE, label="Observed default rate")
    ax.set_xticks(x, [str(d) for d in df["decile"]])
    ax.set_xlabel("Risk decile (1 = lowest predicted PD)", fontsize=8, color=INK_2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    last = df.iloc[-1]
    for xi, val, off in ((x[-1] - w / 2, last["predicted_rate"], -0.01), (x[-1] + w / 2, last["observed_rate"], 0.01)):
        ax.annotate(f"{val:.0%}", (xi + off, val), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color=INK_2)
    return _finish(fig, ax, "Calibration on the validation sample", path)


def credit_challengers(models: dict[str, dict], path: Path) -> Path:
    names = list(models)
    aucs = [models[n]["auc"] for n in names]
    fig, ax = _figure(height=0.55 * len(names) + 1.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    y = np.arange(len(names))[::-1]
    colors = [BLUE if i == 0 else ORANGE for i in range(len(names))]
    ax.hlines(y, 0.5, aucs, color=GRID, linewidth=1)
    ax.scatter(aucs, y, s=60, color=colors, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    for yi, a in zip(y, aucs):
        ax.annotate(f"{a:.3f}", (a, yi), textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    lo = max(0.5, min(aucs) - 0.05)
    ax.set_xlim(lo, min(1.0, max(aucs) + 0.04))
    ax.set_xlabel("Validation AUC (champion in blue, challengers in orange)", fontsize=8, color=INK_2)
    return _finish(fig, ax, "Champion vs challenger discrimination", path)


def var_backtest(series: pd.DataFrame, confidence: float, path: Path) -> Path:
    df = series.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = _figure(height=3.2)
    ax.bar(df["date"], df["return"], width=1.0, color=AXIS, label="Daily portfolio return")
    ax.plot(df["date"], -df["var"], color=BLUE, linewidth=1.5, label=f"−VaR ({confidence:.0%})")
    exc = df[df["exception"] == 1]
    ax.scatter(exc["date"], exc["return"], s=36, color=CRITICAL, zorder=3, edgecolors=SURFACE,
               linewidths=1.2, label=f"Exception ({len(exc)})")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    lo = max(df["return"].min(), -4 * df["var"].max())  # keep one extreme day from flattening the chart
    ax.set_ylim(lo * 1.1, df["return"].max() * 1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=3)
    return _finish(fig, ax, "VaR backtest: daily returns against the previous day's VaR", path)


def var_challengers(models: dict[str, dict], path: Path) -> Path:
    names = list(models)
    exc = [models[n]["exceptions"] for n in names]
    expected = models[names[0]]["expected"]
    fig, ax = _figure(height=2.8)
    x = np.arange(len(names))
    colors = [BLUE if i == 0 else ORANGE for i in range(len(names))]
    ax.bar(x, exc, 0.5, color=colors)
    ax.axhline(expected, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlim(-0.5, len(names) - 0.5 + 0.6)
    ax.annotate(f"expected\n{expected:g}", (len(names) - 0.5 + 0.05, expected), textcoords="offset points",
                xytext=(0, -2), ha="left", va="center", fontsize=8, color=INK_2)
    for xi, e, n in zip(x, exc, names):
        zone = models[n].get("traffic_light_last_250", "")
        ax.annotate(f"{e}  ({zone})", (xi, e), textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Exceptions", fontsize=8, color=INK_2)
    return _finish(fig, ax, "Exceptions on identical days (traffic-light zone, last 250 days)", path)
