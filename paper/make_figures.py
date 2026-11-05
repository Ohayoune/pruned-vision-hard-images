"""Writes the paper's two figures (paper/fig1_data.pdf, paper/fig2_gap.pdf) at the width of one column.

Same data and bootstrap as notebook 03 (same seed, same order), redrawn for print: notebook 03's
figures are 7 in wide and become unreadable when shrunk into a 3.6 in column.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

ARCHS = ["cnn", "vit"]
METHODS = ["magnitude", "random"]
SPARSITIES = [0.3, 0.5, 0.7, 0.9, 0.95]
DISPLAY = {"cnn": "ResNet-18", "vit": "DeiT-Tiny"}
N_BOOT = 2000
SEED = 0
WIDTH = 3.6  # inches: one column of the two-page paper

COLORS = {"cnn": "#2a78d6", "vit": "#eb6834"}  # categorical blue and orange, checked for colour-blind readers
INK, GRAY, LIGHT = "#3d3d3a", "#8a8a85", "#c3c2b7"
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.fontsize": 6.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

test = data.test_set()
labels = np.array(test.targets)
counts = data.human_counts()
agreement = data.human_agreement(labels)
ambiguous = agreement <= data.AMBIGUOUS_MAX_AGREEMENT
clear = ~ambiguous
n_amb, n_clear = int(ambiguous.sum()), int(clear.sum())
summary = json.loads((ROOT / "results" / "summary.json").read_text())

# Figure 1: the agreement distribution and examples
rng = np.random.default_rng(SEED)
examples = {"ambiguous": rng.choice(np.flatnonzero(ambiguous), 6, replace=False)[:3],
            "clear": rng.choice(np.flatnonzero(clear), 6, replace=False)[:3]}
share = counts / counts.sum(1, keepdims=True)

fig = plt.figure(figsize=(WIDTH, 1.75))
grid = fig.add_gridspec(2, 5, width_ratios=[2.2, 0.4, 1, 1, 1], wspace=0.24, hspace=0.45)
ax = fig.add_subplot(grid[:, 0])
ax.hist([agreement[ambiguous], agreement[clear]], bins=np.linspace(0, 1, 26), stacked=True,
        color=[INK, LIGHT], label=["ambiguous", "clear"])
ax.axvline(data.AMBIGUOUS_MAX_AGREEMENT, color=INK, lw=0.7, ls="--")
ax.set_yscale("log")
ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
ax.set_xlabel("human agreement")
ax.set_ylabel("test images")
ax.legend(frameon=False, loc="upper left", handlelength=1, borderaxespad=0.2)
for r, group in enumerate(examples):
    for col, i in enumerate(examples[group]):
        a = fig.add_subplot(grid[r, 2 + col])
        a.imshow(test.data[i])
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.spines[:].set_visible(False)
        title = f"{test.classes[labels[i]]} {share[i, labels[i]]:.0%}"
        if group == "ambiguous":
            other = max((c for c in range(10) if c != labels[i]), key=lambda c: counts[i, c])
            title += f"\n{test.classes[other]} {share[i, other]:.0%}"
        a.set_title(title, fontsize=5.5, pad=2, linespacing=1.1)
        if col == 0:
            a.set_ylabel(group, fontsize=6.5)
fig.savefig(ROOT / "paper" / "fig1_data.pdf")
plt.close(fig)

# Bootstrap of the test images, as in notebook 03 (same seed, models in the same order)
preds = np.load(ROOT / "results" / "test_predictions.npz")
correct = {k: preds[k] == labels for k in preds.files}
keys = sorted(correct)
row = {k: i for i, k in enumerate(keys)}
M = np.stack([correct[k] for k in keys]).astype(np.float32)
rng = np.random.default_rng(SEED)


def resampled_accuracy(idx):
    n = len(idx)
    draws = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float32)
    return np.concatenate([M[:, idx].mean(1, keepdims=True), M[:, idx] @ draws.T / n], axis=1)


acc_amb = resampled_accuracy(np.flatnonzero(ambiguous))
acc_clear = resampled_accuracy(np.flatnonzero(clear))
acc_all = (n_amb * acc_amb + n_clear * acc_clear) / (n_amb + n_clear)


def ci(x):
    return float(x[0]), float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))


def key(arch, method, amount):
    """The saved model after the 1-epoch fine-tune. ('none', 0.0) is the reference."""
    return f"{arch}.none_0_finetuned" if method == "none" else f"{arch}.{method}_{round(amount * 100)}_finetuned"


def drops(arch, method, amount):
    i, j = row[key(arch, method, amount)], row[key(arch, "none", 0.0)]
    return {g: 100 * (acc[j] - acc[i]) for g, acc in [("amb", acc_amb), ("clear", acc_clear), ("all", acc_all)]}


# Figure 2: the gap against the overall drop, after the fine-tune (main result)
fig, ax = plt.subplots(figsize=(WIDTH, 2.15))
floor_slope = {}
for arch in ARCHS:
    ref = row[key(arch, "none", 0.0)]
    a_amb, a_clear, a_all = (100 * acc[ref, 0] for acc in (acc_amb, acc_clear, acc_all))
    floor_x = np.array([0, a_all - 10])  # same share of above-chance accuracy lost by both groups
    floor_slope[arch] = (a_amb - a_clear) / (a_all - 10)
    ax.plot(floor_x, floor_x * floor_slope[arch], color=COLORS[arch], lw=0.7, alpha=0.5, zorder=1)
    for method in METHODS:
        pts = [drops(arch, "none", 0.0)] + [drops(arch, method, s) for s in SPARSITIES]
        x = [p["all"][0] for p in pts]
        gaps = np.array([ci(p["amb"] - p["clear"]) for p in pts])
        filled = method == "magnitude"
        ax.errorbar(x, gaps[:, 0], yerr=[gaps[:, 0] - gaps[:, 1], gaps[:, 2] - gaps[:, 0]],
                    color=COLORS[arch], ls="-" if filled else (0, (3, 1.5)), marker="o" if filled else "s",
                    mfc=COLORS[arch] if filled else "white", ms=3.2, mew=0.9, lw=1.1, elinewidth=0.6,
                    label=f"{DISPLAY[arch]}, {method}", zorder=3)
ax.axhline(0, color=INK, lw=0.7, zorder=2)
d_star = summary["matched"]["finetuned"]["d_star"]
ax.axvline(d_star, color=GRAY, lw=0.7, ls=":", zorder=1)
ax.set_xlim(-3, 90)
ax.set_ylim(-20, 16)
ax.annotate(f"D* = {d_star:.1f}", (d_star, 16), fontsize=6.5, color=GRAY, ha="left", va="top",
            xytext=(2, -1), textcoords="offset points")
fig.canvas.draw()  # the label follows the lower chance-floor line, so its angle needs the final layout
x0, x1 = 22, 34
y0, y1 = (x * min(floor_slope.values()) for x in (x0, x1))
(p0, p1) = ax.transData.transform([(x0, y0), (x1, y1)])
ax.text(x0, y0 - 1.6, "chance floor", fontsize=6, color=GRAY, ha="left", va="top",
        rotation=np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])), rotation_mode="anchor")
ax.set_xlabel("overall accuracy drop (points)")
ax.set_ylabel("Δ ambiguous − Δ clear (points)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=2.6,
          columnspacing=1.5, borderaxespad=0.2)
fig.savefig(ROOT / "paper" / "fig2_gap.pdf")
plt.close(fig)
print("wrote paper/fig1_data.pdf, paper/fig2_gap.pdf")
