"""Generate the README images from the saved checkpoints.

Run `train.py --model mlp` and `train.py --model cnn` first.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models import MODELS
from train import CLASSES, get_device, get_loaders

OUT = Path("images")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CONNECTOR = "#c3c2b7"
MLP_COLOR = "#2a78d6"
CNN_COLOR = "#eb6834"
CORRECT = "#006300"
WRONG = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "font.size": 11,
})


def to_image(tensor):
    """Undo the normalization in train.py so pixels are back in [0, 1]."""
    return (tensor.squeeze() * 0.3530 + 0.2860).clamp(0, 1).numpy()


@torch.no_grad()
def predict(name, test_set, device):
    model = MODELS[name]().to(device)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location=device, weights_only=True))
    model.eval()
    loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    return torch.cat([model(x.to(device)).argmax(1).cpu() for x, _ in loader])


def plot_samples(test_set, labels, per_class=4):
    fig, axes = plt.subplots(per_class, len(CLASSES), figsize=(13, 1.45 * per_class + 0.5))
    for c, name in enumerate(CLASSES):
        indices = (labels == c).nonzero().flatten()[:per_class]
        for row, i in enumerate(indices):
            ax = axes[row, c]
            ax.imshow(to_image(test_set[i][0]), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[0, c].set_title(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=120)
    plt.close(fig)


def plot_image_grid(test_set, indices, titles, colors, filename, cols=8):
    rows = -(-len(indices) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.65 * cols, 2.3 * rows))
    for ax, i, title, color in zip(axes.flat, indices, titles, colors):
        ax.imshow(to_image(test_set[i][0]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9.5, color=color)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=120)
    plt.close(fig)


def plot_predictions(test_set, labels, preds, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(labels), generator=g)[:n].tolist()
    titles, colors = [], []
    for i in indices:
        if preds[i] == labels[i]:
            titles.append(f"✓ {CLASSES[preds[i]]}")
            colors.append(CORRECT)
        else:
            titles.append(f"✗ {CLASSES[preds[i]]}\ntrue: {CLASSES[labels[i]]}")
            colors.append(WRONG)
    plot_image_grid(test_set, indices, titles, colors, "predictions.png")


def plot_mistakes(test_set, labels, preds, n=16, seed=0):
    wrong = (preds != labels).nonzero().flatten()
    g = torch.Generator().manual_seed(seed)
    indices = wrong[torch.randperm(len(wrong), generator=g)[:n]].tolist()
    titles = [f"true: {CLASSES[labels[i]]}\npred: {CLASSES[preds[i]]}" for i in indices]
    plot_image_grid(test_set, indices, titles, [SECONDARY] * len(indices), "mistakes.png")


def per_class_accuracy(labels, preds):
    return torch.tensor([(preds[labels == c] == c).float().mean().item() for c in range(len(CLASSES))])


def plot_per_class(labels, mlp_preds, cnn_preds):
    mlp_acc = per_class_accuracy(labels, mlp_preds) * 100
    cnn_acc = per_class_accuracy(labels, cnn_preds) * 100
    order = cnn_acc.argsort()  # lowest at the bottom, best at the top

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for y, c in enumerate(order.tolist()):
        ax.plot([mlp_acc[c], cnn_acc[c]], [y, y], color=CONNECTOR, linewidth=2, zorder=1)
    ys = range(len(CLASSES))
    ring = dict(s=90, edgecolors=SURFACE, linewidths=2, zorder=2)
    ax.scatter(mlp_acc[order], ys, color=MLP_COLOR, label=f"MLP  ({(mlp_preds == labels).float().mean():.1%} overall)", **ring)
    ax.scatter(cnn_acc[order], ys, color=CNN_COLOR, label=f"CNN  ({(cnn_preds == labels).float().mean():.1%} overall)", **ring)

    # Label only the two stories worth calling out: the biggest gain and the weakest class.
    gains = cnn_acc - mlp_acc
    for c in (gains.argmax().item(), cnn_acc.argmin().item()):
        y = order.tolist().index(c)
        ax.annotate(f"{gains[c]:+.1f} pts", (max(mlp_acc[c], cnn_acc[c]), y), xytext=(10, 0),
                    textcoords="offset points", va="center", color=SECONDARY, fontsize=10)

    ax.set_yticks(list(ys), [CLASSES[c] for c in order.tolist()])
    ax.set_xlim(70, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_yticklabels():
        label.set_color(SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Test accuracy by class", loc="left", fontsize=13, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_accuracy.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    device = get_device()
    test_set = get_loaders(batch_size=128)[2].dataset
    labels = test_set.targets
    mlp_preds = predict("mlp", test_set, device)
    cnn_preds = predict("cnn", test_set, device)

    plot_samples(test_set, labels)
    plot_predictions(test_set, labels, cnn_preds)
    plot_mistakes(test_set, labels, cnn_preds)
    plot_per_class(labels, mlp_preds, cnn_preds)
    print(f"Saved figures to {OUT}/")


if __name__ == "__main__":
    main()
