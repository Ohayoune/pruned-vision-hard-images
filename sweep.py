"""The prune-then-fine-tune sweep, shared by the ViT and CNN notebooks (PLAN.md: Pruning,
Fine-tuning after pruning, Sparsity levels and runs)."""
import json
import time

import matplotlib.pyplot as plt
import torch

import models
import pruning
import train

COLORS = {"magnitude": "#2a78d6", "random": "#eb6834"}
MARKERS = {"magnitude": "o", "random": "s"}
NAMES = {"magnitude": "magnitude", "random": "random (control)"}


def condition_name(method, amount):
    return f"{method}_{round(amount * 100)}"


def run(arch, run_dir, train_loader, val_loader, device, img_size, methods, sparsities,
        finetune_epochs=1, seed=0):
    """Prune the saved base model at every (method, sparsity), fine-tune it, save both versions.

    The unpruned model ("none", 0.0) gets the same extra fine-tune, so the extra training isn't
    mistaken for an effect of pruning. Returns one dict of validation results per condition. The
    results are saved after every run and finished conditions are skipped, so an interrupted sweep
    picks up where it stopped.
    """
    skip = models.SKIP[arch]
    base_path = run_dir / "base.pt"
    results_path = run_dir / "val_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["method"], r["sparsity"]) for r in results}

    for method, amount in [("none", 0.0)] + [(m, s) for m in methods for s in sparsities]:
        if (method, amount) in done:
            continue
        start = time.time()
        name = condition_name(method, amount)
        torch.manual_seed(seed)  # the same random masks and data order, whatever order the runs happen in
        model = models.new_model(arch, pretrained=False, device=device)
        model.load_state_dict(torch.load(base_path, map_location=device))
        if method != "none":
            pruning.global_prune(model, skip, amount, method)
            torch.save(pruning.plain_state_dict(model), run_dir / f"{name}_pruned.pt")
        acc_pruned = train.accuracy(model, val_loader, device, img_size)

        train.train(model, train_loader, finetune_epochs, device, img_size)
        pruning.make_permanent(model)
        torch.save(model.state_dict(), run_dir / f"{name}_finetuned.pt")

        results.append({
            "method": method,
            "sparsity": amount,
            "sparsity_after_finetune": pruning.sparsity(model, skip),  # should still equal `sparsity`
            "val_acc_pruned": acc_pruned,
            "val_acc_finetuned": train.accuracy(model, val_loader, device, img_size),
            "minutes": (time.time() - start) / 60,
        })
        results_path.write_text(json.dumps(results, indent=1))  # a crash loses at most one run
        print(f"{name:>13}: {acc_pruned:.2%} after pruning -> {results[-1]['val_acc_finetuned']:.2%} "
              f"after fine-tuning ({results[-1]['minutes']:.1f} min)")
    return results


def print_table(results):
    print(f"{'method':>9} {'sparsity':>8} {'measured':>8} {'val acc pruned':>14} {'val acc fine-tuned':>18}")
    for r in sorted(results, key=lambda r: (r["method"], r["sparsity"])):
        print(f"{r['method']:>9} {r['sparsity']:>8.0%} {r['sparsity_after_finetune']:>8.4f} "
              f"{r['val_acc_pruned']:>14.2%} {r['val_acc_finetuned']:>18.2%}")


def plot_validation(results, methods, sparsities, title):
    """Dashed lines, open markers: right after pruning. Solid lines: after the 1-epoch fine-tune.

    The 0% point is the base model (dashed) and the base model plus the extra epoch (solid).
    """
    by = {(r["method"], r["sparsity"]): r for r in results}
    x = [0] + [s * 100 for s in sparsities]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method in methods:
        for key, style, when in [("val_acc_finetuned", "-", "after fine-tune"),
                                 ("val_acc_pruned", "--", "right after pruning")]:
            y = [by[("none", 0.0)][key] * 100] + [by[(method, s)][key] * 100 for s in sparsities]
            ax.plot(x, y, style, color=COLORS[method], marker=MARKERS[method], lw=2, ms=6,
                    mfc=COLORS[method] if style == "-" else "white", label=f"{NAMES[method]}, {when}")
    ax.set_xlabel("prunable weights set to zero (%)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title(title, loc="left")
    return fig
