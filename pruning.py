"""Global unstructured pruning (PLAN.md: Pruning, Fine-tuning after pruning)."""
from torch import nn
from torch.nn.utils import prune

METHODS = {
    "magnitude": prune.L1Unstructured,  # zero the smallest |w|
    "random": prune.RandomUnstructured,  # control: zero a random set of the same size
}


def prunable(model, skip):
    """(module, "weight") for every conv and linear layer, except the layers named in `skip`."""
    return [
        (module, "weight")
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name not in skip
    ]


def global_prune(model, skip, amount, method):
    """Zero a fraction `amount` of the prunable weights, chosen across all layers at once.

    PyTorch keeps a 0/1 mask next to each pruned weight and multiplies it in on every forward pass,
    so the pruned weights stay at zero while the model is fine-tuned afterwards.
    """
    prune.global_unstructured(prunable(model, skip), pruning_method=METHODS[method], amount=amount)


def make_permanent(model):
    """Fold the masks into the weights, so the state dict has plain `weight` tensors again."""
    for module in model.modules():
        if hasattr(module, "weight_mask"):
            prune.remove(module, "weight")


def plain_state_dict(model):
    """The state dict with the masks folded in, leaving `model` itself untouched."""
    state = model.state_dict()
    for key in [k for k in state if k.endswith(".weight_orig")]:
        name = key.removesuffix("_orig")
        state[name] = state.pop(key) * state.pop(name + "_mask")
    return state


def sparsity(model, skip):
    """Fraction of the prunable weights that are exactly zero."""
    weights = [getattr(module, name) for module, name in prunable(model, skip)]
    zeros = sum((w == 0).sum().item() for w in weights)
    return zeros / sum(w.numel() for w in weights)
