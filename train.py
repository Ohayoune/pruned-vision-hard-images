"""Training and evaluation, shared by the CNN and ViT notebooks (PLAN.md: Training recipe)."""
import itertools
import math
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def upscale(images, size):
    """Bilinear upscale of a batch of 32x32 images to size x size."""
    return F.interpolate(images, size=size, mode="bilinear", align_corners=False)


def sync(device):
    """Wait for the GPU to finish, so timings measure the work and not just the queueing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def make_optimizer(model, lr=1e-4, weight_decay=0.05):
    """AdamW. Biases and norm layers (1-D parameters) get no weight decay, as is standard."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim > 1], "weight_decay": weight_decay},
        {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def warmup_cosine(optimizer, total_steps, warmup_frac=0.05):
    """Linear warm-up over the first 5% of steps, then cosine decay to 0."""
    warmup = max(1, round(warmup_frac * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(model, train_loader, epochs, device, img_size, val_loader=None):
    """Fine-tune `model` with the shared recipe. Returns one dict per epoch."""
    optimizer = make_optimizer(model)
    scheduler = warmup_cosine(optimizer, epochs * len(train_loader))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss, n = torch.zeros((), device=device), 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            images, labels = upscale(images.to(device), img_size), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach() * len(labels)
            n += len(labels)
        row = {"epoch": epoch, "train_loss": total_loss.item() / n}
        if val_loader is not None:
            row["val_acc"] = accuracy(model, val_loader, device, img_size)
        row["minutes"] = (time.time() - start) / 60
        history.append(row)
    return history


@torch.no_grad()
def accuracy(model, loader, device, img_size):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        preds = model(upscale(images.to(device), img_size)).argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total


@torch.no_grad()
def predict(model, loader, device, img_size):
    """The predicted class of every image, in loader order (the loader must not shuffle)."""
    model.eval()
    preds = [model(upscale(images.to(device), img_size)).argmax(1).cpu() for images, _ in loader]
    return torch.cat(preds).numpy()


def seconds_per_step(model, train_loader, device, img_size, steps=20):
    """Average time of one training step. This trains `model`, so pass a throwaway copy."""
    optimizer = make_optimizer(model)
    model.train()
    batches = itertools.cycle(train_loader)  # small loaders (quick test) are reused
    for i in range(steps + 1):
        if i == 1:  # the first step includes one-off setup costs, so the clock starts after it
            sync(device)
            start = time.time()
        images, labels = next(batches)
        images, labels = upscale(images.to(device), img_size), labels.to(device)
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sync(device)
    return (time.time() - start) / steps
