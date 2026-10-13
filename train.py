import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import MODELS

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(batch_size, val_size=5000, seed=42):
    # Mean/std of the Fashion-MNIST training pixels.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST("data", train=False, download=True, transform=transform)

    # Hold out part of the training set for validation; the test set is only used once at the end.
    train, val = random_split(
        full_train,
        [len(full_train) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=1000),
        DataLoader(test, batch_size=1000),
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    class_correct = torch.zeros(len(CLASSES))
    class_total = torch.zeros(len(CLASSES))
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(1)

        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)
        for c in range(len(CLASSES)):
            mask = labels == c
            class_total[c] += mask.sum().item()
            class_correct[c] += (preds[mask] == c).sum().item()
    return total_loss / seen, correct / seen, class_correct / class_total


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on Fashion-MNIST.")
    parser.add_argument("--model", choices=MODELS.keys(), default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader = get_loaders(args.batch_size)

    model = MODELS[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()} ({n_params:,} parameters) on {device}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%}"
        )

    test_loss, test_acc, per_class = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest accuracy: {test_acc:.2%} (loss {test_loss:.4f})")
    for name, acc in zip(CLASSES, per_class):
        print(f"  {name:<12} {acc:.2%}")

    Path("checkpoints").mkdir(exist_ok=True)
    out = Path("checkpoints") / f"{args.model}.pt"
    torch.save(model.state_dict(), out)
    print(f"\nSaved weights to {out}")


if __name__ == "__main__":
    main()
