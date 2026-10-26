"""CIFAR-10 loading and the fixed train/validation split (PLAN.md: Data, Training recipe)."""
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPLIT_FILE = ROOT / "splits" / "val_indices.npy"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentation happens at 32x32. The upscale to the model's input size happens later, on the GPU
# (train.upscale), which is much faster than resizing every image on the CPU.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def make_split(labels, per_class=500, seed=0):
    """Sorted validation indices: `per_class` random training images from each class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    val = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in range(10)]
    return np.sort(np.concatenate(val))


def load_split(labels):
    """The saved validation indices. The file is created the first time and reused after that."""
    if not SPLIT_FILE.exists():
        SPLIT_FILE.parent.mkdir(exist_ok=True)
        np.save(SPLIT_FILE, make_split(labels))
    return np.load(SPLIT_FILE)


def train_val_sets():
    """The 45,000 training images (augmented) and the 5,000 validation images (not augmented)."""
    train_full = datasets.CIFAR10(DATA_DIR, train=True, transform=TRAIN_TRANSFORM)
    val_full = datasets.CIFAR10(DATA_DIR, train=True, transform=EVAL_TRANSFORM)
    val_idx = load_split(train_full.targets)
    train_idx = np.setdiff1d(np.arange(len(train_full)), val_idx)
    return Subset(train_full, train_idx), Subset(val_full, val_idx)


def loader(dataset, batch_size=128, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Test set and human labels. Only notebook 03 opens these, for the final evaluation (Rule 7).
CIFAR10H_COUNTS = DATA_DIR / "cifar10h" / "cifar10h-counts.npy"
AMBIGUOUS_MAX_AGREEMENT = 0.868  # PLAN.md: Data. The bottom ~10% (1,011 images), ties included.


def test_set():
    """The 10,000 CIFAR-10 test images, not augmented."""
    return datasets.CIFAR10(DATA_DIR, train=False, transform=EVAL_TRANSFORM)


def human_counts():
    """CIFAR-10H: for each test image (in test-set order), how many annotators chose each class.

    The authors' cleaned file, which leaves out 16 inattentive annotators (about 50 labels per image).
    """
    return np.load(CIFAR10H_COUNTS)


def human_agreement(labels):
    """Share of CIFAR-10H annotators who chose each test image's CIFAR-10 label."""
    counts = human_counts()
    return counts[np.arange(len(labels)), labels] / counts.sum(1)
