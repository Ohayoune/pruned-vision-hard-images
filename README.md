# Do pruned vision transformers also forget what humans find hard?

When pruned, do ViTs, like CNNs, lose accuracy disproportionately on the CIFAR-10 images that
humans find ambiguous? Ambiguous images are fixed in advance from CIFAR-10H human labels (the 1,011
test images with the lowest agreement). ResNet-18 and DeiT-Tiny are pruned (global magnitude and
random pruning, 30–95%), fine-tuned for one epoch, and compared at a matched overall accuracy drop.

**Result:** at a 12-point overall drop, ambiguous images lose 1.7× (ResNet-18) and 1.9×
(DeiT-Tiny) as many points as clear ones, with no detectable difference between the architectures.
Paper: [`paper/paper.pdf`](paper/paper.pdf). Plan, decisions and predictions: [`PLAN.md`](PLAN.md).

| File | What it does |
|---|---|
| `01_vit.ipynb` | DeiT-Tiny: base model, then the prune-and-fine-tune sweep (validation only) |
| `02_cnn.ipynb` | ResNet-18: the same steps with the same recipe |
| `03_test.ipynb` | The only notebook that opens the test set and CIFAR-10H: drops, figures, predictions checked |
| `data.py`, `models.py`, `train.py`, `pruning.py`, `sweep.py` | Code shared by the notebooks |
| `results/` | Per-image test predictions, `drops.csv`, `summary.json`, figures |
| `paper/` | Typst source, bibliography, and the table generator |

**Reproduce** (Apple-silicon Mac, about 2 hours of training): put CIFAR-10 in `data/` and
CIFAR-10H's `cifar10h-counts.npy` in `data/cifar10h/`, then run notebooks 01, 02 and 03 in order
(each skips work already saved in `checkpoints/`). Build the paper (needs `pip install typst`) with:

```bash
.venv/bin/python paper/make_table.py && .venv/bin/python -c "import typst; typst.compile('paper/paper.typ', output='paper/paper.pdf', root='.')"
```
