"""Writes paper/table1.typ from results/drops.csv (notebook 03), so the paper's table is never typed by hand.

Test accuracy of every model after the 1-epoch fine-tune and right after pruning, before it. The gaps
after the fine-tune, with their intervals, are in Figure 2.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS = ["ResNet-18", "DeiT-Tiny"]
METHODS = ["magnitude", "random"]
WHEN = {"finetuned": "After the 1-epoch fine-tune", "pruned": "Right after pruning"}
acc = {(r["when"], r["model"], r["method"], float(r["sparsity"])): float(r["acc_all"])
       for r in csv.DictReader(open(ROOT / "results" / "drops.csv"))}

lines = ["#table(", "  columns: (auto,) + (1fr,) * 8,", "  align: (left,) + (right,) * 8,",
         "  column-gutter: (0pt,) * 4 + (8pt,) + (0pt,) * 3,",
         "  table.hline(),",
         "  table.header(",
         "    [], " + ", ".join(f"table.cell(colspan: 4, align: center)[{w}]" for w in WHEN.values()) + ",",
         "    table.hline(start: 1, end: 5, stroke: 0.4pt), table.hline(start: 5, end: 9, stroke: 0.4pt),",
         "    [], " + ", ".join(f"table.cell(colspan: 2, align: center)[{m}]" for _ in WHEN for m in MODELS) + ",",
         "    [sparsity], " + ", ".join("[mag.], [rand.]" for _ in WHEN for _ in MODELS) + ",",
         "  ),",
         "  table.hline(stroke: 0.5pt),"]
for s in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]:
    cells = [f"[{s:.0%}]"]
    for when in WHEN:
        for m in MODELS:
            if s == 0:  # the reference is shared by both methods
                cells.append(f"table.cell(colspan: 2, align: center)[{acc[(when, m, 'none', 0.0)]:.1f}]")
            else:
                cells += [f"[{acc[(when, m, meth, s)]:.1f}]" for meth in METHODS]
    lines.append("  " + ", ".join(cells) + ",")
lines += ["  table.hline(),", ")"]
(ROOT / "paper" / "table1.typ").write_text("\n".join(lines) + "\n")
print("wrote paper/table1.typ")
