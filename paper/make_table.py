"""Writes paper/table1.typ from results/drops.csv (notebook 03), so the paper's table is never typed by hand."""
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
rows = [r for r in csv.DictReader(open(ROOT / "results" / "drops.csv")) if r["when"] == "finetuned"]
curves = [("ResNet-18", "magnitude"), ("ResNet-18", "random"), ("DeiT-Tiny", "magnitude"), ("DeiT-Tiny", "random")]
by = {(r["model"], r["method"], float(r["sparsity"])): r for r in rows}


def cell(model, method, s):
    r = by[(model, "none" if s == 0 else method, s)]
    acc = f"{float(r['acc_all']):.1f}"
    if s == 0:
        return [acc, "—"]
    gap, lo, hi = (float(r[k]) for k in ("gap", "gap_lo", "gap_hi"))
    text = f"{gap:+.1f}".replace("-", "−")
    return [acc, f"*{text}*" if lo > 0 or hi < 0 else text]


lines = ["#table(", "  columns: (auto,) + (1fr,) * 8,", "  align: (left,) + (right,) * 8,",
         "  table.hline(),",
         "  table.header([], " + ", ".join(f"table.cell(colspan: 2, align: center)[{m}, {meth}]" for m, meth in curves) + ",",
         "    [sparsity], " + ", ".join("[acc.], [gap]" for _ in curves) + "),",
         "  table.hline(stroke: 0.5pt),"]
for s in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]:
    cells = [f"[{s:.0%}]"]
    for m, meth in curves:
        if s == 0:
            acc, amb_clear = cell(m, meth, s)
            cells += [f"[{acc}]", f"[{amb_clear}]"]
        else:
            cells += [f"[{c}]" for c in cell(m, meth, s)]
    lines.append("  " + ", ".join(cells) + ",")
lines += ["  table.hline(),", ")"]
(ROOT / "paper" / "table1.typ").write_text("\n".join(lines) + "\n")
print("wrote paper/table1.typ")
