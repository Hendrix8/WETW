#!/usr/bin/env python3
"""Summarize fresh runs without substituting saved paper numbers."""

import argparse, csv, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True)
p.add_argument("--output", default="outputs/summary")
a = p.parse_args()
root = Path(a.runs)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(root.rglob("metrics.json")):
    d = json.loads(f.read_text())
    for point in d["search"]:
        row = {
            "run": str(f.parent.relative_to(root)),
            "dataset": d["dataset"],
            "smoke": d.get("smoke", False),
            "mean_abs_re": d["fidelity"]["mean_abs_re"],
            **point,
        }
        rows.append(row)
if not rows:
    raise SystemExit("No completed metrics.json files found")
keys = list(dict.fromkeys(k for r in rows for k in r))
with (out / "metrics.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
lines = [
    "# Fresh experiment summary",
    "",
    "Missing configurations are not filled from paper snapshots. No cross-dataset mean is reported unless the required dataset set is checked explicitly.",
    "",
    "|Run|Dataset|Shortlist|Recall@1|Seconds/query|Mean absolute RE|",
    "|---|---|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"|{r['run']}|{r['dataset']}|{r['rho']:.3g}|{r['recall@1']:.3f}|{r['seconds']:.4g}|{r['mean_abs_re']:.3f}|"
    )
(out / "README.md").write_text("\n".join(lines) + "\n")
print(out)
