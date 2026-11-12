#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
wetw-train study=smoke data.root=outputs/smoke_data
wetw paper
wetw benchmark --checkpoint outputs/smoke/model.pt --data outputs/smoke_data/whalesounds --output outputs/smoke/benchmark --methods wetw euclidean exact ds4 --k 3 --rhos 0.1 1
for mode in usp lbk lbw; do
  python scripts/benchmark_native.py --data outputs/smoke_data/whalesounds --output outputs/smoke/native --mode "$mode" --k 3 --check
done
