from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run(metrics, output):
    r = json.loads(Path(metrics).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for key, label in [("search", "WETW-FV"), ("euclidean_search", "Euclidean + DTW")]:
        if key not in r:
            continue
        rows = r[key]
        rk = next(k for k in rows[0] if k.startswith("recall@") and k != "recall@1")
        axes[0].plot([v["rho"] for v in rows], [v[rk] for v in rows], "-o", label=label)
        axes[1].plot(
            [v["seconds"] for v in rows], [v[rk] for v in rows], "-o", label=label
        )
    axes[0].set(xlabel="Shortlist fraction", ylabel=rk)
    axes[1].set(xlabel="Seconds per query", ylabel=rk)
    for a in axes:
        a.grid(alpha=0.2)
        a.legend(fontsize=8)
    fig.suptitle(r["dataset"] + (" (smoke test)" if r.get("smoke") else ""))
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
