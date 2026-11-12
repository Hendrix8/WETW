"""Replot the paper's exported numeric evidence; never run training implicitly."""

from pathlib import Path
import csv, json, math, shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _marker(m):
    """JSON marker -> matplotlib: names stay strings, [n, style, angle] becomes a
    regular-polygon tuple and a list of [x, y] pairs a vertex path."""
    if m in (None, "None", "none", ""):
        return None
    if isinstance(m, list) and m and all(not isinstance(v, list) for v in m):
        return (int(m[0]), int(m[1]), float(m[2]))
    if isinstance(m, list):
        return [tuple(v) for v in m]
    return m


def reproduce(results, output):
    data = json.loads((results / "figures.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, figure in data.items():
        count = len(figure["axes"])
        cols = 3 if count in (3, 6) else 4 if count == 8 else 2 if count == 2 else 1
        rows = math.ceil(count / cols)
        fig = plt.figure(figsize=figure["size_inches"])
        for ai, spec in enumerate(figure["axes"]):
            ax = fig.add_axes(spec.get("position", [0.12, 0.15, 0.8, 0.75]))
            for li, line in enumerate(spec["lines"]):
                marker = _marker(line["marker"])
                ax.plot(
                    line["x"],
                    line["y"],
                    label=line["label"],
                    color=line["color"],
                    alpha=line.get("alpha"),
                    linestyle=line["linestyle"],
                    marker=marker,
                    markersize=3,
                    linewidth=1,
                )
                for x, y in zip(line["x"], line["y"]):
                    records.append(
                        {
                            "figure": name,
                            "panel": ai,
                            "series": li,
                            "label": line["label"],
                            "x": x,
                            "y": y,
                        }
                    )
            ax.set(
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                title=spec["title"],
                xscale=spec["xscale"] if spec["xscale"] != "function" else "linear",
                yscale=spec["yscale"] if spec["yscale"] != "function" else "linear",
                xlim=spec["xlim"],
                ylim=spec["ylim"],
            )
            if spec.get("xticks") and len(spec["xticks"]) == len(
                spec.get("xticklabels", [])
            ):
                lo, hi = sorted(spec["xlim"])
                ticks = [
                    (x, label)
                    for x, label in zip(spec["xticks"], spec["xticklabels"])
                    if lo <= x <= hi
                ]
                if ticks:
                    ax.set_xticks(
                        [v[0] for v in ticks], [v[1] for v in ticks], fontsize=7
                    )
            from matplotlib.ticker import NullLocator

            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.get_offset_text().set_visible(False)
            ax.set_xlabel(spec["xlabel"], fontsize=8)
            ax.set_ylabel(spec["ylabel"], fontsize=8)
            for annotation in spec.get("annotations", []):
                transform = {
                    "axes": ax.transAxes,
                    "xaxis": ax.get_xaxis_transform(),
                    "yaxis": ax.get_yaxis_transform(),
                    "data": ax.transData,
                }[annotation["transform"]]
                ax.text(
                    *annotation["position"],
                    annotation["text"],
                    transform=transform,
                    ha=annotation["ha"],
                    va=annotation["va"],
                    rotation=annotation["rotation"],
                    fontsize=min(8, annotation["fontsize"]),
                )
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)
        if figure.get("legend"):
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [],
                    [],
                    color=v["color"],
                    linestyle=v["linestyle"],
                    marker=_marker(v["marker"]),
                    label=v["label"],
                    markersize=3,
                    linewidth=1,
                )
                for v in figure["legend"]
            ]
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=min(6, len(handles)),
                fontsize=7,
                frameon=False,
            )
        fig.savefig(output / f"{name}.pdf")
        fig.savefig(output / f"{name}.png", dpi=180)
        plt.close(fig)
    with (output / "figure_points.csv").open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["figure", "panel", "series", "label", "x", "y"]
        )
        w.writeheader()
        w.writerows(records)
    shutil.copytree(results / "tables", output / "tables", dirs_exist_ok=True)
    (output / "README.md").write_text(
        "These figures redraw the saved paper evidence, without training or new timing measurements.\n\nSee results/paper/manifest.json for timing provenance and scaling caveats.\n"
    )
    print(
        f"Regenerated {len(data)} figures, {len(records)} numeric points, and paper tables in {output}"
    )
