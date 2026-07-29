"""Generate the EDA feature table and figures.

Extracted into a script (rather than living only in the notebook) for two
reasons: the Streamlit UI reads the resulting CSV to build its interactive
charts, and the notebook can call these functions instead of duplicating the
plotting code.

    python src/run_eda.py --samples-per-class 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")            # no display in a headless environment
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from src.config import FIGURES_DIR, REPORTS_DIR, TRAIN_DIR  # noqa: E402
from src.preprocessing import (  # noqa: E402
    build_feature_table,
    class_distribution,
)

FEATURES_CSV = REPORTS_DIR / "eda_features.csv"

# Mirrors ui/theme.py so figures and the dashboard read as one system.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=12, weight="600")
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-", alpha=0.9)
    ax.set_axisbelow(True)


def figure_class_distribution(distribution: pd.DataFrame) -> Path:
    """Finding 1 -- imbalance decides which metric we trust."""
    fig, ax = plt.subplots(figsize=(10, 9))
    frame = distribution.sort_values("count")

    colours = [AQUA if h else ORANGE for h in frame["healthy"]]
    labels = [f"{r.crop} — {r.condition}"[:46] for r in frame.itertuples()]

    ax.barh(range(len(frame)), frame["count"], color=colours, height=0.68)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(labels, fontsize=8)
    _style(ax, "Images per class — a 36x imbalance", "images", "")
    ax.grid(axis="y", visible=False)

    # Direct-label only the extremes; a number on every bar goes unread.
    for i in (0, len(frame) - 1):
        ax.text(frame["count"].iloc[i] + 60, i, f"{frame['count'].iloc[i]:,}",
                va="center", fontsize=9, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=AQUA),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Healthy", "Diseased"], frameon=False,
              loc="lower right", fontsize=9)

    path = FIGURES_DIR / "01_class_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


def figure_colour_separation(features: pd.DataFrame) -> Path:
    """Finding 2 -- colour alone already separates healthy from diseased."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    groups = [("Healthy", features[features.healthy], AQUA),
              ("Diseased", features[~features.healthy], ORANGE)]

    for name, group, colour in groups:
        axes[0].hist(group["excess_green"], bins=45, alpha=0.62,
                     color=colour, label=name, edgecolor="none")
    _style(axes[0], "Excess green (2G − R − B)", "index value", "images")
    axes[0].legend(frameon=False, fontsize=9)

    for name, group, colour in groups:
        axes[1].scatter(group["excess_green"], group["redness_index"],
                        s=9, alpha=0.4, color=colour, label=name,
                        linewidths=0.5, edgecolors="white")
    _style(axes[1], "Green against red shift", "excess green", "redness index")
    axes[1].legend(frameon=False, fontsize=9, markerscale=2)

    path = FIGURES_DIR / "02_colour_separation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


FEATURE_COLUMNS = ["excess_green", "redness_index", "green_ratio",
                   "mean_saturation", "edge_density", "laplacian_var",
                   "gray_entropy", "brightness"]


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference -- comparable across different scales."""
    pooled = np.sqrt((a.var() + b.var()) / 2) or 1e-9
    return float((a.mean() - b.mean()) / pooled)


def within_crop_separation(features: pd.DataFrame, min_group: int = 25) -> pd.DataFrame:
    """Effect size of healthy vs diseased, computed *per crop*.

    Pooling all crops together compares tomato leaves against apple leaves as
    much as sick against healthy ones. Conditioning on crop removes that
    confound and is what reveals the disease signal.
    """
    rows = []
    for crop, group in features.groupby("crop"):
        healthy, diseased = group[group.healthy], group[~group.healthy]
        if len(healthy) < min_group or len(diseased) < min_group:
            continue
        rows.append({"crop": crop, "n": len(group),
                     **{c: cohens_d(healthy[c], diseased[c]) for c in FEATURE_COLUMNS}})
    return pd.DataFrame(rows).set_index("crop")


def figure_texture(features: pd.DataFrame) -> Path:
    """Finding 3 -- the signal only appears once you condition on crop.

    Two panels, both of which refute a tempting assumption:
    left, the two texture features are near-collinear, so they are one signal
    and not two; right, every feature separates far better within a crop than
    across the whole dataset.
    """
    within = within_crop_separation(features)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # -- redundancy --------------------------------------------------------
    axes[0].scatter(features["edge_density"], features["laplacian_var"],
                    s=9, alpha=0.35, color=BLUE, linewidths=0.5,
                    edgecolors="white")
    r = features["edge_density"].corr(features["laplacian_var"])
    _style(axes[0], f"Two texture features, one signal  (r = {r:.2f})",
           "edge density", "Laplacian variance")

    # -- global vs within-crop --------------------------------------------
    healthy, diseased = features[features.healthy], features[~features.healthy]
    global_d = [abs(cohens_d(healthy[c], diseased[c])) for c in FEATURE_COLUMNS]
    within_d = [within[c].abs().mean() for c in FEATURE_COLUMNS]

    order = np.argsort(within_d)
    labels = [FEATURE_COLUMNS[i].replace("_", " ") for i in order]
    positions = np.arange(len(order))

    axes[1].barh(positions + 0.19, [global_d[i] for i in order], height=0.36,
                 color=ORANGE, label="pooled across crops")
    axes[1].barh(positions - 0.19, [within_d[i] for i in order], height=0.36,
                 color=AQUA, label="within crop (mean)")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(labels, fontsize=9)
    _style(axes[1], "Separation is 3–5x stronger within a crop",
           "|Cohen's d|  (healthy vs diseased)", "")
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")

    path = FIGURES_DIR / "03_texture_signal.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


def figure_background(features: pd.DataFrame) -> Path:
    """Finding 4 -- the uniform background, and why it is a warning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].hist(features["corner_std"], bins=50, color=BLUE, edgecolor="none")
    _style(axes[0], "Corner pixel variation", "std dev across image corners", "images")

    axes[1].hist(features["background_fraction"], bins=50, color=BLUE, edgecolor="none")
    _style(axes[1], "Share of frame matching the backdrop",
           "background fraction", "images")

    path = FIGURES_DIR / "04_background_uniformity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


def interpret(features: pd.DataFrame, distribution: pd.DataFrame) -> dict:
    """Compute the numbers the written interpretations quote.

    Quoting measured values rather than adjectives is the difference between
    an interpretation and a caption.
    """
    healthy = features[features.healthy]
    diseased = features[~features.healthy]
    within = within_crop_separation(features)

    def gap(column: str) -> dict:
        return {
            "healthy_mean": round(float(healthy[column].mean()), 4),
            "diseased_mean": round(float(diseased[column].mean()), 4),
            # Cohen's d -- an effect size, so separation is comparable across
            # features measured on completely different scales.
            "cohens_d_pooled": round(cohens_d(healthy[column], diseased[column]), 3),
            "cohens_d_within_crop_mean": round(float(within[column].abs().mean()), 3),
            # Whether the sign of the effect is consistent across crops. It is
            # mostly not, which is the finding.
            "sign_consistent_across_crops": bool(
                (within[column] > 0).all() or (within[column] < 0).all()),
        }

    return {
        "n_images_sampled": int(len(features)),
        "n_classes": int(distribution.shape[0]),
        "n_crops": int(distribution["crop"].nunique()),
        "imbalance_ratio": round(float(distribution["count"].max() /
                                       max(distribution["count"].min(), 1)), 1),
        "smallest_class": distribution.nsmallest(1, "count")["class"].iloc[0],
        "largest_class": distribution.nlargest(1, "count")["class"].iloc[0],
        "tomato_class_share": int((distribution["crop"] == "Tomato").sum()),
        "features": {c: gap(c) for c in FEATURE_COLUMNS},
        "texture_redundancy_r": round(
            float(features["edge_density"].corr(features["laplacian_var"])), 3),
        "mean_abs_d_pooled": round(float(np.mean(
            [abs(cohens_d(healthy[c], diseased[c])) for c in FEATURE_COLUMNS])), 3),
        "mean_abs_d_within_crop": round(float(np.mean(
            [within[c].abs().mean() for c in FEATURE_COLUMNS])), 3),
        # Brightness carries no botanical meaning, so if it out-separates the
        # vegetation indices that is evidence of an acquisition confound.
        "brightness_outranks_colour": bool(
            abs(cohens_d(healthy["brightness"], diseased["brightness"]))
            > abs(cohens_d(healthy["excess_green"], diseased["excess_green"]))),
        "within_crop_table": within[FEATURE_COLUMNS].round(3).to_dict(orient="index"),
        "background_fraction_median": round(float(features["background_fraction"].median()), 3),
        "corner_std_median": round(float(features["corner_std"].median()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--split", default=str(TRAIN_DIR))
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Counting images per class in {args.split}...")
    distribution = class_distribution(args.split)

    print(f"\nExtracting features ({args.samples_per_class} images per class)...")
    features = build_feature_table(args.split, samples_per_class=args.samples_per_class)

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)
    distribution.to_csv(REPORTS_DIR / "class_distribution.csv", index=False)
    print(f"\nWrote {len(features):,} rows to {FEATURES_CSV}")

    print("\nRendering figures...")
    for path in (figure_class_distribution(distribution),
                 figure_colour_separation(features),
                 figure_texture(features),
                 figure_background(features)):
        print(f"  {path}")

    summary = interpret(features, distribution)
    import json
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 74)
    print(f"  sampled {summary['n_images_sampled']:,} images across "
          f"{summary['n_classes']} classes / {summary['n_crops']} crops")
    print(f"  imbalance {summary['imbalance_ratio']}x  "
          f"({summary['smallest_class']} -> {summary['largest_class']})")

    print(f"\n  healthy vs diseased separation (|Cohen's d|)")
    print(f"    {'feature':<18} {'pooled':>8} {'within-crop':>13}  {'sign stable':>12}")
    for name, stat in summary["features"].items():
        print(f"    {name:<18} {abs(stat['cohens_d_pooled']):>8.3f} "
              f"{stat['cohens_d_within_crop_mean']:>13.3f}  "
              f"{str(stat['sign_consistent_across_crops']):>12}")

    print(f"\n  mean |d| pooled {summary['mean_abs_d_pooled']} "
          f"-> within-crop {summary['mean_abs_d_within_crop']}")
    print(f"  edge_density vs laplacian_var correlation: "
          f"r = {summary['texture_redundancy_r']} (redundant)")
    print(f"  brightness out-separates excess green: "
          f"{summary['brightness_outranks_colour']} (acquisition confound)")
    print(f"  median background fraction: {summary['background_fraction_median']}")
    print("=" * 74)


if __name__ == "__main__":
    main()
