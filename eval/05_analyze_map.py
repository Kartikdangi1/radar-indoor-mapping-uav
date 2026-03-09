#!/usr/bin/env python3
"""
05_analyze_map.py
=================
Offline analysis of saved PGM occupancy grid maps.

Computes map quality metrics:
  - Occupied / free / unknown cell ratios
  - Map entropy (uncertainty measure)
  - Connected components (number of distinct wall segments)
  - Frontier length (boundary between free and unknown → exploration coverage)
  - Point density and spatial statistics
  - Ghost detection proxy: counts isolated single-cell occupied regions

Generates figures:
  eval/figures/map_analysis.pdf       — map with overlaid metrics
  eval/figures/map_histogram.pdf      — cell value distribution
  eval/figures/map_comparison.pdf     — before/after comparison (if two maps given)

Usage:
    # Single map:
    python3 eval/05_analyze_map.py --map eval/maps/radar_map_final.pgm

    # Comparison (e.g. with/without ghost filter):
    python3 eval/05_analyze_map.py \\
        --map eval/maps/with_ghost_filter.pgm \\
        --map2 eval/maps/without_ghost_filter.pgm \\
        --labels "With ghost filter" "Without ghost filter"

    # Ablation comparison (directory of maps):
    python3 eval/05_analyze_map.py --ablation-dir eval/maps/ablation/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage

THESIS_RC = {
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
}
plt.rcParams.update(THESIS_RC)


# ---------------------------------------------------------------------------
# PGM / OccupancyGrid loading
# ---------------------------------------------------------------------------

def load_pgm(path: str) -> np.ndarray:
    """Load a PGM file and return pixel values [0, 255]."""
    with open(path, "rb") as f:
        magic = f.readline().decode().strip()
        if magic not in ("P5", "P2"):
            raise ValueError(f"Not a PGM file: {magic}")
        # Skip comments
        while True:
            line = f.readline().decode().strip()
            if not line.startswith("#"):
                break
        dims = line.split()
        if len(dims) < 2:
            dims += f.readline().decode().strip().split()
        width, height = int(dims[0]), int(dims[1])
        maxval = int(f.readline().decode().strip())
        if magic == "P5":
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:  # P2 ASCII
            data = np.array([int(v) for v in f.read().split()], dtype=np.uint8)
    img = data.reshape((height, width))
    return img


def pgm_to_occupancy(img: np.ndarray) -> np.ndarray:
    """
    Convert PGM pixel values to ROS OccupancyGrid convention:
      255 (white) → free   → 0
      0   (black) → occ    → 100
      205 (grey)  → unkn   → -1

    Returns float array with values in {-1, 0..100}.
    """
    occ = np.full(img.shape, -1.0, dtype=np.float32)
    occ[img >= 240] = 0.0    # free (white)
    occ[img <= 10]  = 100.0  # occupied (black)
    # In-between: scale to 0-100
    mid = (img > 10) & (img < 240) & (img != 205)
    occ[mid] = 100.0 * (1.0 - img[mid].astype(float) / 255.0)
    return occ


def load_npz_map(path: str) -> np.ndarray:
    """Load occupancy grid saved as numpy (alternative to PGM)."""
    return np.load(path)["data"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_map_metrics(occ: np.ndarray, resolution: float = 0.07,
                        room_w_m: float = 0.0, room_h_m: float = 0.0) -> dict:
    """
    Args:
        occ:        occupancy grid (−1 unknown, 0 free, 1–100 occupied)
        resolution: metres per cell (default 7 cm)
        room_w_m:   actual room width  in metres  (0 = not provided)
        room_h_m:   actual room height in metres  (0 = not provided)
    Returns:
        dict of metrics, including room coverage % when room dims are provided
    """
    total = occ.size
    free_mask  = occ == 0
    occ_mask   = occ >= 65        # occupied threshold
    unkn_mask  = occ == -1
    free_frac  = float(free_mask.sum()) / total
    occ_frac   = float(occ_mask.sum())  / total
    unkn_frac  = float(unkn_mask.sum()) / total
    known_frac = free_frac + occ_frac

    # Map entropy (Shannon entropy over cell probabilities)
    p = occ[occ >= 0] / 100.0
    eps = 1e-9
    p_clipped = np.clip(p, eps, 1 - eps)
    entropy_bits = -p_clipped * np.log2(p_clipped) - (1 - p_clipped) * np.log2(1 - p_clipped)
    mean_entropy = float(np.nanmean(entropy_bits)) if len(entropy_bits) > 0 else 0.0

    # Connected components of occupied cells
    binary_occ = (occ_mask).astype(np.int8)
    labeled, n_components = ndimage.label(binary_occ)
    sizes = ndimage.sum(binary_occ, labeled, range(1, n_components + 1))
    n_wall_segments = int(np.sum(np.array(sizes) >= 3))
    n_isolated      = int(np.sum(np.array(sizes) < 3))

    # Frontier length: free-unknown boundary
    dilated_free = ndimage.binary_dilation(free_mask.astype(np.int8), iterations=1)
    frontier_cells = int((dilated_free & unkn_mask.astype(np.int8)).sum())
    frontier_m     = float(frontier_cells) * resolution

    # Explored area
    explored_m2 = float(free_mask.sum() + occ_mask.sum()) * resolution**2

    # Wall length estimate (occupied perimeter pixels)
    eroded = ndimage.binary_erosion(binary_occ, iterations=1)
    wall_boundary = binary_occ.astype(bool) & ~eroded
    wall_length_m = float(wall_boundary.sum()) * resolution

    metrics = {
        "total_cells":       total,
        "free_fraction":     round(free_frac,    4),
        "occupied_fraction": round(occ_frac,     4),
        "unknown_fraction":  round(unkn_frac,    4),
        "known_fraction":    round(known_frac,   4),
        "mean_entropy_bits": round(mean_entropy, 4),
        "n_wall_segments":   n_wall_segments,
        "n_isolated_points": n_isolated,
        "frontier_m":        round(frontier_m,   2),
        "explored_m2":       round(explored_m2,  2),
        "wall_length_m":     round(wall_length_m, 2),
        "resolution_m":      resolution,
    }

    # Room coverage (only when actual dimensions provided)
    if room_w_m > 0 and room_h_m > 0:
        room_area_m2   = room_w_m * room_h_m
        room_perim_m   = 2 * (room_w_m + room_h_m)
        coverage_pct   = 100.0 * explored_m2 / room_area_m2
        metrics.update({
            "room_w_m":          round(room_w_m,      2),
            "room_h_m":          round(room_h_m,      2),
            "room_area_m2":      round(room_area_m2,  2),
            "room_perimeter_m":  round(room_perim_m,  2),
            "room_coverage_pct": round(coverage_pct,  1),
        })

    return metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render_occ_map(occ: np.ndarray) -> np.ndarray:
    """Convert occupancy array to RGB for display."""
    h, w = occ.shape
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)  # grey = unknown
    rgb[occ == 0]    = [230, 230, 230]  # light grey = free
    rgb[occ >= 65]   = [30,  30,  30]   # dark = occupied
    rgb[(occ > 0) & (occ < 65)] = [180, 180, 240]  # uncertain = light blue
    return rgb


def plot_single_map(occ: np.ndarray, metrics: dict, out_path: str, title: str = "Occupancy Grid"):
    rgb = render_occ_map(occ)

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 1])

    # Main map
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.imshow(rgb, origin="upper")
    ax_map.set_title(title)
    ax_map.axis("off")

    # Metrics text box
    cov_line = ""
    if "room_coverage_pct" in metrics:
        cov_line = (
            f"Room: {metrics['room_w_m']:.2f} m × {metrics['room_h_m']:.2f} m  "
            f"({metrics['room_area_m2']:.1f} m²)\n"
            f"Coverage: {metrics['explored_m2']:.1f} / {metrics['room_area_m2']:.1f} m²  "
            f"= {metrics['room_coverage_pct']:.1f}%\n"
        )
    metric_str = (
        f"Resolution: {metrics['resolution_m']*100:.0f} cm/cell\n"
        + cov_line +
        f"Explored area: {metrics['explored_m2']:.1f} m²\n"
        f"Occupied: {metrics['occupied_fraction']*100:.1f}%  "
        f"Free: {metrics['free_fraction']*100:.1f}%  "
        f"Unknown: {metrics['unknown_fraction']*100:.1f}%\n"
        f"Mean entropy: {metrics['mean_entropy_bits']:.3f} bits\n"
        f"Wall segments: {metrics['n_wall_segments']}  "
        f"Isolated points: {metrics['n_isolated_points']}\n"
        f"Est. wall length: {metrics['wall_length_m']:.1f} m  "
        f"(actual perimeter: {metrics.get('room_perimeter_m', '?')} m)"
    )
    ax_map.text(0.01, 0.02, metric_str, transform=ax_map.transAxes,
                fontsize=8, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # Cell value histogram
    ax_hist = fig.add_subplot(gs[1, 0])
    known = occ[occ >= 0].ravel()
    ax_hist.hist(known, bins=50, color="#2196F3", alpha=0.7, edgecolor="none")
    ax_hist.set_xlabel("Occupancy probability (%)")
    ax_hist.set_ylabel("Cell count")
    ax_hist.set_title("Known-cell distribution")
    ax_hist.grid(True, linewidth=0.4, alpha=0.5)

    # Component size distribution
    ax_comp = fig.add_subplot(gs[1, 1])
    binary_occ = (occ >= 65).astype(np.int8)
    labeled, n = ndimage.label(binary_occ)
    if n > 0:
        sizes = ndimage.sum(binary_occ, labeled, range(1, n + 1))
        ax_comp.hist(sizes, bins=min(30, n), color="#FF9800", alpha=0.7, edgecolor="none")
        ax_comp.set_xlabel("Component size (cells)")
        ax_comp.set_ylabel("Count")
        ax_comp.set_title("Occupied component sizes")
        ax_comp.grid(True, linewidth=0.4, alpha=0.5)
    else:
        ax_comp.text(0.5, 0.5, "No occupied cells", ha="center", va="center",
                     transform=ax_comp.transAxes)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(occ1: np.ndarray, occ2: np.ndarray, labels: list, out_path: str):
    """Side-by-side map comparison (e.g. ghost filter on vs off)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, occ, label in zip(axes, [occ1, occ2], labels):
        ax.imshow(render_occ_map(occ), origin="upper")
        ax.set_title(label)
        ax.axis("off")
        m = compute_map_metrics(occ)
        info = (
            f"Occupied: {m['occupied_fraction']*100:.1f}%\n"
            f"Isolated pts: {m['n_isolated_points']}\n"
            f"Entropy: {m['mean_entropy_bits']:.3f}"
        )
        ax.text(0.01, 0.02, info, transform=ax.transAxes, fontsize=8,
                va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.suptitle("Map Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ablation_grid(maps_and_labels: list, out_path: str, ncols: int = 3):
    """Grid of ablation maps with metric annotations."""
    n = len(maps_and_labels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    axes_flat = np.array(axes).ravel()

    for i, (occ, label) in enumerate(maps_and_labels):
        ax = axes_flat[i]
        ax.imshow(render_occ_map(occ), origin="upper")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
        m = compute_map_metrics(occ)
        info = (
            f"Occ: {m['occupied_fraction']*100:.1f}%\n"
            f"Isolated: {m['n_isolated_points']}\n"
            f"H: {m['mean_entropy_bits']:.3f}"
        )
        ax.text(0.01, 0.02, info, transform=ax.transAxes, fontsize=7,
                va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Ablation Study — Occupancy Grid Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Ablation metric bar chart
# ---------------------------------------------------------------------------

def plot_ablation_metrics(maps_and_labels: list, out_path: str):
    names    = [label for _, label in maps_and_labels]
    metrics_ = [compute_map_metrics(occ) for occ, _ in maps_and_labels]

    keys_to_plot = [
        ("occupied_fraction",   "Occupied fraction",       True),
        ("n_isolated_points",   "Isolated points (noise)", True),
        ("mean_entropy_bits",   "Mean entropy (bits)",     False),
        ("wall_length_m",       "Est. wall length (m)",    False),
    ]

    fig, axes = plt.subplots(1, len(keys_to_plot), figsize=(14, 4))
    x = np.arange(len(names))

    for ax, (key, title, _) in zip(axes, keys_to_plot):
        vals = [m[key] for m in metrics_]
        colours = ["#2196F3" if i == 0 else "#FF9800" for i in range(len(vals))]
        colours[0] = "#4CAF50"  # baseline = green
        bars = ax.bar(x, vals, color=colours, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.set_ylim(bottom=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", fontsize=6.5)

    fig.suptitle("Ablation Study — Map Quality Metrics", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",    help="Path to primary map PGM")
    parser.add_argument("--map2",   help="Path to second map for comparison")
    parser.add_argument("--labels", nargs=2, default=["Map 1", "Map 2"],
                        help="Labels for two-map comparison")
    parser.add_argument("--resolution", type=float, default=0.07,
                        help="Map resolution in m/cell (default: 0.07)")
    parser.add_argument("--room-w", type=float, default=0.0,
                        help="Actual room width  in metres (for coverage %%)")
    parser.add_argument("--room-h", type=float, default=0.0,
                        help="Actual room height in metres (for coverage %%)")
    parser.add_argument("--ablation-dir", help="Directory with ablation PGM maps")
    parser.add_argument("--out", default="eval/figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Single map analysis
    if args.map:
        occ1 = pgm_to_occupancy(load_pgm(args.map))
        m1   = compute_map_metrics(occ1, args.resolution,
                                   room_w_m=args.room_w, room_h_m=args.room_h)

        name = Path(args.map).stem
        plot_single_map(occ1, m1,
                        os.path.join(args.out, f"map_analysis_{name}.pdf"),
                        title=name.replace("_", " "))

        stats_path = os.path.join(args.out, f"map_metrics_{name}.json")
        with open(stats_path, "w") as f:
            json.dump(m1, f, indent=2)
        print(f"\nMetrics: {stats_path}")
        print(json.dumps(m1, indent=2))

        # Two-map comparison
        if args.map2:
            occ2 = pgm_to_occupancy(load_pgm(args.map2))
            plot_comparison(occ1, occ2, args.labels,
                            os.path.join(args.out, "map_comparison.pdf"))
            m2 = compute_map_metrics(occ2, args.resolution)
            diff = {k: round(m1.get(k, 0) - m2.get(k, 0), 4)
                    for k in m1 if isinstance(m1[k], (int, float))}
            print("\nMetric differences (Map1 - Map2):")
            for k, v in diff.items():
                print(f"  {k:<30} {v:+.4f}")

    # Ablation directory
    elif args.ablation_dir:
        pgm_files = sorted(Path(args.ablation_dir).glob("*.pgm"))
        if not pgm_files:
            print(f"No PGM files found in {args.ablation_dir}")
            sys.exit(1)

        maps_and_labels = []
        all_metrics = {}
        for p in pgm_files:
            occ = pgm_to_occupancy(load_pgm(str(p)))
            label = p.stem.replace("_", " ")
            maps_and_labels.append((occ, label))
            all_metrics[label] = compute_map_metrics(occ, args.resolution)

        plot_ablation_grid(maps_and_labels,
                           os.path.join(args.out, "ablation_maps.pdf"))
        plot_ablation_metrics(maps_and_labels,
                              os.path.join(args.out, "ablation_metrics.pdf"))

        metrics_path = os.path.join(args.out, "ablation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAblation metrics: {metrics_path}")

    else:
        print("Usage: python3 05_analyze_map.py --map <path.pgm>  OR  --ablation-dir <dir>")
        parser.print_help()


if __name__ == "__main__":
    main()
