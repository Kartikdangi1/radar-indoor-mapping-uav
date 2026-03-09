#!/usr/bin/env python3
"""
10_generate_experiment_figures.py
===================================
Generates clean thesis figures for the three evaluation experiments:

  Exp 1 — Room baseline occupancy map
  Exp 2 — Occlusion filtering ON vs OFF side-by-side
  Exp 3 — Corridor 2-lap trajectory, per-lap maps, resources, drift table

Does NOT call 05_analyze_map.py (which adds metric overlays we don't want).
Delegates performance + trajectory plots to 04_plot_performance.py and
02_plot_trajectory.py via subprocess.

Arguments:
  --maps-dir   eval/maps/experiments      directory with experiment PGMs
  --baseline   eval/maps/baseline.pgm     room baseline (Exp 1 + Exp 2 ON side)
  --data       eval/data/corridor         corridor odom.npz + performance_metrics.json
  --out        eval/figures/experiments   output directory
  --resolution 0.07                       m/cell
  --lap-split  1772903432.755             Unix timestamp splitting lap1/lap2

Usage:
    python3 eval/10_generate_experiment_figures.py \\
        --maps-dir eval/maps/experiments \\
        --baseline eval/maps/baseline.pgm \\
        --data     eval/data/corridor \\
        --out      eval/figures/experiments
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THESIS_RC = {
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
}
plt.rcParams.update(THESIS_RC)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable


# ---------------------------------------------------------------------------
# Map rendering helpers
# ---------------------------------------------------------------------------

def load_pgm(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load PGM: {path}")
    return img


def pgm_to_rgb(img: np.ndarray) -> np.ndarray:
    """ROS occupancy convention: 0=occupied(dark), 255=free(light), 205=unknown(gray)."""
    rgb = np.full((*img.shape, 3), 200, dtype=np.uint8)   # default: unknown mid-gray
    rgb[img > 210] = [240, 240, 240]                       # free → light gray
    rgb[img < 30]  = [30,  30,  30]                        # occupied → near-black
    return rgb


def add_scale_bar(ax, resolution_m: float, bar_len_m: float = 1.0):
    """Draw a scale bar in the bottom-right corner of ax."""
    bar_px = bar_len_m / resolution_m
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_right = xlim[1]
    y_bottom = ylim[1] if ylim[1] > ylim[0] else ylim[0]   # image y goes down
    margin_x = (xlim[1] - xlim[0]) * 0.04
    margin_y = (max(ylim) - min(ylim)) * 0.04
    x_end   = x_right - margin_x
    x_start = x_end - bar_px
    y_bar   = y_bottom - margin_y
    ax.plot([x_start, x_end], [y_bar, y_bar], color="black", linewidth=2, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2, y_bar - margin_y * 0.5,
            f"{bar_len_m:.0f} m", ha="center", va="top", fontsize=8, color="black")


def plot_clean_map(pgm_path: str, title: str, out_path: str, resolution: float = 0.07):
    """Single clean map with scale bar and title. No metric textboxes."""
    img = load_pgm(pgm_path)
    rgb = pgm_to_rgb(img)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="upper")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    add_scale_bar(ax, resolution)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_two_maps(pgm_a: str, pgm_b: str,
                  label_a: str, label_b: str,
                  out_path: str, resolution: float = 0.07):
    """Side-by-side 1×2 comparison. No metric overlays."""
    img_a = load_pgm(pgm_a)
    img_b = load_pgm(pgm_b)
    rgb_a = pgm_to_rgb(img_a)
    rgb_b = pgm_to_rgb(img_b)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, rgb, label in [(axes[0], rgb_a, label_a), (axes[1], rgb_b, label_b)]:
        ax.imshow(rgb, origin="upper")
        ax.set_title(label, fontsize=12)
        ax.axis("off")
        add_scale_bar(ax, resolution)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Drift stats LaTeX table
# ---------------------------------------------------------------------------

def write_drift_stats_tex(stats_path: str, out_path: str):
    """Read trajectory_stats.json and emit a drift summary LaTeX table."""
    if not os.path.exists(stats_path):
        print(f"  [SKIP] Drift stats: {stats_path} not found")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    dual = stats.get("dual_lap", {})
    if not dual:
        print(f"  [SKIP] Drift stats: no 'dual_lap' key in {stats_path}")
        return

    lap1_dist = dual.get("lap1_distance_m", 0.0)
    lap2_dist = dual.get("lap2_distance_m", 0.0)
    drift_m   = dual.get("start_to_end_drift_m", 0.0)
    drift_pct = dual.get("drift_percent", 0.0)

    # Estimate duration from split index if available (rough heuristic)
    # trajectory_stats only has split_idx, not per-lap duration — omit duration rows
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Lap 1 & Lap 2 \\",
        r"\midrule",
        f"Distance (m) & {lap1_dist:.1f} & {lap2_dist:.1f} \\\\",
        r"\midrule",
        f"Start-to-end drift & \\multicolumn{{2}}{{c}}{{{drift_m:.2f} m ({drift_pct:.1f}\\%)}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_script(script: str, extra_args: list = None):
    cmd = [PYTHON, os.path.join(EVAL_DIR, script)] + (extra_args or [])
    print(f"  >>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] {script} exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir",   default="eval/maps/experiments")
    parser.add_argument("--baseline",   default="eval/maps/baseline.pgm")
    parser.add_argument("--data",       default="eval/data/corridor")
    parser.add_argument("--out",        default="eval/figures/experiments")
    parser.add_argument("--resolution", type=float, default=0.07)
    parser.add_argument("--lap-split",  default="1772903432.755")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    maps  = args.maps_dir
    base  = args.baseline
    data  = args.data
    out   = args.out
    res   = args.resolution

    print("=" * 60)
    print("Experiment figure generation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Experiment 1 — Room baseline map
    # ------------------------------------------------------------------
    if os.path.exists(base):
        plot_clean_map(
            base,
            title="Experiment 1: Room Baseline Occupancy Map (All Features ON)",
            out_path=os.path.join(out, "fig_exp1_room_map.pdf"),
            resolution=res,
        )
    else:
        print(f"  [SKIP] Exp 1: baseline not found: {base}")

    # ------------------------------------------------------------------
    # Experiment 2 — Occlusion ON vs OFF
    # ------------------------------------------------------------------
    occ_off = os.path.join(maps, "occlusion_off.pgm")
    if os.path.exists(base) and os.path.exists(occ_off):
        plot_two_maps(
            base, occ_off,
            label_a="Occlusion ON (Baseline)",
            label_b="Occlusion OFF",
            out_path=os.path.join(out, "fig_exp2_occlusion.pdf"),
            resolution=res,
        )
    else:
        missing = [p for p in [base, occ_off] if not os.path.exists(p)]
        print(f"  [SKIP] Exp 2: missing files: {missing}")

    # ------------------------------------------------------------------
    # Experiment 3 — Corridor full map
    # ------------------------------------------------------------------
    corridor_full = os.path.join(maps, "corridor_full.pgm")
    if os.path.exists(corridor_full):
        plot_clean_map(
            corridor_full,
            title="Experiment 3: Corridor Full-Run Occupancy Map",
            out_path=os.path.join(out, "fig_exp3_corridor_map.pdf"),
            resolution=res,
        )
    else:
        print(f"  [SKIP] Exp 3 corridor map: {corridor_full} not found")

    # ------------------------------------------------------------------
    # Experiment 3 — Lap 1 vs Lap 2 comparison
    # ------------------------------------------------------------------
    lap1 = os.path.join(maps, "corridor_lap1.pgm")
    lap2 = os.path.join(maps, "corridor_lap2.pgm")
    if os.path.exists(lap1) and os.path.exists(lap2):
        plot_two_maps(
            lap1, lap2,
            label_a="Lap 1",
            label_b="Lap 2",
            out_path=os.path.join(out, "fig_exp3_lap_comparison.pdf"),
            resolution=res,
        )
    else:
        missing = [p for p in [lap1, lap2] if not os.path.exists(p)]
        print(f"  [SKIP] Exp 3 lap comparison: missing: {missing}")

    # ------------------------------------------------------------------
    # Experiment 3 — CPU/RAM (delegate to 04_plot_performance.py)
    # ------------------------------------------------------------------
    perf_json = os.path.join(data, "performance_metrics.json")
    if os.path.exists(perf_json):
        run_script("04_plot_performance.py", ["--data", data, "--out", out])
        # Rename output to experiment-specific name
        src = os.path.join(out, "performance_cpu_ram.pdf")
        dst = os.path.join(out, "fig_exp3_cpu_ram.pdf")
        if os.path.exists(src):
            os.replace(src, dst)
            print(f"  Renamed performance_cpu_ram.pdf → fig_exp3_cpu_ram.pdf")
    else:
        print(f"  [SKIP] Exp 3 CPU/RAM: {perf_json} not found")

    # ------------------------------------------------------------------
    # Experiment 3 — Trajectory + drift (delegate to 02_plot_trajectory.py)
    # ------------------------------------------------------------------
    odom_npz = os.path.join(data, "odom.npz")
    if os.path.exists(odom_npz):
        run_script("02_plot_trajectory.py", [
            "--data",      data,
            "--out",       out,
            "--lap-split", args.lap_split,
        ])
        # Rename to experiment-specific names
        renames = {
            "trajectory_2d.pdf":    "fig_exp3_trajectory.pdf",
        }
        for src_name, dst_name in renames.items():
            src = os.path.join(out, src_name)
            dst = os.path.join(out, dst_name)
            if os.path.exists(src):
                os.replace(src, dst)
                print(f"  Renamed {src_name} → {dst_name}")
        # Use dual-lap figure if it was generated
        dual_src = os.path.join(out, "trajectory_dual_lap.pdf")
        if os.path.exists(dual_src):
            os.replace(dual_src, os.path.join(out, "fig_exp3_trajectory.pdf"))
            print("  Renamed trajectory_dual_lap.pdf → fig_exp3_trajectory.pdf")
    else:
        print(f"  [SKIP] Exp 3 trajectory: {odom_npz} not found")

    # ------------------------------------------------------------------
    # Experiment 3 — Drift stats LaTeX table
    # ------------------------------------------------------------------
    stats_json = os.path.join(out, "trajectory_stats.json")
    write_drift_stats_tex(
        stats_path=stats_json,
        out_path=os.path.join(out, "fig_exp3_drift_stats.tex"),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Experiment figures written to: {out}")
    print("=" * 60)
    for f in sorted(Path(out).glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
