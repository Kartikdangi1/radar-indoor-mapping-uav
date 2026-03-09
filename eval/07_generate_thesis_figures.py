#!/usr/bin/env python3
"""
07_generate_thesis_figures.py
==============================
Orchestrates all thesis figure generation by calling scripts 02, 04, and 05.
Also produces a combined dataset-rates bar chart (Fig 5.1) and the ablation
LaTeX table directly.

Prerequisites (run in order):
  1. python3 eval/01_extract_bag_data.py --bag <bag>   → eval/data/
  2. ./eval/06_run_ablation.sh --bag <bag>             → eval/maps/ablation/
  3. python3 eval/03_collect_live_performance.py       → eval/data/performance_metrics.json

Usage:
    python3 eval/07_generate_thesis_figures.py \\
        [--data eval/data] \\
        [--maps eval/maps] \\
        [--perf eval/data/performance_metrics.json] \\
        [--out  eval/figures/thesis]

All PDFs and .tex snippets land in eval/figures/thesis/.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THESIS_RC = {
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
}
plt.rcParams.update(THESIS_RC)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable


# ---------------------------------------------------------------------------
# Helper: run a child script
# ---------------------------------------------------------------------------

def run_script(script: str, extra_args: list = None):
    cmd = [PYTHON, os.path.join(EVAL_DIR, script)] + (extra_args or [])
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] {script} exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Fig 5.1 — Dataset sensor message-count bar chart
# ---------------------------------------------------------------------------

def fig_dataset_rates(out_path: str):
    sensors = [
        ("IMU\n(≈500 Hz)",          57_785),
        ("Downward\nLiDAR (≈50 Hz)", 5_779),
        ("Camera\n(≈30 Hz)",         3_459),
        ("Radar\n(≈20 Hz)",          2_311),
    ]
    names, counts = zip(*sensors)
    colours = ["#2196F3", "#4CAF50", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, counts, color=colours, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{c:,}", ha="center", fontsize=8)
    ax.set_ylabel("Message count (115.6 s dataset)")
    ax.set_title("Sensor Message Counts in Evaluation Dataset")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Fig 5.1 → {out_path}")


# ---------------------------------------------------------------------------
# Ablation LaTeX table (from eval/figures/ablation_metrics.json)
# ---------------------------------------------------------------------------

ABLATION_DESCRIPTIONS = {
    "baseline": "Baseline (all features enabled)",
    "A1": "A1: Temporal decay disabled",
    "A2": "A2: Multipath rejection off",
    "A3": "A3: Per-ray occlusion off",
    "A4": "A4: Map-based occlusion off",
    "A5": "A5: Dynamic object rejection off",
    "A6": "A6: Ego-motion compensation off",
    "A7": "A7: GICP drift correction off",
}


def write_ablation_latex_table(metrics_path: str, out_path: str):
    if not os.path.exists(metrics_path):
        print(f"  [SKIP] Ablation metrics not found: {metrics_path}")
        return
    with open(metrics_path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Ablation study --- map quality metrics per configuration.}",
        r"    \label{tab:ablation_metrics}",
        r"    \small",
        r"    \begin{tabular}{lrrrr}",
        r"        \toprule",
        r"        Configuration & Occ.\ (\%) & Isolated pts & Entropy (bits) & Wall (m) \\",
        r"        \midrule",
    ]
    for label, m in data.items():
        desc = ABLATION_DESCRIPTIONS.get(label, label)
        bold_open  = r"\textbf{" if label == "baseline" else ""
        bold_close = r"}" if label == "baseline" else ""
        lines.append(
            f"        {bold_open}{desc}{bold_close} & "
            f"{bold_open}{m.get('occupied_fraction', 0)*100:.1f}{bold_close} & "
            f"{bold_open}{m.get('n_isolated_points', 0)}{bold_close} & "
            f"{bold_open}{m.get('mean_entropy_bits', 0):.3f}{bold_close} & "
            f"{bold_open}{m.get('wall_length_m', 0):.1f}{bold_close} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Ablation LaTeX table → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="eval/data")
    parser.add_argument("--maps", default="eval/maps")
    parser.add_argument("--perf", default="eval/data/performance_metrics.json")
    parser.add_argument("--out",  default="eval/figures/thesis")
    parser.add_argument("--figs", default="eval/figures",
                        help="Intermediate figures directory used by child scripts")
    parser.add_argument("--lap-split", default="",
                        help="Unix timestamp for lap split in corridor bag (enables dual-lap figure)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    print("=" * 60)
    print("Thesis figure generation pipeline")
    print("=" * 60)

    # ── Fig 5.1 (standalone) ─────────────────────────────────────────────
    fig_dataset_rates(os.path.join(args.out, "fig5_1_dataset_rates.pdf"))

    # ── Figs 5.2–5.4, height, IMU  (script 02) ───────────────────────────
    odom_npz = os.path.join(args.data, "odom.npz")
    if os.path.exists(odom_npz):
        traj_args = ["--data", args.data, "--out", args.figs]
        if args.lap_split:
            traj_args.extend(["--lap-split", args.lap_split])
        run_script("02_plot_trajectory.py", traj_args)
        # Copy to thesis folder with standard names
        copies = {
            "trajectory_2d.pdf":       "fig5_2_trajectory_2d.pdf",
            "velocity_profile.pdf":    "fig5_3a_velocity_profile.pdf",
            "orientation_profile.pdf": "fig5_3b_orientation_rpy.pdf",
            "radar_point_counts.pdf":  "fig5_4_radar_point_counts.pdf",
        }
        for src, dst in copies.items():
            sp = os.path.join(args.figs, src)
            dp = os.path.join(args.out, dst)
            if os.path.exists(sp):
                import shutil
                shutil.copy2(sp, dp)
                print(f"  Copied {src} → {dst}")
    else:
        print(f"\n[SKIP] Trajectory figures: {odom_npz} not found.")
        print("       Run: python3 eval/01_extract_bag_data.py --bag <bag>")

    # ── Fig 5.5: ghost suppression comparison  (script 05) ────────────────
    baseline_pgm = os.path.join(args.maps, "baseline.pgm")
    a2_pgm       = os.path.join(args.maps, "ablation", "A2.pgm")
    if os.path.exists(baseline_pgm) and os.path.exists(a2_pgm):
        run_script("05_analyze_map.py", [
            "--map",    baseline_pgm,
            "--map2",   a2_pgm,
            "--labels", "Baseline (multipath rejection ON)",
                        "A2 (multipath rejection OFF)",
            "--out",    args.figs,
        ])
        sp = os.path.join(args.figs, "map_comparison.pdf")
        dp = os.path.join(args.out,  "fig5_5_ghost_suppression.pdf")
        if os.path.exists(sp):
            import shutil
            shutil.copy2(sp, dp)
            print(f"  Copied map_comparison.pdf → fig5_5_ghost_suppression.pdf")
    else:
        print(f"\n[SKIP] Fig 5.5: need {baseline_pgm} and {a2_pgm}")

    # ── Figs 5.6 + 5.7: ablation maps + metrics  (script 05) ──────────────
    ablation_dir = os.path.join(args.maps, "ablation")
    if os.path.isdir(ablation_dir) and list(Path(ablation_dir).glob("*.pgm")):
        run_script("05_analyze_map.py", [
            "--ablation-dir", ablation_dir,
            "--out", args.figs,
        ])
        import shutil
        for src, dst in [
            ("ablation_maps.pdf",    "fig5_6_ablation_maps.pdf"),
            ("ablation_metrics.pdf", "fig5_7_ablation_metrics.pdf"),
        ]:
            sp = os.path.join(args.figs, src)
            dp = os.path.join(args.out, dst)
            if os.path.exists(sp):
                shutil.copy2(sp, dp)
                print(f"  Copied {src} → {dst}")

        # Write ablation LaTeX table
        write_ablation_latex_table(
            os.path.join(args.figs, "ablation_metrics.json"),
            os.path.join(args.out,  "ablation_metrics_table.tex")
        )
    else:
        print(f"\n[SKIP] Ablation figures: no PGMs found in {ablation_dir}")
        print("       Run: ./eval/06_run_ablation.sh --bag <bag>")

    # ── Fig 5.10: map enhancement comparison  (script 08, simple mode) ──────
    enhanced_dir = os.path.join(args.maps, "enhanced")
    if os.path.isdir(enhanced_dir):
        # Re-run 08 with --simple to regenerate a clean 1×2 (RAW vs Auto+Median) PDF
        run_script("08_enhance_map.py", [
            "--map",     os.path.join(args.maps, "baseline.pgm"),
            "--out-dir", enhanced_dir,
            "--simple",
        ])
        import shutil
        sp = os.path.join(enhanced_dir, "enhancement_comparison.pdf")
        dp = os.path.join(args.out, "fig5_10_map_enhancement.pdf")
        if os.path.exists(sp):
            shutil.copy2(sp, dp)
            print(f"  Copied enhancement_comparison.pdf → fig5_10_map_enhancement.pdf")
        # Fig 5.11 (bar chart) is removed — no copy

        # Write simplified enhancement LaTeX table (2 rows only)
        enh_metrics_path = os.path.join(enhanced_dir, "enhancement_metrics.json")
        if os.path.exists(enh_metrics_path):
            with open(enh_metrics_path) as f:
                enh_data = json.load(f)
            tex_path = os.path.join(args.out, "enhancement_table.tex")
            display_names = {
                "raw":        "Raw (no filter)",
                "auto_median": "Auto-Enhanced + Median",
            }
            lines = [
                r"\begin{table}[htbp]",
                r"    \centering",
                r"    \caption{Map quality metrics before and after post-processing.}",
                r"    \label{tab:map_enhancement}",
                r"    \begin{tabular}{lrrrr}",
                r"        \toprule",
                r"        Variant & Occ.\ (\%) & Isolated pts & Wall (m) & Entropy \\",
                r"        \midrule",
            ]
            for key in ["raw", "auto_median"]:
                m = enh_data.get("variants", {}).get(key)
                if m is None:
                    continue
                name = display_names.get(key, key)
                bold = r"\textbf{" if key == "auto_median" else ""
                endb = r"}" if key == "auto_median" else ""
                lines.append(
                    f"        {bold}{name}{endb} & "
                    f"{bold}{m['occupied_fraction']*100:.1f}{endb} & "
                    f"{bold}{m['n_isolated_points']}{endb} & "
                    f"{bold}{m['wall_length_m']:.1f}{endb} & "
                    f"{bold}{m['mean_entropy_bits']:.3f}{endb} \\\\"
                )
            lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
            with open(tex_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"  Enhancement LaTeX table → {tex_path}")
    else:
        print(f"\n[SKIP] Fig 5.10: {enhanced_dir} not found")
        print("       Run: python3 eval/08_enhance_map.py --map eval/maps/baseline.pgm --simple")

    # ── Figs 5.8 + 5.9 + perf_table.tex  (script 04) ─────────────────────
    if os.path.exists(args.perf):
        run_script("04_plot_performance.py",
                   ["--data", args.data, "--out", args.figs])
        import shutil
        for src, dst in [
            ("performance_latency_bar.pdf", "fig5_8_performance_latency.pdf"),
            ("performance_cpu_ram.pdf",     "fig5_9_cpu_ram.pdf"),
            ("performance_table.tex",       "perf_table.tex"),
        ]:
            sp = os.path.join(args.figs, src)
            dp = os.path.join(args.out, dst)
            if os.path.exists(sp):
                shutil.copy2(sp, dp)
                print(f"  Copied {src} → {dst}")
    else:
        print(f"\n[SKIP] Performance figures: {args.perf} not found.")
        print("       Run: python3 eval/03_collect_live_performance.py --duration 120")

    # ── Experiment figures  (script 10) ──────────────────────────────────
    exp_maps_dir = os.path.join(args.maps, "experiments")
    exp_out_dir  = os.path.join(args.figs, "experiments")
    extra = ["--lap-split", args.lap_split] if args.lap_split else []
    run_script("10_generate_experiment_figures.py", [
        "--maps-dir",   exp_maps_dir,
        "--baseline",   os.path.join(args.maps, "baseline.pgm"),
        "--data",       os.path.join(args.data, "corridor"),
        "--out",        exp_out_dir,
        "--resolution", "0.21",
    ] + extra)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Thesis figures written to: {args.out}")
    print("=" * 60)
    files = sorted(Path(args.out).glob("*"))
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
