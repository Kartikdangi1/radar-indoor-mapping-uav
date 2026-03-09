#!/usr/bin/env python3
"""
04_plot_performance.py
======================
Generates thesis figures from performance_metrics.json produced by
03_collect_live_performance.py.

Output (eval/figures/):
    performance_latency_bar.pdf   — bar chart with error bars per node
    performance_cpu_ram.pdf       — CPU % and RAM over time
    performance_table.tex         — LaTeX table snippet (copy into thesis)

Usage:
    python3 eval/04_plot_performance.py [--data eval/data] [--out eval/figures]
"""

import argparse
import json
import os
import sys

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

# Display names for stages
STAGE_LABELS = {
    "preproc_odom":   "Preprocessing\n(odom path)",
    "preproc_map":    "Preprocessing\n(map path)",
    "fused_odom":     "Fused Odom\nNode",
    "occupancy_grid": "Occupancy\nGrid",
}

STAGE_LABELS_TEX = {
    "preproc_odom":   r"\texttt{radar\_preprocessing\_node} (odom)",
    "preproc_map":    r"\texttt{radar\_preprocessing\_mapping\_node}",
    "fused_odom":     r"\texttt{fused\_odom\_node}",
    "occupancy_grid": r"\texttt{temporal\_radar\_occupancy\_grid}",
}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Latency bar chart
# ---------------------------------------------------------------------------

def plot_latency_bar(latency: dict, out_path: str):
    stages = [s for s in STAGE_LABELS if s in latency and latency[s].get("count", 0) > 0]
    means  = [latency[s]["mean_ms"] for s in stages]
    stds   = [latency[s]["std_ms"]  for s in stages]
    p95s   = [latency[s]["p95_ms"]  for s in stages]
    counts = [latency[s]["count"]   for s in stages]

    x = np.arange(len(stages))
    width = 0.5
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                  color=colours[:len(stages)],
                  alpha=0.85, edgecolor="black", linewidth=0.6,
                  error_kw={"elinewidth": 1.2, "ecolor": "black"})

    # P95 markers — plotted above each bar
    ax.scatter(x, p95s, marker="D", color="red", s=35, zorder=5, label="P95 latency")

    # Mean value label: above the error cap (bar_height + std + small gap)
    gap = max(stds + p95s) * 0.06 + 0.3
    for bar, mean, std in zip(bars, means, stds):
        label_y = bar.get_height() + std + gap
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{mean:.1f} ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # n= count inside the bar near the bottom (white text)
    for bar, count in zip(bars, counts):
        bar_h = bar.get_height()
        # Only draw inside text if bar is tall enough to fit
        inside_y = min(bar_h * 0.15, bar_h - 0.5)
        if inside_y > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, inside_y,
                    f"n={count:,}", ha="center", va="bottom",
                    fontsize=7, color="white", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[s] for s in stages], fontsize=9,
                       multialignment="center")
    ax.set_ylabel("Wall-clock latency (ms)")
    ax.set_title("Pipeline Node Processing Latency (mean ± std,  ◆ P95)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    # Top headroom: enough space above the highest mean-label
    top = max(m + s + gap * 2.5 + 1.0 for m, s in zip(means, stds))
    ax.set_ylim(bottom=0, top=max(top, max(p95s) * 1.2))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CPU / RAM over time
# ---------------------------------------------------------------------------

def plot_cpu_ram(cpu_ram: list, out_path: str):
    if not cpu_ram:
        print("  [SKIP] No CPU/RAM data")
        return
    t   = [r["t"] for r in cpu_ram]
    cpu = [r["cpu_percent"] for r in cpu_ram]
    ram = [r["ram_mb"] for r in cpu_ram]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)

    ax1.plot(t, cpu, color="#F44336", linewidth=1.3)
    ax1.fill_between(t, cpu, alpha=0.15, color="#F44336")
    ax1.set_ylabel("CPU (%)")
    ax1.set_title("System Resource Usage During Bag Playback")
    ax1.grid(True, linewidth=0.4, alpha=0.5)
    ax1.axhline(np.mean(cpu), color="#F44336", linestyle="--",
                linewidth=0.8, label=f"Mean {np.mean(cpu):.1f}%")
    ax1.legend(fontsize=8)

    ax2.plot(t, ram, color="#2196F3", linewidth=1.3)
    ax2.fill_between(t, ram, alpha=0.15, color="#2196F3")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RAM (MB)")
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    ax2.axhline(np.mean(ram), color="#2196F3", linestyle="--",
                linewidth=0.8, label=f"Mean {np.mean(ram):.0f} MB")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_latex_table(latency: dict, freq: dict, out_path: str):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Processing time per node (mean $\pm$ std) and output frequency.}",
        r"    \label{tab:perf}",
        r"    \begin{tabular}{lrrl}",
        r"        \toprule",
        r"        Node & Mean (ms) & Std (ms) & Rate (Hz) \\",
        r"        \midrule",
    ]

    topic_to_stage = {
        "/PointCloudDetectionFiltered":  "preproc_odom",
        "/PointCloudDetectionMapping":   "preproc_map",
        "/fused_odom/odometry":          "fused_odom",
        "/map":                          "occupancy_grid",
    }
    stage_to_topic = {v: k for k, v in topic_to_stage.items()}

    for stage, label in STAGE_LABELS_TEX.items():
        if stage not in latency or latency[stage].get("count", 0) == 0:
            lines.append(f"        {label} & -- & -- & -- \\\\")
            continue
        s = latency[stage]
        out_topic = stage_to_topic.get(stage, "")
        hz = freq.get(out_topic, 0.0)
        lines.append(
            f"        {label} & "
            f"{s['mean_ms']:.2f} & "
            f"{s['std_ms']:.2f} & "
            f"{hz:.1f} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="eval/data")
    parser.add_argument("--out",  default="eval/figures")
    args = parser.parse_args()

    metrics_path = os.path.join(args.data, "performance_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"ERROR: {metrics_path} not found. Run 03_collect_live_performance.py first.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    data = load_json(metrics_path)

    latency = data.get("latency", {})
    freq    = data.get("frequency_hz", {})
    cpu_ram = data.get("cpu_ram", [])

    print("Generating performance figures...")

    plot_latency_bar(latency, os.path.join(args.out, "performance_latency_bar.pdf"))
    plot_cpu_ram(cpu_ram, os.path.join(args.out, "performance_cpu_ram.pdf"))
    write_latex_table(latency, freq, os.path.join(args.out, "performance_table.tex"))

    print("\n=== Summary ===")
    for stage, s in latency.items():
        if s.get("count", 0) > 0:
            print(f"  {stage:<25}  mean={s['mean_ms']:.2f} ms  std={s['std_ms']:.2f} ms  "
                  f"p95={s['p95_ms']:.2f} ms  (n={s['count']})")


if __name__ == "__main__":
    main()
