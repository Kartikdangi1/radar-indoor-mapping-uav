#!/usr/bin/env python3
"""
02_plot_trajectory.py
=====================
Generates thesis-quality trajectory and odometry figures from extracted data.

Odom NPZ columns: t, x, y, roll, pitch, yaw, vx, vy

Input:  eval/data/odom.npz  (from 03_collect_live_performance.py)
Output: eval/figures/
    trajectory_2d.pdf        — top-down X-Y trajectory coloured by speed
    trajectory_dual_lap.pdf  — two laps in different colours (requires --lap-split)
    velocity_profile.pdf     — Vx, Vy, ||V_xy|| vs time
    orientation_profile.pdf  — roll, pitch, yaw vs time
    radar_point_counts.pdf   — points/scan for each preprocessing stage
    imu_overview.pdf         — ax, ay  and  ωx, ωy, ωz vs time
    trajectory_stats.json    — quantitative summary

Usage:
    python3 eval/02_plot_trajectory.py [--data eval/data] [--out eval/figures]

    # Two-lap drift visualisation (split at a known timestamp):
    python3 eval/02_plot_trajectory.py --lap-split 1772903432.755
    # Or with an ISO datetime string (timezone-aware):
    python3 eval/02_plot_trajectory.py --lap-split "2026-03-07T18:10:32.755+01:00"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
THESIS_RC = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "lines.linewidth":   1.5,
}
plt.rcParams.update(THESIS_RC)

# odom column indices  (t, x, y, roll, pitch, yaw, vx, vy)
C_T, C_X, C_Y = 0, 1, 2
C_ROLL, C_PITCH, C_YAW = 3, 4, 5
C_VX, C_VY = 6, 7

C_TRAJ  = "#2196F3"
C_START = "#4CAF50"
C_END   = "#F44336"
C_LOOP  = "#FF9800"
C_LAP1  = "#2196F3"   # blue  — lap 1
C_LAP2  = "#FF9800"   # amber — lap 2
C_SPLIT = "#9C27B0"   # purple — lap split marker


def parse_lap_split(value: str) -> float:
    """Accept either a Unix float string or an ISO-8601 datetime string."""
    try:
        return float(value)
    except ValueError:
        pass
    # Try ISO-8601 with timezone (e.g. "2026-03-07T18:10:32.755+01:00")
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            raise ValueError("datetime string must include timezone (e.g. +01:00 for CET)")
        return dt.astimezone(timezone.utc).timestamp()
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Cannot parse --lap-split {value!r}: {exc}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: str):
    if not os.path.exists(path):
        return None
    return np.load(path)["data"]


def cumulative_distance(xy: np.ndarray) -> np.ndarray:
    diffs = np.diff(xy, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def colour_by_speed(xy: np.ndarray, speeds: np.ndarray, cmap="plasma"):
    points = xy.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(speeds.min(), speeds.max())
    return LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.8)


# ---------------------------------------------------------------------------
# Plot 1 — 2-D X/Y trajectory
# ---------------------------------------------------------------------------

def plot_trajectory_2d(odom: np.ndarray, loop_stamps, out_path: str,
                       room_w_m: float = 0.0, room_h_m: float = 0.0):
    t  = odom[:, C_T]
    x  = odom[:, C_X]
    y  = odom[:, C_Y]
    vx = odom[:, C_VX]
    vy = odom[:, C_VY]
    speed = np.sqrt(vx**2 + vy**2)

    fig, ax = plt.subplots(figsize=(6, 6))

    lc = colour_by_speed(odom[:, C_X:C_Y+1], speed)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Speed (m/s)")

    # Room boundary (centred on trajectory bounding box)
    if room_w_m > 0 and room_h_m > 0:
        cx = (x.min() + x.max()) / 2
        cy = (y.min() + y.max()) / 2
        rect = plt.Rectangle(
            (cx - room_w_m / 2, cy - room_h_m / 2),
            room_w_m, room_h_m,
            linewidth=1.5, edgecolor="#E91E63", facecolor="none",
            linestyle="--", zorder=2, label=f"Room {room_w_m:.2f}×{room_h_m:.2f} m"
        )
        ax.add_patch(rect)

    ax.scatter(x[0],  y[0],  color=C_START, s=80, zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], color=C_END,   s=80, marker="s", zorder=5, label="End")

    if loop_stamps is not None and len(loop_stamps) > 0:
        for lc_t in loop_stamps[:, 0]:
            idx = np.argmin(np.abs(t - lc_t))
            ax.scatter(x[idx], y[idx], color=C_LOOP, s=60, marker="*", zorder=6)
        ax.scatter([], [], color=C_LOOP, marker="*", s=60, label="Loop closure")

    drift = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    if drift > 0.1:
        ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[0], y[0]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    dist = cumulative_distance(odom[:, C_X:C_Y+1])
    total_dist = dist[-1]
    drift_pct  = 100.0 * drift / max(total_dist, 1e-6)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Fused Odometry Trajectory  "
        f"(total = {total_dist:.1f} m,  drift = {drift:.2f} m = {drift_pct:.1f}%)"
    )
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {
        "total_distance_m":     float(total_dist),
        "start_to_end_drift_m": float(drift),
        "drift_percent":        float(drift_pct),
        "duration_s":           float(t[-1] - t[0]),
        "n_loop_closures":      int(len(loop_stamps)) if loop_stamps is not None else 0,
    }


# ---------------------------------------------------------------------------
# Plot 2 — X/Y velocity profile  (no Z)
# ---------------------------------------------------------------------------

def plot_velocity_profile(odom: np.ndarray, out_path: str):
    t  = odom[:, C_T] - odom[0, C_T]
    vx = odom[:, C_VX]
    vy = odom[:, C_VY]
    speed = np.sqrt(vx**2 + vy**2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax = axes[0]
    ax.plot(t, vx, label="$v_x$", linewidth=1.2)
    ax.plot(t, vy, label="$v_y$", linewidth=1.2)
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.6)

    ax2 = axes[1]
    ax2.plot(t, speed, color=C_TRAJ, linewidth=1.4, label="$\\|v_{xy}\\|$")
    ax2.fill_between(t, speed, alpha=0.15, color=C_TRAJ)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle("Fused Odometry — X/Y Velocity Profile", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {
        "mean_speed_m_s": float(np.mean(speed)),
        "max_speed_m_s":  float(np.max(speed)),
    }


# ---------------------------------------------------------------------------
# Plot 3 — orientation profile  (roll / pitch / yaw)
# ---------------------------------------------------------------------------

def plot_orientation_profile(odom: np.ndarray, out_path: str):
    t     = odom[:, C_T] - odom[0, C_T]
    roll  = np.degrees(odom[:, C_ROLL])
    pitch = np.degrees(odom[:, C_PITCH])
    yaw   = np.degrees(odom[:, C_YAW])

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for ax, data, label, colour in [
        (axes[0], roll,  "Roll (°)",  "#E91E63"),
        (axes[1], pitch, "Pitch (°)", "#FF9800"),
        (axes[2], yaw,   "Yaw (°)",   "#2196F3"),
    ]:
        ax.plot(t, data, linewidth=1.2, color=colour)
        ax.set_ylabel(label)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Fused Odometry — Orientation (Roll / Pitch / Yaw)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — radar point count over time
# ---------------------------------------------------------------------------

def plot_radar_point_counts(radar_raw, radar_odom, radar_map, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    t0 = None

    for label, data, colour in [
        ("Raw (/PointCloudDetection)",   radar_raw,  "#607D8B"),
        ("Odom filter (−20 dB, 40°)",    radar_odom, C_TRAJ),
        ("Mapping filter (−4 dB, 70°)",  radar_map,  "#4CAF50"),
    ]:
        if data is None:
            continue
        t = data[:, 0]
        n = data[:, 1]
        if t0 is None:
            t0 = t[0]
        ax.plot(t - t0, n, label=label, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Points per scan")
    ax.set_title("ARS548 Radar — Points per Scan After Each Preprocessing Filter")
    ax.legend(loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    stats = {}
    for key, data in [("raw", radar_raw), ("odom_filter", radar_odom), ("map_filter", radar_map)]:
        if data is not None:
            stats[key] = {
                "mean_points":   float(np.mean(data[:, 1])),
                "median_points": float(np.median(data[:, 1])),
                "max_points":    float(np.max(data[:, 1])),
                "min_points":    float(np.min(data[:, 1])),
            }
    return stats


# ---------------------------------------------------------------------------
# Plot 5 — IMU:  ax, ay  |  ωx, ωy, ωz  (no az)
# ---------------------------------------------------------------------------

def plot_imu_overview(imu: np.ndarray, out_path: str):
    if imu is None:
        return
    t  = imu[:, 0] - imu[0, 0]
    ax_d, ay_d      = imu[:, 1], imu[:, 2]      # X/Y linear accel only
    gx, gy, gz      = imu[:, 4], imu[:, 5], imu[:, 6]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax = axes[0]
    ax.plot(t, ax_d, linewidth=0.6, alpha=0.8, label="$a_x$")
    ax.plot(t, ay_d, linewidth=0.6, alpha=0.8, label="$a_y$")
    ax.set_ylabel("Linear accel. (m/s²)")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    ax2 = axes[1]
    ax2.plot(t, np.degrees(gx), linewidth=0.6, alpha=0.8, label="$\\omega_x$ (roll rate)")
    ax2.plot(t, np.degrees(gy), linewidth=0.6, alpha=0.8, label="$\\omega_y$ (pitch rate)")
    ax2.plot(t, np.degrees(gz), linewidth=0.6, alpha=0.8, label="$\\omega_z$ (yaw rate)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular rate (°/s)")
    ax2.legend(ncol=3, loc="upper right")
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle("IMU Measurements — X/Y Accel + Roll/Pitch/Yaw Rates", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 6 — Dual-lap trajectory: two rounds in different colours to show drift
# ---------------------------------------------------------------------------

def plot_trajectory_dual_lap(odom: np.ndarray, split_t: float, out_path: str):
    """
    Split the trajectory at *split_t* (Unix seconds) and draw each lap in a
    distinct colour.  Annotates the start-to-end drift accumulated over both laps.
    """
    t = odom[:, C_T]
    x = odom[:, C_X]
    y = odom[:, C_Y]

    split_idx = int(np.argmin(np.abs(t - split_t)))

    if split_idx == 0 or split_idx == len(t) - 1:
        print(f"  [WARN] lap-split timestamp {split_t:.3f} is outside data range "
              f"[{t[0]:.3f}, {t[-1]:.3f}] — skipping dual-lap plot")
        return {}

    x1, y1 = x[:split_idx + 1], y[:split_idx + 1]
    x2, y2 = x[split_idx:],     y[split_idx:]

    dist1 = cumulative_distance(np.column_stack([x1, y1]))[-1]
    dist2 = cumulative_distance(np.column_stack([x2, y2]))[-1]
    drift = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    drift_pct = 100.0 * drift / max(dist1 + dist2, 1e-6)

    split_utc = datetime.utcfromtimestamp(split_t).strftime("%H:%M:%S UTC")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x1, y1, color=C_LAP1, linewidth=1.8, label=f"Lap 1  ({dist1:.1f} m)", zorder=2)
    ax.plot(x2, y2, color=C_LAP2, linewidth=1.8, label=f"Lap 2  ({dist2:.1f} m)", zorder=2)

    ax.scatter(x[0],         y[0],         color=C_START, s=100, zorder=5, label="Start")
    ax.scatter(x[split_idx], y[split_idx], color=C_SPLIT, s=100, marker="D", zorder=5,
               label=f"Lap split  ({split_utc})")
    ax.scatter(x[-1],        y[-1],        color=C_END,   s=100, marker="s", zorder=5,
               label=f"End  (drift {drift:.2f} m,  {drift_pct:.1f}%)")

    # Drift arrow from end back toward start
    if drift > 0.05:
        ax.annotate(
            "", xy=(x[0], y[0]), xytext=(x[-1], y[-1]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5,
                            connectionstyle="arc3,rad=0.3"),
            zorder=4,
        )
        mx = (x[0] + x[-1]) / 2
        my = (y[0] + y[-1]) / 2
        ax.text(mx, my, f"  drift\n  {drift:.2f} m", fontsize=8, color="gray", va="center")

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Two-Lap Trajectory  —  Drift = {drift:.2f} m ({drift_pct:.1f}%)"
        f"  after {dist1 + dist2:.1f} m total\n"
        "(Residual drift requires future pose-graph optimisation)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {
        "lap1_distance_m":      float(dist1),
        "lap2_distance_m":      float(dist2),
        "total_distance_m":     float(dist1 + dist2),
        "split_unix_t":         float(split_t),
        "split_idx":            int(split_idx),
        "start_to_end_drift_m": drift,
        "drift_percent":        drift_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="eval/data")
    parser.add_argument("--out",    default="eval/figures")
    parser.add_argument("--room-w", type=float, default=0.0,
                        help="Actual room width  in metres (draws reference rectangle)")
    parser.add_argument("--room-h", type=float, default=0.0,
                        help="Actual room height in metres (draws reference rectangle)")
    parser.add_argument("--lap-split", default=None,
                        help="Enable dual-lap drift plot. Pass a Unix timestamp (float) or "
                             "ISO-8601 datetime string with timezone, e.g. "
                             "\"2026-03-07T18:10:32.755+01:00\" or 1772903432.755")
    args = parser.parse_args()

    # Parse lap-split timestamp if provided
    lap_split_t = None
    if args.lap_split is not None:
        try:
            try:
                lap_split_t = float(args.lap_split)
            except ValueError:
                dt = datetime.fromisoformat(args.lap_split)
                if dt.tzinfo is None:
                    parser.error("--lap-split datetime must include timezone, e.g. +01:00")
                from datetime import timezone as _tz
                lap_split_t = dt.astimezone(_tz.utc).timestamp()
        except Exception as e:
            parser.error(f"Cannot parse --lap-split: {e}")

    os.makedirs(args.out, exist_ok=True)

    def load(name):
        p = os.path.join(args.data, f"{name}.npz")
        d = load_npz(p)
        print(f"  {'[OK]  ' if d is not None else '[MISS]'} {p}" +
              (f"  shape={d.shape}" if d is not None else ""))
        return d

    print("Loading extracted data...")
    odom          = load("odom")
    imu           = load("imu")
    radar_raw     = load("radar_raw")
    radar_odom    = load("radar_odom")
    radar_map     = load("radar_map")
    loop_closures = load("loop_closures")

    if odom is None:
        print("\nERROR: No odometry data. Run 03_collect_live_performance.py first.")
        sys.exit(1)

    # Migrate old 7-column format (t,x,y,z,vx,vy,vz) → new 8-column (t,x,y,r,p,y,vx,vy)
    if odom.shape[1] == 7:
        print("  [INFO] Old odom format detected (7 cols) — RPY set to 0 until next live run")
        zeros = np.zeros((len(odom), 1))
        odom = np.hstack([odom[:, :3], zeros, zeros, zeros, odom[:, 4:6]])

    all_stats = {}

    print("\nGenerating figures...")

    all_stats.update(plot_trajectory_2d(
        odom, loop_closures,
        os.path.join(args.out, "trajectory_2d.pdf"),
        room_w_m=args.room_w, room_h_m=args.room_h,
    ))

    all_stats.update(plot_velocity_profile(
        odom,
        os.path.join(args.out, "velocity_profile.pdf")
    ))

    plot_orientation_profile(
        odom,
        os.path.join(args.out, "orientation_profile.pdf")
    )

    all_stats["radar_stats"] = plot_radar_point_counts(
        radar_raw, radar_odom, radar_map,
        os.path.join(args.out, "radar_point_counts.pdf")
    )

    if imu is not None:
        plot_imu_overview(imu, os.path.join(args.out, "imu_overview.pdf"))

    if lap_split_t is not None:
        print(f"\nGenerating dual-lap drift figure (split at Unix {lap_split_t:.3f}) ...")
        lap_stats = plot_trajectory_dual_lap(
            odom, lap_split_t,
            os.path.join(args.out, "trajectory_dual_lap.pdf"),
        )
        if lap_stats:
            all_stats["dual_lap"] = lap_stats

    stats_path = os.path.join(args.out, "trajectory_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStatistics summary: {stats_path}")
    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
