#!/usr/bin/env python3
"""
08_enhance_map.py
=================
Headless map post-processing — extracts the core filter + auto-enhance logic
from Map_Enhancer_Wizard (~/Thesis/Map_Enhancer_Wizard/code) and applies it
to a raw occupancy-grid PGM without any Tkinter UI.

Produces four output maps (all saved to --out-dir):
  raw.pgm          — original (unchanged copy)
  auto_enhanced.pgm — auto-enhance algorithm (Otsu threshold + adaptive morph)
  median.pgm        — full median filter (kernel=9) + threshold
  gaussian.pgm      — full Gaussian blur  (kernel=9) + threshold

Also produces:
  enhancement_comparison.pdf  — 2×2 grid comparison figure
  enhancement_metrics.json    — per-variant map quality metrics

Usage:
    python3 eval/08_enhance_map.py --map eval/maps/baseline.pgm [--out-dir eval/maps/enhanced]
    python3 eval/08_enhance_map.py --map eval/maps/baseline.pgm --resolution 0.07
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage


# ---------------------------------------------------------------------------
# Inline utilities (from Map_Enhancer_Wizard/code/utils/)
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def _morph_kernel(size: int):
    size = _clamp(int(size), 1, 99)
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


# ---------------------------------------------------------------------------
# Core filter pipeline (mirrors MapEnhancerWizard.apply_filters)
# ---------------------------------------------------------------------------

def apply_filters(img: np.ndarray, *,
                  median: int = 0,
                  blur: int = 0,
                  threshold: float = 0.5,
                  use_adaptive: bool = False,
                  opening: int = 0,
                  closing: int = 0,
                  dilation: int = 0,
                  erosion: int = 0) -> np.ndarray:
    """
    Replicate MapEnhancerWizard.apply_filters() without Tkinter.
    All parameters match the wizard's UI variables.
    """
    out = img.copy()

    # 1. Median blur (salt-and-pepper removal)
    med = _clamp(int(median), 0, 99)
    if med > 0:
        k = med if med % 2 == 1 else med + 1
        out = cv2.medianBlur(out, k)

    # 2. Gaussian blur (smoothing)
    g = _clamp(int(blur), 0, 99)
    if g > 0:
        k = g if g % 2 == 1 else g + 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    # 3. Threshold
    if use_adaptive:
        block = max(15, (min(out.shape[:2]) // 30) | 1)
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 5)
    else:
        thr = _clamp(float(threshold), 0.0, 1.0)
        _, out = cv2.threshold(out, int(thr * 255), 255, cv2.THRESH_BINARY)

    # 4. Morphological operations
    op = _clamp(int(opening), 0, 99)
    if op > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _morph_kernel(op))

    cl = _clamp(int(closing), 0, 99)
    if cl > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _morph_kernel(cl))

    dil = _clamp(int(dilation), 0, 99)
    if dil > 0:
        out = cv2.dilate(out, _morph_kernel(dil))

    ero = _clamp(int(erosion), 0, 99)
    if ero > 0:
        out = cv2.erode(out, _morph_kernel(ero))

    return out


# ---------------------------------------------------------------------------
# Auto-enhance algorithm (mirrors MapEnhancerWizard.auto_enhance)
# ---------------------------------------------------------------------------

def compute_auto_enhance_params(img: np.ndarray,
                                resolution_m: float = 0.07) -> dict:
    """
    Run the auto-enhance heuristic and return a parameter dict.
    Replicates MapEnhancerWizard.auto_enhance() without UI side-effects.
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    total = img.size
    mean_val = float((hist * np.arange(256)).sum() / max(total, 1))

    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = float(lap.var())

    try:
        ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = ret / 255.0
        use_adapt = lap_var > 120.0
        if use_adapt:
            thr = 0.5
    except Exception:
        thr = 0.5
        use_adapt = False

    if lap_var < 30:
        g, med = 0, 0
    elif lap_var < 120:
        g, med = 1, 1
    else:
        g, med = 3, 3

    if use_adapt:
        block = max(15, (min(img.shape[:2]) // 30) | 1)
        bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block, 5)
    else:
        _, bw = cv2.threshold(img, int(thr * 255), 255, cv2.THRESH_BINARY)

    dark_ratio   = hist[:64].sum() / total
    bright_ratio = hist[192:].sum() / total
    obstacles_are_black = dark_ratio >= bright_ratio
    bin_obs = (bw == (0 if obstacles_are_black else 255)).astype(np.uint8) * 255

    if bin_obs.max() > 0:
        dist = cv2.distanceTransform(bin_obs, cv2.DIST_L2, 3)
        edge = cv2.Canny(bin_obs, 50, 150)
        dvals = dist[edge > 0]
        mean_thick = float(dvals.mean() * 2.0) if dvals.size else 1.0
    else:
        mean_thick = 1.0

    known_mask = (img != 205) if 205 in np.unique(img) else np.ones_like(img, dtype=bool)
    occ_ratio  = float((bin_obs > 0).sum()) / float(known_mask.sum() + 1e-6)
    res_m      = _safe_float(resolution_m, 0.05)

    target_wall_m  = 0.15
    target_px      = _clamp(int(round(target_wall_m / max(res_m, 1e-6))), 1, 15)
    dilation = erosion = opening = closing = 0

    if occ_ratio < 0.02 or lap_var > 120:
        opening = _clamp(int(round(0.05 / res_m)), 0, 7)
    closing = _clamp(int(round(0.04 / res_m)), 0, 7)
    if mean_thick < target_px:
        dilation = _clamp(int(round(target_px - mean_thick)), 0, 8)
    elif mean_thick > target_px * 1.8:
        erosion = _clamp(int(round(mean_thick - target_px)), 0, 8)
    if opening and erosion:
        erosion = max(0, erosion - 1)

    return {
        "threshold":    round(thr, 3),
        "use_adaptive": bool(use_adapt),
        "blur":         int(g),
        "median":       int(med),
        "opening":      int(opening),
        "closing":      int(closing),
        "dilation":     int(dilation),
        "erosion":      int(erosion),
        # diagnostics
        "_lap_var":     round(lap_var, 2),
        "_mean_val":    round(mean_val, 2),
        "_mean_thick":  round(mean_thick, 2),
        "_occ_ratio":   round(occ_ratio, 4),
        "_target_px":   target_px,
    }


# ---------------------------------------------------------------------------
# Map quality metrics (shared with 05_analyze_map.py)
# ---------------------------------------------------------------------------

def compute_map_metrics(img_gray: np.ndarray, resolution: float = 0.07) -> dict:
    """
    Compute quality metrics on a processed grayscale PGM.
    Pixels: 0=occupied(black), 205=unknown(grey), 255=free(white).
    """
    occ_mask  = img_gray < 50
    free_mask = img_gray > 200
    unkn_mask = ~occ_mask & ~free_mask
    total     = img_gray.size

    occ_frac  = float(occ_mask.sum()) / total
    free_frac = float(free_mask.sum()) / total
    unkn_frac = float(unkn_mask.sum()) / total

    # Shannon entropy on known pixels (treat pixel value as probability proxy)
    known = img_gray[~unkn_mask].astype(np.float32) / 255.0
    eps = 1e-9
    p = np.clip(known, eps, 1 - eps)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    mean_entropy = float(np.nanmean(H)) if len(H) > 0 else 0.0

    # Connected components of occupied cells
    labeled, n = ndimage.label(occ_mask.astype(np.int8))
    if n > 0:
        sizes = ndimage.sum(occ_mask.astype(np.int8), labeled, range(1, n + 1))
        sizes = np.array(sizes)
        n_wall_segs   = int((sizes >= 3).sum())
        n_isolated    = int((sizes < 3).sum())
    else:
        n_wall_segs = n_isolated = 0

    # Wall length (occupied boundary pixels)
    eroded = ndimage.binary_erosion(occ_mask, iterations=1)
    wall_boundary = occ_mask & ~eroded
    wall_length_m = float(wall_boundary.sum()) * resolution

    explored_m2 = float((occ_mask | free_mask).sum()) * resolution ** 2

    return {
        "occupied_fraction":  round(occ_frac,  4),
        "free_fraction":      round(free_frac,  4),
        "unknown_fraction":   round(unkn_frac,  4),
        "mean_entropy_bits":  round(mean_entropy, 4),
        "n_wall_segments":    n_wall_segs,
        "n_isolated_points":  n_isolated,
        "wall_length_m":      round(wall_length_m, 2),
        "explored_m2":        round(explored_m2,   2),
    }


# ---------------------------------------------------------------------------
# PGM I/O
# ---------------------------------------------------------------------------

def load_pgm(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


def save_pgm(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

THESIS_RC = {
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
}
plt.rcParams.update(THESIS_RC)


def plot_comparison(variants: list, out_path: str, annotate: bool = True):
    """
    variants: list of (label, img_gray, metrics_dict)
    Produces a 1×N comparison grid.
    annotate: if True, add metric textboxes on each panel (default full-mode).
              Pass annotate=False for clean thesis panels with no overlay text.
    """
    n = len(variants)
    ncols = min(n, 4)
    nrows = 1 + (n > 4)   # second row if >4 variants
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows))
    axes_flat = np.array(axes).ravel()

    for i, (label, img, metrics) in enumerate(variants):
        ax = axes_flat[i]
        # Display: invert so black=occupied, white=free (standard map view)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

        if annotate:
            # Metrics annotation
            m = metrics
            info = (
                f"Occ: {m['occupied_fraction']*100:.1f}%  "
                f"Isolated: {m['n_isolated_points']}\n"
                f"Wall: {m['wall_length_m']:.1f} m  "
                f"H: {m['mean_entropy_bits']:.3f}"
            )
            ax.text(0.01, 0.02, info, transform=ax.transAxes, fontsize=7,
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Map Enhancement Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_bar(variants: list, out_path: str):
    """Bar chart comparing key metrics across variants."""
    labels  = [v[0] for v in variants]
    metrics_list = [v[2] for v in variants]

    keys_to_plot = [
        ("occupied_fraction",  "Occupied fraction"),
        ("n_isolated_points",  "Isolated points (noise)"),
        ("wall_length_m",      "Wall length (m)"),
        ("mean_entropy_bits",  "Mean entropy (bits)"),
    ]

    fig, axes = plt.subplots(1, len(keys_to_plot), figsize=(4.5 * len(keys_to_plot), 4))
    x = np.arange(len(labels))
    colours = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    for ax, (key, title) in zip(axes, keys_to_plot):
        vals = [m[key] for m in metrics_list]
        bars = ax.bar(x, vals, color=colours[:len(labels)], alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.set_ylim(bottom=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", fontsize=7)

    fig.suptitle("Enhancement Variant Metrics", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_variants(median_k: int, gaussian_k: int,
                    simple: bool = False) -> list:
    """Return the processing chain for the current kernel configuration.

    simple=True: only two entries — raw and auto_median (Auto-Enhanced + Median),
                 intended for the simplified 1×2 thesis figure.
    """
    if simple:
        k = median_k if median_k % 2 == 1 else median_k + 1
        return [
            ("raw",        {}, "raw"),
            ("auto_median", {"median": k}, "auto_enhanced"),
        ]
    variants = [
        ("raw",           {},                  "raw"),
        ("auto_enhanced", None,                "raw"),
    ]
    if median_k > 0:
        k = median_k if median_k % 2 == 1 else median_k + 1
        variants.append(("median",   {"median": k}, "auto_enhanced"))
    if gaussian_k > 0:
        k = gaussian_k if gaussian_k % 2 == 1 else gaussian_k + 1
        variants.append(("gaussian", {"blur": k},   "auto_enhanced"))
    return variants


def run(pgm_path: str, out_dir: str, resolution: float = 0.07,
        median_k: int = 6, gaussian_k: int = 6, simple: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    img = load_pgm(pgm_path)
    print(f"  Loaded: {pgm_path}  shape={img.shape}")

    # Compute auto-enhance params once
    ae_params = compute_auto_enhance_params(img, resolution_m=resolution)
    print(f"  Auto-enhance params: {ae_params}")

    variants_out = []
    all_metrics  = {}
    ae_params_saved = {k: v for k, v in ae_params.items() if not k.startswith("_")}
    # Pre-compute auto_enhanced so it can be used as a base in simple mode
    _ae_filter_kwargs = {k: v for k, v in ae_params.items() if not k.startswith("_")}
    saved_imgs: dict = {
        "auto_enhanced": apply_filters(img, **_ae_filter_kwargs)
    }

    for name, kwargs, base in _build_variants(median_k, gaussian_k, simple=simple):
        src = img if base == "raw" else saved_imgs.get(base, saved_imgs.get("auto_enhanced", img))

        if name == "raw":
            processed = src.copy()
        elif kwargs is None:
            # Auto-enhance applied to src
            filter_kwargs = {k: v for k, v in ae_params.items() if not k.startswith("_")}
            processed = apply_filters(src, **filter_kwargs)
        else:
            processed = apply_filters(src, **kwargs)

        saved_imgs[name] = processed

        pgm_out = os.path.join(out_dir, f"{name}.pgm")
        save_pgm(processed, pgm_out)

        metrics = compute_map_metrics(processed, resolution)
        all_metrics[name] = metrics
        # Use a human-readable label; auto_median gets a descriptive name for thesis
        if name == "auto_median":
            display_label = "Auto-Enhanced + Median"
        else:
            display_label = name.replace("_", " ").title()
        variants_out.append((display_label, processed, metrics))

        print(f"  [{name}] occ={metrics['occupied_fraction']*100:.1f}%  "
              f"isolated={metrics['n_isolated_points']}  "
              f"wall={metrics['wall_length_m']:.1f}m  → {pgm_out}")

    # Save metrics JSON
    result = {
        "variants": all_metrics,
        "auto_enhance_params": ae_params_saved,
        "resolution_m": resolution,
    }
    json_path = os.path.join(out_dir, "enhancement_metrics.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Metrics: {json_path}")

    # Save auto-enhance params separately for reference
    params_path = os.path.join(out_dir, "auto_enhance_params.json")
    with open(params_path, "w") as f:
        json.dump(ae_params, f, indent=2)

    # Figures
    plot_comparison(
        variants_out,
        os.path.join(out_dir, "enhancement_comparison.pdf"),
        annotate=not simple,
    )
    if not simple:
        plot_metrics_bar(
            variants_out,
            os.path.join(out_dir, "enhancement_metrics_bar.pdf")
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",        required=True, help="Input PGM path")
    parser.add_argument("--out-dir",    default="eval/maps/enhanced")
    parser.add_argument("--resolution", type=float, default=0.07,
                        help="Map resolution m/cell (default 0.07)")
    parser.add_argument("--median-k",   type=int, default=6,
                        help="Median filter kernel applied on auto_enhanced (0=off, default 6)")
    parser.add_argument("--gaussian-k", type=int, default=6,
                        help="Gaussian filter kernel applied on auto_enhanced (0=off, default 6)")
    parser.add_argument("--simple", action="store_true", default=False,
                        help="Simple mode: produce only RAW and Auto-Enhanced+Median (1×2 figure, "
                             "no metric annotations, no bar chart)")
    args = parser.parse_args()

    if not os.path.exists(args.map):
        print(f"ERROR: {args.map} not found")
        sys.exit(1)

    print(f"Enhancing: {args.map}  (median_k={args.median_k}, gaussian_k={args.gaussian_k}"
          + ("  [simple mode]" if args.simple else "") + ")")
    result = run(args.map, args.out_dir, args.resolution,
                 median_k=args.median_k, gaussian_k=args.gaussian_k, simple=args.simple)

    print("\n=== Enhancement Summary ===")
    for name, m in result["variants"].items():
        print(f"  {name:<15}  occ={m['occupied_fraction']*100:.1f}%  "
              f"noise={m['n_isolated_points']}  wall={m['wall_length_m']:.1f}m  "
              f"H={m['mean_entropy_bits']:.3f}")


if __name__ == "__main__":
    main()
