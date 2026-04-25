"""
Underexposure Threshold Tuner
==============================
Runs ONLY the underexposure detector in isolation and sweeps
multiple threshold combinations to find the optimal values.

Focuses on two tunable parameters:
    under_exposure_threshold  — pixel value below which a pixel is "dark"
    exposure_pixel_ratio      — fraction of dark pixels needed to flag fault

Also checks night images at every threshold combination to ensure
zero false positives on night frames.

Run:
    python tune_underexposure.py
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = "F:/Wildfire_Camera/Dataset_images/generated_test_dataset"
OUTPUT_DIR  = "underexposure_tuning_results"
TARGET_SIZE = (1280, 960)

# Threshold sweep ranges — test every combination
UNDER_THRESH_VALUES  = [5, 8, 10, 12, 15, 18, 20, 25, 30]   # pixel darkness cutoff
RATIO_THRESH_VALUES  = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]  # fraction of dark pixels

# Current values for baseline comparison
CURRENT_UNDER_THRESH = 10
CURRENT_RATIO_THRESH = 0.10
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def load_images(folder):
    if not os.path.exists(folder):
        return []
    paths = (sorted(Path(folder).glob("*.jpg")) +
             sorted(Path(folder).glob("*.png")))
    result = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            result.append((str(p), cv2.resize(frame, TARGET_SIZE)))
    return result


# ── Load images ───────────────────────────────────────────────────────────────
print("\nLoading images...")
good_imgs     = load_images(os.path.join(DATASET_DIR, "good"))
underexp_imgs = load_images(os.path.join(DATASET_DIR, "underexposed"))

night_imgs = [(p, f) for p, f in good_imgs if "night" in Path(p).stem.lower()]
day_imgs   = [(p, f) for p, f in good_imgs if "night" not in Path(p).stem.lower()]

print(f"  Good images       : {len(good_imgs)} "
      f"({len(day_imgs)} day + {len(night_imgs)} night)")
print(f"  Underexposed imgs : {len(underexp_imgs)}")

if not underexp_imgs:
    print(f"ERROR: No underexposed images in {DATASET_DIR}/underexposed")
    exit(1)

# ── Pre-compute dark pixel ratios for all images at every threshold ───────────
print("\nPre-computing dark pixel ratios...")

def get_dark_ratios(images, pixel_thresh):
    """Compute fraction of pixels below pixel_thresh for each image."""
    ratios = []
    for _, frame in images:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = np.sum(gray < pixel_thresh) / gray.size
        ratios.append(ratio)
    return ratios


# Pre-compute for all pixel threshold values
good_ratios_cache    = {}
fault_ratios_cache   = {}
night_ratios_cache   = {}

for pt in UNDER_THRESH_VALUES:
    good_ratios_cache[pt]  = get_dark_ratios(good_imgs,     pt)
    fault_ratios_cache[pt] = get_dark_ratios(underexp_imgs, pt)
    night_ratios_cache[pt] = get_dark_ratios(night_imgs,    pt)

print("  Done.")


# ── Show current threshold performance first ──────────────────────────────────
print(f"\n{'='*60}")
print(f"  CURRENT THRESHOLD PERFORMANCE (baseline)")
print(f"{'='*60}")
print(f"  under_exposure_threshold = {CURRENT_UNDER_THRESH}")
print(f"  exposure_pixel_ratio     = {CURRENT_RATIO_THRESH}")

curr_fault_detected = sum(
    1 for r in fault_ratios_cache[CURRENT_UNDER_THRESH]
    if r > CURRENT_RATIO_THRESH
)
curr_good_fp = sum(
    1 for r in good_ratios_cache[CURRENT_UNDER_THRESH]
    if r > CURRENT_RATIO_THRESH
)
curr_night_fp = sum(
    1 for r in night_ratios_cache[CURRENT_UNDER_THRESH]
    if r > CURRENT_RATIO_THRESH
)

print(f"\n  Fault detection rate : {curr_fault_detected}/{len(underexp_imgs)} "
      f"({100*curr_fault_detected/len(underexp_imgs):.1f}%)")
print(f"  Good false positives : {curr_good_fp}/{len(good_imgs)} "
      f"({100*curr_good_fp/len(good_imgs):.1f}%)")
print(f"  Night false positives: {curr_night_fp}/{len(night_imgs)} "
      f"({'✓ PASS' if curr_night_fp == 0 else '✗ FAIL'})")


# ── Sweep all threshold combinations ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  THRESHOLD SWEEP")
print(f"{'='*60}")
print(f"  Testing {len(UNDER_THRESH_VALUES)} × "
      f"{len(RATIO_THRESH_VALUES)} = "
      f"{len(UNDER_THRESH_VALUES)*len(RATIO_THRESH_VALUES)} combinations...")
print(f"  ✗ = night images flagged (unacceptable)")
print()

results = []

header = (f"  {'under_thr':>10} {'ratio_thr':>10} "
          f"{'recall':>8} {'fp_rate':>8} {'night_fp':>10} {'status':>12}")
print(header)
print(f"  {'-'*62}")

for pt in UNDER_THRESH_VALUES:
    for rt in RATIO_THRESH_VALUES:
        fault_detected = sum(1 for r in fault_ratios_cache[pt] if r > rt)
        good_fp        = sum(1 for r in good_ratios_cache[pt]  if r > rt)
        night_fp       = sum(1 for r in night_ratios_cache[pt] if r > rt)

        recall   = fault_detected / len(underexp_imgs) * 100
        fp_rate  = good_fp        / len(good_imgs)     * 100
        night_ok = night_fp == 0

        status = "✓ SAFE" if night_ok else "✗ NIGHT FP"

        results.append({
            "under_thresh":    pt,
            "ratio_thresh":    rt,
            "fault_detected":  fault_detected,
            "total_fault":     len(underexp_imgs),
            "recall_pct":      round(recall, 1),
            "good_fp":         good_fp,
            "good_fp_pct":     round(fp_rate, 1),
            "night_fp":        night_fp,
            "night_ok":        night_ok,
        })

        # Highlight promising combinations
        highlight = recall >= 80 and night_ok and fp_rate < 5
        marker    = " ◄" if highlight else ""

        print(f"  {pt:>10} {rt:>10.3f} "
              f"{recall:>7.1f}% {fp_rate:>7.1f}% "
              f"{night_fp:>10} {status:>12}{marker}")

    print()   # blank line between pixel threshold groups


# ── Find best combination ─────────────────────────────────────────────────────
# Best = highest recall with zero night false positives and <5% good FP rate
safe_results = [r for r in results if r["night_ok"] and r["good_fp_pct"] < 5.0]

if safe_results:
    best = max(safe_results, key=lambda r: (r["recall_pct"], -r["good_fp_pct"]))
else:
    # Relax good FP constraint if nothing found under 5%
    safe_results = [r for r in results if r["night_ok"]]
    best = max(safe_results, key=lambda r: r["recall_pct"]) if safe_results else None

print(f"\n{'='*60}")
print(f"  RECOMMENDED THRESHOLDS")
print(f"{'='*60}")

if best:
    print(f"\n  under_exposure_threshold : {best['under_thresh']}  "
          f"(was {CURRENT_UNDER_THRESH})")
    print(f"  exposure_pixel_ratio     : {best['ratio_thresh']}  "
          f"(was {CURRENT_RATIO_THRESH})")
    print(f"\n  Fault detection (recall) : {best['recall_pct']}%  "
          f"(was {100*curr_fault_detected/len(underexp_imgs):.1f}%)")
    print(f"  Good FP rate             : {best['good_fp_pct']}%")
    print(f"  Night FP rate            : {best['night_fp']}/{len(night_imgs)}  ✓ SAFE")

    print(f"\n  Update in config_setup.json:")
    print(f'  "under_exposure_threshold": {best["under_thresh"]},')
    print(f'  "exposure_pixel_ratio":     {best["ratio_thresh"]}')
else:
    print("\n  No safe combination found — all options produce night false positives.")
    print("  Consider collecting more diverse night images or")
    print("  using time-of-day awareness to skip underexposure check at night.")


# ── Score distribution plot ───────────────────────────────────────────────────
print(f"\nGenerating plots...")

# Use best thresholds for the distribution plot
plot_pt = best["under_thresh"] if best else CURRENT_UNDER_THRESH
plot_rt = best["ratio_thresh"] if best else CURRENT_RATIO_THRESH

good_scores  = good_ratios_cache[plot_pt]
fault_scores = fault_ratios_cache[plot_pt]
night_scores = night_ratios_cache[plot_pt]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# Scatter
ax1.scatter(range(len(good_scores)),  good_scores,
            color="green", label="Good (day)", s=20, alpha=0.5)
ax1.scatter(range(len(good_scores), len(good_scores) + len(night_scores)),
            night_scores,
            color="blue", label="Good (night)", s=20, alpha=0.6, marker="^")
ax1.scatter(range(len(good_scores) + len(night_scores),
                  len(good_scores) + len(night_scores) + len(fault_scores)),
            fault_scores,
            color="red", label="Underexposed fault", s=20, alpha=0.5, marker="x")
ax1.axhline(y=plot_rt, color="orange", linestyle="--",
            linewidth=2, label=f"Threshold={plot_rt}")
ax1.set_title(f"Dark pixel ratio (pixel<{plot_pt})")
ax1.set_ylabel("Dark pixel ratio")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Histogram
ax2.hist(good_scores,  bins=30, color="green", alpha=0.5, label="Good (day)")
ax2.hist(night_scores, bins=30, color="blue",  alpha=0.5, label="Good (night)")
ax2.hist(fault_scores, bins=30, color="red",   alpha=0.5, label="Underexposed")
ax2.axvline(x=plot_rt, color="orange", linestyle="--",
            linewidth=2, label=f"Threshold={plot_rt}")
ax2.set_title(f"Distribution (pixel<{plot_pt})")
ax2.set_xlabel("Dark pixel ratio")
ax2.set_ylabel("Count")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle(f"Underexposure Tuning — best: pixel<{plot_pt}, ratio>{plot_rt}",
             fontsize=11)
plt.tight_layout()
dist_path = f"{OUTPUT_DIR}/underexposure_distribution.png"
plt.savefig(dist_path, dpi=150)
plt.close()
print(f"  {PASS} Distribution plot → {dist_path}")

# Heatmap — recall across all threshold combinations (night-safe only)
recall_grid = np.zeros((len(UNDER_THRESH_VALUES), len(RATIO_THRESH_VALUES)))
mask_grid   = np.zeros_like(recall_grid, dtype=bool)

for i, pt in enumerate(UNDER_THRESH_VALUES):
    for j, rt in enumerate(RATIO_THRESH_VALUES):
        r = next(x for x in results
                 if x["under_thresh"] == pt and x["ratio_thresh"] == rt)
        recall_grid[i, j] = r["recall_pct"]
        mask_grid[i, j]   = not r["night_ok"]   # True = unsafe (has night FP)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(recall_grid, cmap="RdYlGn", aspect="auto",
               vmin=0, vmax=100)

# Cross out unsafe combinations
for i in range(len(UNDER_THRESH_VALUES)):
    for j in range(len(RATIO_THRESH_VALUES)):
        val = f"{recall_grid[i,j]:.0f}%"
        if mask_grid[i, j]:
            ax.text(j, i, "✗", ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold")
        else:
            ax.text(j, i, val, ha="center", va="center",
                    color="black", fontsize=9)

ax.set_xticks(range(len(RATIO_THRESH_VALUES)))
ax.set_yticks(range(len(UNDER_THRESH_VALUES)))
ax.set_xticklabels([str(r) for r in RATIO_THRESH_VALUES])
ax.set_yticklabels([str(p) for p in UNDER_THRESH_VALUES])
ax.set_xlabel("exposure_pixel_ratio (ratio threshold)")
ax.set_ylabel("under_exposure_threshold (pixel darkness cutoff)")
ax.set_title("Recall % across threshold combinations\n"
             "✗ = night false positives (unsafe)  |  Green = high recall")
plt.colorbar(im, ax=ax, label="Recall %")

if best:
    bi = UNDER_THRESH_VALUES.index(best["under_thresh"])
    bj = RATIO_THRESH_VALUES.index(best["ratio_thresh"])
    ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1,
                                fill=False, edgecolor="blue",
                                linewidth=3, label="Recommended"))
    ax.legend(fontsize=9)

plt.tight_layout()
heatmap_path = f"{OUTPUT_DIR}/threshold_heatmap.png"
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"  {PASS} Threshold heatmap → {heatmap_path}")

print(f"\n{'='*60}")
print(f"  Done. Results in {OUTPUT_DIR}/")
print(f"{'='*60}\n")