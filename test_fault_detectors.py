
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS — update DATASET_DIR before running
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = "F:/Wildfire_Camera/Dataset_images/generated_test_dataset"
OUTPUT_DIR  = "fault_detector_test_results"
TARGET_SIZE = (1280, 960)

# Starting thresholds — Phase 7 will suggest better values from your data
THRESHOLDS = {
    "blur_laplacian_threshold":   100.0,
    "noise_laplacian_threshold":  3000.0,
    "over_exposure_pixel_ratio":  0.10,    
    "under_exposure_pixel_ratio": 0.02,    # new separate key
    "over_exposure_threshold":    245,
    "under_exposure_threshold":   25,
    "pov_correlation_threshold":  0.4,
}
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

phase_results = {}


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_images(folder: str) -> list:
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


def print_phase_header(num, title):
    print(f"\n{'='*60}")
    print(f"  Phase {num} — {title}")
    print(f"{'='*60}")


def print_result(name, detected, total, expected_detected):
    if total == 0:
        print(f"  - {name}: no images found")
        return 0, 0
    rate   = detected / total * 100
    target = detected if expected_detected else total - detected
    icon   = PASS if target == total else (WARN if rate >= 80 else FAIL)
    label  = "detected" if expected_detected else "clean (not flagged)"
    print(f"  {icon} {name}: {target}/{total} {label} ({rate:.1f}%)")
    return target, total


def compute_metrics(scores_fault, scores_good, threshold, higher_is_fault):
    if not scores_fault or not scores_good:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "precision": 0, "recall": 0, "accuracy": 0,
                "suggested_threshold": threshold}

    if higher_is_fault:
        tp = sum(1 for s in scores_fault if s >  threshold)
        fp = sum(1 for s in scores_good  if s >  threshold)
        fn = sum(1 for s in scores_fault if s <= threshold)
        tn = sum(1 for s in scores_good  if s <= threshold)
    else:
        tp = sum(1 for s in scores_fault if s <  threshold)
        fp = sum(1 for s in scores_good  if s <  threshold)
        fn = sum(1 for s in scores_fault if s >= threshold)
        tn = sum(1 for s in scores_good  if s >= threshold)

    suggested = round((np.mean(scores_fault) + np.mean(scores_good)) / 2, 2)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":  tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall":     tp / (tp + fn) if (tp + fn) > 0 else 0,
        "accuracy":   (tp + tn) / (tp + fp + fn + tn) if (tp+fp+fn+tn) > 0 else 0,
        "suggested_threshold": suggested,
    }


def hsv_correlation(f1, f2):
    score = 0.0
    for ch in range(3):
        h1 = cv2.calcHist([cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)],
                          [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)],
                          [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        score += cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return score / 3.0


def save_score_plot(scores_good, scores_fault, threshold,
                    title, filename, higher_is_fault=False):
    if not scores_good and not scores_fault:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.scatter(range(len(scores_good)), scores_good,
                color="green", label="Good frames", s=40, alpha=0.7)
    ax1.scatter(range(len(scores_good), len(scores_good) + len(scores_fault)),
                scores_fault,
                color="red",   label="Fault frames", s=40, alpha=0.7, marker="x")
    ax1.axhline(y=threshold, color="orange", linestyle="--",
                linewidth=2, label=f"Threshold={threshold}")
    ax1.set_title(f"{title} — Score per frame")
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if scores_good:
        ax2.hist(scores_good,  bins=20, color="green", alpha=0.6, label="Good")
    if scores_fault:
        ax2.hist(scores_fault, bins=20, color="red",   alpha=0.6, label="Fault")
    ax2.axvline(x=threshold, color="orange", linestyle="--",
                linewidth=2, label=f"Threshold={threshold}")
    ax2.set_title(f"{title} — Distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=130)
    plt.close()
    print(f"  Plot saved → {OUTPUT_DIR}/{filename}")


def severity_curve_plot(good_imgs, fault_imgs_by_severity,
                        compute_score_fn, threshold,
                        higher_is_fault, title, filename):
    """
    Plot detection rate vs fault severity level.
    fault_imgs_by_severity: list of (severity_label, [(path, frame), ...])
    """
    if not fault_imgs_by_severity:
        return

    labels       = []
    detect_rates = []

    for sev_label, imgs in fault_imgs_by_severity:
        if not imgs:
            continue
        detected = 0
        for _, frame in imgs:
            score = compute_score_fn(frame)
            if higher_is_fault:
                detected += 1 if score > threshold else 0
            else:
                detected += 1 if score < threshold else 0
        labels.append(sev_label)
        detect_rates.append(detected / len(imgs) * 100)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(labels, detect_rates, marker="o", color="red",
            linewidth=2, markersize=8, label="Detection rate")
    ax.axhline(y=95, color="green", linestyle="--",
               linewidth=1.5, label="95% target")
    ax.set_title(f"{title} — Detection Rate vs Severity")
    ax.set_xlabel("Severity level")
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=130)
    plt.close()
    print(f"  Severity curve saved → {OUTPUT_DIR}/{filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Load all images once
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
good_imgs     = load_images(os.path.join(DATASET_DIR, "good"))
blur_imgs     = load_images(os.path.join(DATASET_DIR, "blur"))
noise_imgs    = load_images(os.path.join(DATASET_DIR, "noise"))
overexp_imgs  = load_images(os.path.join(DATASET_DIR, "overexposed"))
underexp_imgs = load_images(os.path.join(DATASET_DIR, "underexposed"))
pov_imgs      = load_images(os.path.join(DATASET_DIR, "pov_change"))

# Separate night images from good for special reporting
night_imgs = [(p, f) for p, f in good_imgs if "night" in Path(p).stem.lower()]
day_imgs   = [(p, f) for p, f in good_imgs if "night" not in Path(p).stem.lower()]

print(f"  good/         : {len(good_imgs)} total "
      f"({len(day_imgs)} day + {len(night_imgs)} night variants)")
print(f"  blur/         : {len(blur_imgs)}")
print(f"  noise/        : {len(noise_imgs)}")
print(f"  overexposed/  : {len(overexp_imgs)}")
print(f"  underexposed/ : {len(underexp_imgs)}")
print(f"  pov_change/   : {len(pov_imgs)}")

if not good_imgs:
    print(f"\nERROR: No good images in {DATASET_DIR}/good — check DATASET_DIR")
    exit(1)

# ── Pre-compute Laplacian scores for good images (reused in Phase 1 and 2) ───
good_lap = []
for _, frame in good_imgs:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    good_lap.append(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Blur Detection
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(1, "Blur Detection")
print(f"  Metric  : Laplacian variance")
print(f"  Rule    : lap_var < {THRESHOLDS['blur_laplacian_threshold']} → BLUR")

thresh     = THRESHOLDS["blur_laplacian_threshold"]
fault_lap  = []
good_flags = sum(1 for v in good_lap if v < thresh)
fault_flags= 0

for _, frame in blur_imgs:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v    = cv2.Laplacian(gray, cv2.CV_64F).var()
    fault_lap.append(v)
    if v < thresh:
        fault_flags += 1

print(f"\n  Good  Laplacian — min={min(good_lap):.1f}  "
      f"max={max(good_lap):.1f}  mean={np.mean(good_lap):.1f}")
if fault_lap:
    print(f"  Blur  Laplacian — min={min(fault_lap):.1f}  "
          f"max={max(fault_lap):.1f}  mean={np.mean(fault_lap):.1f}")

print()
if blur_imgs:
    print_result("Blur detected", fault_flags, len(blur_imgs), True)
print_result("Good frames clean", len(good_imgs) - good_flags, len(good_imgs), False)

# Night-specific check
if night_imgs:
    night_lap   = []
    night_flags = 0
    for _, frame in night_imgs:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v    = cv2.Laplacian(gray, cv2.CV_64F).var()
        night_lap.append(v)
        if v < thresh:
            night_flags += 1
    icon = PASS if night_flags == 0 else FAIL
    print(f"  {icon} Night images not flagged as blur: "
          f"{len(night_imgs)-night_flags}/{len(night_imgs)}")

m1 = compute_metrics(fault_lap, good_lap, thresh, False)
print(f"\n  Accuracy={m1['accuracy']*100:.1f}%  "
      f"Precision={m1['precision']*100:.1f}%  "
      f"Recall={m1['recall']*100:.1f}%")
print(f"  Suggested threshold: {m1['suggested_threshold']}")

if blur_imgs:
    save_score_plot(good_lap, fault_lap, thresh,
                    "Blur — Laplacian Variance", "phase1_blur.png", False)

    # Severity curve — group by kernel size from filename
    kernel_groups = {}
    for p, f in blur_imgs:
        stem = Path(p).stem
        if "_k" in stem:
            k = stem.split("_k")[1].split("_")[0]
            kernel_groups.setdefault(f"k={k}", []).append((p, f))
    if kernel_groups:
        severity_curve_plot(
            good_imgs,
            sorted(kernel_groups.items(), key=lambda x: int(x[0].replace("k=",""))),
            lambda frame: cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        cv2.CV_64F).var(),
            thresh, False,
            "Blur", "phase1_blur_severity.png"
        )

phase_results["blur"] = {"threshold": thresh, "metrics": m1,
                          "fault_scores": fault_lap, "good_scores": good_lap}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Noise Detection
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(2, "Noise / Salt & Pepper Detection")
print(f"  Metric  : Laplacian variance")
print(f"  Rule    : lap_var > {THRESHOLDS['noise_laplacian_threshold']} → NOISE")

thresh      = THRESHOLDS["noise_laplacian_threshold"]
fault_lap_n = []
good_flags  = sum(1 for v in good_lap if v > thresh)
fault_flags = 0

for _, frame in noise_imgs:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v    = cv2.Laplacian(gray, cv2.CV_64F).var()
    fault_lap_n.append(v)
    if v > thresh:
        fault_flags += 1

print(f"\n  Good  Laplacian — min={min(good_lap):.1f}  "
      f"max={max(good_lap):.1f}  mean={np.mean(good_lap):.1f}")
if fault_lap_n:
    print(f"  Noise Laplacian — min={min(fault_lap_n):.1f}  "
          f"max={max(fault_lap_n):.1f}  mean={np.mean(fault_lap_n):.1f}")

print()
if noise_imgs:
    print_result("Noise detected", fault_flags, len(noise_imgs), True)
print_result("Good frames clean", len(good_imgs) - good_flags, len(good_imgs), False)

m2 = compute_metrics(fault_lap_n, good_lap, thresh, True)
print(f"\n  Accuracy={m2['accuracy']*100:.1f}%  "
      f"Precision={m2['precision']*100:.1f}%  "
      f"Recall={m2['recall']*100:.1f}%")
print(f"  Suggested threshold: {m2['suggested_threshold']}")

if noise_imgs:
    save_score_plot(good_lap, fault_lap_n, thresh,
                    "Noise — Laplacian Variance", "phase2_noise.png", True)

    # Severity curve — group by density from filename
    density_groups = {}
    for p, f in noise_imgs:
        stem = Path(p).stem
        if "_d" in stem:
            d = stem.split("_d")[1].split("_")[0]
            density_groups.setdefault(f"d={d}", []).append((p, f))
    if density_groups:
        severity_curve_plot(
            good_imgs,
            sorted(density_groups.items(), key=lambda x: float(x[0].replace("d=",""))/100),
            lambda frame: cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        cv2.CV_64F).var(),
            thresh, True,
            "Noise", "phase2_noise_severity.png"
        )

phase_results["noise"] = {"threshold": thresh, "metrics": m2,
                           "fault_scores": fault_lap_n, "good_scores": good_lap}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Overexposure Detection
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(3, "Overexposure Detection")
print(f"  Metric  : fraction of pixels > {THRESHOLDS['over_exposure_threshold']}")
print(f"  Rule    : ratio > {THRESHOLDS['exposure_pixel_ratio']} → OVEREXPOSED")

over_thresh  = THRESHOLDS["over_exposure_threshold"]
ratio_thresh = THRESHOLDS["exposure_pixel_ratio"]

good_over   = []
fault_over  = []
good_flags  = 0
fault_flags = 0

for _, frame in good_imgs:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio = np.sum(gray > over_thresh) / gray.size
    good_over.append(ratio)
    if ratio > ratio_thresh:
        good_flags += 1

for _, frame in overexp_imgs:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio = np.sum(gray > over_thresh) / gray.size
    fault_over.append(ratio)
    if ratio > ratio_thresh:
        fault_flags += 1

print(f"\n  Good       over-ratio — min={min(good_over):.3f}  "
      f"max={max(good_over):.3f}  mean={np.mean(good_over):.3f}")
if fault_over:
    print(f"  Overexposed over-ratio — min={min(fault_over):.3f}  "
          f"max={max(fault_over):.3f}  mean={np.mean(fault_over):.3f}")

print()
if overexp_imgs:
    print_result("Overexposed detected", fault_flags, len(overexp_imgs), True)
print_result("Good frames clean", len(good_imgs) - good_flags, len(good_imgs), False)

# Night check
if night_imgs:
    night_over  = []
    night_flags = 0
    for _, frame in night_imgs:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = np.sum(gray > over_thresh) / gray.size
        night_over.append(ratio)
        if ratio > ratio_thresh:
            night_flags += 1
    icon = PASS if night_flags == 0 else FAIL
    print(f"  {icon} Night images not flagged as overexposed: "
          f"{len(night_imgs)-night_flags}/{len(night_imgs)}")

m3 = compute_metrics(fault_over, good_over, ratio_thresh, True)
print(f"\n  Accuracy={m3['accuracy']*100:.1f}%  "
      f"Precision={m3['precision']*100:.1f}%  "
      f"Recall={m3['recall']*100:.1f}%")
print(f"  Suggested threshold: {m3['suggested_threshold']}")

if overexp_imgs:
    save_score_plot(good_over, fault_over, ratio_thresh,
                    "Overexposure — Saturated pixel ratio",
                    "phase3_overexposed.png", True)

    # Severity curve — group by factor from filename
    factor_groups = {}
    for p, f in overexp_imgs:
        stem = Path(p).stem
        if "_f" in stem:
            fv = stem.split("_f")[1].split("_")[0]
            factor_groups.setdefault(f"f={fv}", []).append((p, f))
    if factor_groups:
        def over_score(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.sum(gray > over_thresh) / gray.size
        severity_curve_plot(
            good_imgs,
            sorted(factor_groups.items(),
                   key=lambda x: float(x[0].replace("f=",""))),
            over_score, ratio_thresh, True,
            "Overexposure", "phase3_overexposed_severity.png"
        )

phase_results["overexposed"] = {"threshold": ratio_thresh, "metrics": m3,
                                 "fault_scores": fault_over, "good_scores": good_over}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Underexposure Detection
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(4, "Underexposure Detection")
print(f"  Metric  : fraction of pixels < {THRESHOLDS['under_exposure_threshold']}")
print(f"  Rule    : ratio > {THRESHOLDS['exposure_pixel_ratio']} → UNDEREXPOSED")

under_thresh = THRESHOLDS["under_exposure_threshold"]

good_under   = []
fault_under  = []
good_flags   = 0
fault_flags  = 0

for _, frame in good_imgs:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio = np.sum(gray < under_thresh) / gray.size
    good_under.append(ratio)
    if ratio > ratio_thresh:
        good_flags += 1

for _, frame in underexp_imgs:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio = np.sum(gray < under_thresh) / gray.size
    fault_under.append(ratio)
    if ratio > ratio_thresh:
        fault_flags += 1

print(f"\n  Good        under-ratio — min={min(good_under):.3f}  "
      f"max={max(good_under):.3f}  mean={np.mean(good_under):.3f}")
if fault_under:
    print(f"  Underexposed under-ratio — min={min(fault_under):.3f}  "
          f"max={max(fault_under):.3f}  mean={np.mean(fault_under):.3f}")

print()
if underexp_imgs:
    print_result("Underexposed detected", fault_flags, len(underexp_imgs), True)
print_result("Good frames clean", len(good_imgs) - good_flags, len(good_imgs), False)

# ── CRITICAL: Night image false positive check ────────────────────────────────
# This is the most important check in the entire test suite.
# Night images must NOT be flagged as underexposed.
if night_imgs:
    night_under = []
    night_flags = 0
    for _, frame in night_imgs:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = np.sum(gray < under_thresh) / gray.size
        night_under.append(ratio)
        if ratio > ratio_thresh:
            night_flags += 1
    icon = PASS if night_flags == 0 else FAIL
    print(f"\n  {icon} CRITICAL — Night images not flagged as underexposed: "
          f"{len(night_imgs)-night_flags}/{len(night_imgs)}")
    if night_flags > 0:
        print(f"  ⚠  {night_flags} night images are being falsely flagged.")
        print(f"  Fix: lower under_exposure_threshold in THRESHOLDS above.")
        print(f"  Night under-ratio — min={min(night_under):.4f}  "
              f"max={max(night_under):.4f}  mean={np.mean(night_under):.4f}")

m4 = compute_metrics(fault_under, good_under, ratio_thresh, True)
print(f"\n  Accuracy={m4['accuracy']*100:.1f}%  "
      f"Precision={m4['precision']*100:.1f}%  "
      f"Recall={m4['recall']*100:.1f}%")
print(f"  Suggested threshold: {m4['suggested_threshold']}")

if underexp_imgs:
    save_score_plot(good_under, fault_under, ratio_thresh,
                    "Underexposure — Dark pixel ratio",
                    "phase4_underexposed.png", True)

    factor_groups = {}
    for p, f in underexp_imgs:
        stem = Path(p).stem
        if "_f" in stem:
            fv = stem.split("_f")[1].split("_")[0]
            factor_groups.setdefault(f"f={fv}", []).append((p, f))
    if factor_groups:
        def under_score(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.sum(gray < under_thresh) / gray.size
        severity_curve_plot(
            good_imgs,
            sorted(factor_groups.items(),
                   key=lambda x: float(x[0].replace("f=",""))),
            under_score, ratio_thresh, True,
            "Underexposure", "phase4_underexposed_severity.png"
        )

phase_results["underexposed"] = {"threshold": ratio_thresh, "metrics": m4,
                                  "fault_scores": fault_under, "good_scores": good_under}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — POV Change Detection
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(5, "POV Change Detection")
print(f"  Metric  : HSV histogram correlation vs reference frame")
print(f"  Rule    : correlation < {THRESHOLDS['pov_correlation_threshold']} → POV_CHANGE")
print(f"  Note    : POV detection is camera-node specific.")
print(f"            Reference = first good image in dataset.")

thresh = THRESHOLDS["pov_correlation_threshold"]

if not good_imgs:
    print("  Skipped — no good images")
else:
    ref_frame       = good_imgs[0][1]
    good_pov_scores = []
    fault_pov_scores= []
    good_flags      = 0
    fault_flags     = 0

    for _, frame in good_imgs:
        score = hsv_correlation(frame, ref_frame)
        good_pov_scores.append(score)
        if score < thresh:
            good_flags += 1

    for _, frame in pov_imgs:
        score = hsv_correlation(frame, ref_frame)
        fault_pov_scores.append(score)
        if score < thresh:
            fault_flags += 1

    print(f"\n  Good     HSV — min={min(good_pov_scores):.3f}  "
          f"max={max(good_pov_scores):.3f}  mean={np.mean(good_pov_scores):.3f}")
    if fault_pov_scores:
        print(f"  POV chg  HSV — min={min(fault_pov_scores):.3f}  "
              f"max={max(fault_pov_scores):.3f}  mean={np.mean(fault_pov_scores):.3f}")

    print()
    if pov_imgs:
        print_result("POV change detected", fault_flags, len(pov_imgs), True)
    else:
        print(f"  (no pov_change images — skipping detection test)")
    print_result("Good frames clean", len(good_imgs) - good_flags,
                 len(good_imgs), False)

    m5 = compute_metrics(fault_pov_scores, good_pov_scores, thresh, False)
    print(f"\n  Accuracy={m5['accuracy']*100:.1f}%  "
          f"Precision={m5['precision']*100:.1f}%  "
          f"Recall={m5['recall']*100:.1f}%")
    print(f"  Suggested threshold: {m5['suggested_threshold']}")

    if fault_pov_scores:
        save_score_plot(good_pov_scores, fault_pov_scores, thresh,
                        "POV Change — HSV Correlation",
                        "phase5_pov.png", False)

    phase_results["pov_change"] = {
        "threshold": thresh, "metrics": m5,
        "fault_scores": fault_pov_scores, "good_scores": good_pov_scores
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Combined Test
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(6, "Combined Detector Test (all 5 detectors)")

ref_frame_combined = good_imgs[0][1] if good_imgs else None

def run_all_detectors(frame, ref):
    faults = []
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap < THRESHOLDS["blur_laplacian_threshold"]:
        faults.append("BLUR")
    elif lap > THRESHOLDS["noise_laplacian_threshold"]:
        faults.append("NOISE")

    over  = np.sum(gray > THRESHOLDS["over_exposure_threshold"])  / gray.size
    under = np.sum(gray < THRESHOLDS["under_exposure_threshold"]) / gray.size
    if over > THRESHOLDS["exposure_pixel_ratio"]:
        faults.append("OVEREXPOSED")
    elif under > THRESHOLDS["exposure_pixel_ratio"]:
        faults.append("UNDEREXPOSED")

    if ref is not None:
        score = hsv_correlation(frame, ref)
        if score < THRESHOLDS["pov_correlation_threshold"]:
            faults.append("POV_CHANGE")

    return faults

all_fault_imgs = (
    [("blur",        p, f) for p, f in blur_imgs]     +
    [("noise",       p, f) for p, f in noise_imgs]    +
    [("overexposed", p, f) for p, f in overexp_imgs]  +
    [("underexposed",p, f) for p, f in underexp_imgs] +
    [("pov_change",  p, f) for p, f in pov_imgs]
)

good_fp        = 0
night_fp       = 0
fault_detected = 0
fault_missed   = []

for _, frame in good_imgs:
    if run_all_detectors(frame, ref_frame_combined):
        good_fp += 1

for _, frame in night_imgs:
    if run_all_detectors(frame, ref_frame_combined):
        night_fp += 1

for category, path, frame in all_fault_imgs:
    if run_all_detectors(frame, ref_frame_combined):
        fault_detected += 1
    else:
        fault_missed.append((category, Path(path).name))

total_fault  = len(all_fault_imgs)
overall_acc  = (fault_detected + len(good_imgs) - good_fp) / \
               (total_fault + len(good_imgs)) * 100 if (total_fault + len(good_imgs)) > 0 else 0

print(f"\n  Total good images   : {len(good_imgs)} "
      f"({len(night_imgs)} night variants)")
print(f"  Total fault images  : {total_fault}")
print()
print_result("Fault frames detected", fault_detected, total_fault, True)
print_result("Good frames clean",     len(good_imgs) - good_fp, len(good_imgs), False)

icon = PASS if night_fp == 0 else FAIL
print(f"  {icon} Night images clean (CRITICAL): "
      f"{len(night_imgs)-night_fp}/{len(night_imgs)}")

print(f"\n  Overall accuracy    : {overall_acc:.1f}%")

if fault_missed:
    print(f"\n  Missed fault frames ({len(fault_missed)}):")
    missed_by_cat = {}
    for cat, name in fault_missed:
        missed_by_cat.setdefault(cat, []).append(name)
    for cat, names in missed_by_cat.items():
        print(f"    [{cat}] {len(names)} missed")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Threshold Tuning Report
# ─────────────────────────────────────────────────────────────────────────────
print_phase_header(7, "Threshold Tuning Report & Config Recommendation")

suggested_config = {
    "enabled":                     True,
    "consecutive_fault_threshold": 3,
    "blur_laplacian_threshold":    phase_results.get("blur",        {}).get("metrics", {}).get("suggested_threshold", THRESHOLDS["blur_laplacian_threshold"]),
    "noise_laplacian_threshold":   phase_results.get("noise",       {}).get("metrics", {}).get("suggested_threshold", THRESHOLDS["noise_laplacian_threshold"]),
    "exposure_pixel_ratio":        round((
        phase_results.get("overexposed",  {}).get("metrics", {}).get("suggested_threshold", THRESHOLDS["exposure_pixel_ratio"]) +
        phase_results.get("underexposed", {}).get("metrics", {}).get("suggested_threshold", THRESHOLDS["exposure_pixel_ratio"])
    ) / 2, 3),
    "over_exposure_threshold":     THRESHOLDS["over_exposure_threshold"],
    "under_exposure_threshold":    THRESHOLDS["under_exposure_threshold"],
    "pov_correlation_threshold":   phase_results.get("pov_change",  {}).get("metrics", {}).get("suggested_threshold", THRESHOLDS["pov_correlation_threshold"]),
}

print(f"\n  {'Parameter':<35} {'Current':>10} {'Suggested':>12}")
print(f"  {'-'*60}")
param_map = {
    "blur_laplacian_threshold":   "blur_laplacian_threshold",
    "noise_laplacian_threshold":  "noise_laplacian_threshold",
    "exposure_pixel_ratio":       "exposure_pixel_ratio",
    "pov_correlation_threshold":  "pov_correlation_threshold",
}
for param, key in param_map.items():
    current   = THRESHOLDS.get(key, "N/A")
    suggested = suggested_config.get(param, "N/A")
    changed   = " ←" if current != suggested else ""
    print(f"  {param:<35} {str(current):>10} {str(suggested):>12}{changed}")

print(f"\n  Paste this into config_setup.json Camera_SelfCheck block:")
print(f"\n  \"Camera_SelfCheck\": {json.dumps(suggested_config, indent=4)}")

config_path = f"{OUTPUT_DIR}/suggested_camera_selfcheck_config.json"
with open(config_path, "w") as f:
    json.dump({"Camera_SelfCheck": suggested_config}, f, indent=4)
print(f"\n  Saved → {config_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY DASHBOARD")
print(f"{'='*60}")
print(f"  {'Phase':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
print(f"  {'-'*57}")
for phase_name, data in phase_results.items():
    m = data["metrics"]
    print(f"  {phase_name:<25} "
          f"{m['accuracy']*100:>9.1f}% "
          f"{m['precision']*100:>9.1f}% "
          f"{m['recall']*100:>9.1f}%")
print(f"  {'-'*57}")
print(f"  {'Combined (Phase 6)':<25} {overall_acc:>9.1f}%")
print(f"\n  Night FP rate (CRITICAL) : "
      f"{night_fp}/{len(night_imgs)} "
      f"({'✓ PASS' if night_fp == 0 else '✗ FAIL — fix under_exposure_threshold'})")
print(f"\n  All plots → {OUTPUT_DIR}/")
print(f"{'='*60}\n")