"""
Real Image Dataset Test — CameraSelfCheck
==========================================
Tests CameraSelfCheck against your actual Mixed_images dataset
(good/ and fault/ folders) and produces:

  - Per-image fault detection results
  - Confusion matrix (predicted fault vs ground truth)
  - Detection rate per fault type
  - Signal distribution chart
  - Full CSV log of every frame

Run:
    python test_camera_real_images.py

Set MIXED_DATASET_DIR below to your actual path before running.
"""

import cv2
import numpy as np
import json
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from wildfire_env import CameraSelfCheck

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update this path to your dataset
# ─────────────────────────────────────────────────────────────────────────────
MIXED_DATASET_DIR  = "F:/Wildfire_Camera/Dataset_images/Mixed_images"
CONSECUTIVE_THRESH = 3      # must match config value
SHUFFLE_FRAMES     = True   # set False to process in fixed order
OUTPUT_DIR         = "camera_selfcheck_real_test_results"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

# ── Load config ───────────────────────────────────────────────────────────────
with open("config_setup_0_90036452TP_0_58399005FP_1dayReservedEnergy.json") as f:
    config = json.load(f)

check = CameraSelfCheck(config)

# ── Load image paths with ground truth labels ─────────────────────────────────
good_paths  = [(str(p), "good")  for p in Path(MIXED_DATASET_DIR + "/good").glob("*.jpg")]
fault_paths = [(str(p), "fault") for p in Path(MIXED_DATASET_DIR + "/fault").glob("*.jpg")]
all_frames  = good_paths + fault_paths

if SHUFFLE_FRAMES:
    random.seed(42)   # fixed seed for reproducibility
    random.shuffle(all_frames)

print(f"\n{'='*60}")
print(f"  CAMERA SELF-CHECK — REAL IMAGE DATASET TEST")
print(f"{'='*60}")
print(f"  Good images  : {len(good_paths)}")
print(f"  Fault images : {len(fault_paths)}")
print(f"  Total frames : {len(all_frames)}")
print(f"  Output dir   : {OUTPUT_DIR}")
print(f"{'='*60}\n")

if len(all_frames) == 0:
    print("ERROR: No images found. Check MIXED_DATASET_DIR path.")
    exit(1)

# ── Set reference frame from first good image ─────────────────────────────────
# Use the first good image as the POV reference so the POV detector has
# a clean baseline to compare against.
first_good = [p for p, gt in all_frames if gt == "good"]
if first_good:
    ref_frame = cv2.imread(first_good[0])
    if ref_frame is not None:
        check.set_reference(ref_frame)
        print(f"  Reference frame set from: {Path(first_good[0]).name}\n")

# ─────────────────────────────────────────────────────────────────────────────
print("── Processing frames ────────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

check.reset()

log           = []
y_true        = []   # 0 = good, 1 = fault (ground truth)
y_pred        = []   # 0 = CAMERA_OK, 1 = any fault signal
fault_type_counts = {}   # count per fault type string

for i, (img_path, ground_truth) in enumerate(all_frames):

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  Frame {i:04d} — skipped (unreadable): {img_path}")
        continue

    signal, faults, image_ok = check.check(frame, step=i)

    # Ground truth: "fault" → 1, "good" → 0
    gt_label   = 1 if ground_truth == "fault" else 0
    # Prediction: any non-OK signal → 1 (fault detected)
    pred_label = 0 if signal == "CAMERA_OK" else 1

    y_true.append(gt_label)
    y_pred.append(pred_label)

    # Count individual fault types
    for f in faults:
        # Extract just the fault name before the "(" detail
        fault_name = f.split("(")[0]
        fault_type_counts[fault_name] = fault_type_counts.get(fault_name, 0) + 1

    log.append({
        "frame_idx":          i,
        "image_path":         Path(img_path).name,
        "ground_truth":       ground_truth,
        "signal":             signal,
        "image_ok":           image_ok,
        "faults_detected":    "; ".join(faults) if faults else "none",
        "consecutive_faults": check.consecutive_faults,
        "correct":            gt_label == pred_label,
    })

    # Console output every 50 frames
    if i % 50 == 0 or signal != "CAMERA_OK":
        icon = "✓" if signal == "CAMERA_OK" else "⚡"
        print(f"  Frame {i:04d} {icon} {signal:<18} GT:{ground_truth:<8} "
              f"Faults: {'; '.join(faults) if faults else '—'}")

    # Stop processing after SHUT_CAMERA — camera is locked
    # (comment this out if you want to process ALL frames regardless)
    if check.is_shutdown:
        print(f"\n  [!] SHUT_CAMERA triggered at frame {i} — "
              f"camera locked for rest of episode.")
        print(f"  Resetting for continued evaluation...\n")
        # Reset so we can continue evaluating remaining frames
        check.reset()
        if first_good and ref_frame is not None:
            check.set_reference(ref_frame)


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Detection Performance ────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  Total frames processed : {len(log)}")
print(f"  Correct predictions    : {sum(1 for r in log if r['correct'])}")
print(f"  Accuracy               : {accuracy_score(y_true, y_pred)*100:.1f}%")

print(f"\n  Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["Good Frame", "Fault Frame"],
    zero_division=0
))

print(f"\n  Fault type breakdown:")
if fault_type_counts:
    for fault_name, count in sorted(fault_type_counts.items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"    {fault_name:<25} : {count}")
else:
    print("    No faults detected across any frame.")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Signal Distribution ──────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

log_df          = pd.DataFrame(log)
signal_counts   = log_df["signal"].value_counts()
print(f"\n  {signal_counts.to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Saving results ───────────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Full CSV log ───────────────────────────────────────────────────────────
csv_path = f"{OUTPUT_DIR}/frame_log.csv"
log_df.to_csv(csv_path, index=False)
print(f"  {PASS} Frame log saved      → {csv_path}")

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
    xticklabels=["Predicted OK", "Predicted FAULT"],
    yticklabels=["Actually OK",  "Actually FAULT"]
)
ax.set_title("Camera Self-Check — Confusion Matrix\n(Real Image Dataset)")
plt.tight_layout()
cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  {PASS} Confusion matrix saved → {cm_path}")

# ── 3. Signal distribution bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
signal_colors = {
    "CAMERA_OK":      "green",
    "FAULT_WARNING":  "orange",
    "FAULT_CRITICAL": "darkorange",
    "SHUT_CAMERA":    "red",
}
bars = signal_counts.plot(
    kind="bar", ax=ax,
    color=[signal_colors.get(s, "steelblue") for s in signal_counts.index]
)
ax.set_title("Signal Distribution — Real Image Dataset")
ax.set_ylabel("Frame count")
ax.set_xlabel("Signal")
ax.tick_params(axis="x", rotation=30)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
sd_path = f"{OUTPUT_DIR}/signal_distribution.png"
plt.savefig(sd_path, dpi=150)
plt.close()
print(f"  {PASS} Signal distribution saved → {sd_path}")

# ── 4. Fault type bar chart ───────────────────────────────────────────────────
if fault_type_counts:
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(fault_type_counts.keys())
    counts = list(fault_type_counts.values())
    ax.bar(names, counts, color="steelblue")
    ax.set_title("Fault Type Breakdown — Real Image Dataset")
    ax.set_ylabel("Count")
    ax.set_xlabel("Fault Type")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    ft_path = f"{OUTPUT_DIR}/fault_type_breakdown.png"
    plt.savefig(ft_path, dpi=150)
    plt.close()
    print(f"  {PASS} Fault breakdown saved   → {ft_path}")

# ── 5. Per-ground-truth accuracy breakdown ────────────────────────────────────
good_rows  = log_df[log_df["ground_truth"] == "good"]
fault_rows = log_df[log_df["ground_truth"] == "fault"]

good_correct  = good_rows["correct"].sum()
fault_correct = fault_rows["correct"].sum()

print(f"\n── Per-class accuracy ───────────────────────────────────────────────")
print(f"  Good frames  : {good_correct}/{len(good_rows)}  "
      f"correctly identified as OK "
      f"({100*good_correct/max(1,len(good_rows)):.1f}%)")
print(f"  Fault frames : {fault_correct}/{len(fault_rows)}  "
      f"correctly identified as FAULT "
      f"({100*fault_correct/max(1,len(fault_rows)):.1f}%)")

false_positives = good_rows[~good_rows["correct"]]
false_negatives = fault_rows[~fault_rows["correct"]]
print(f"\n  False positives (good flagged as fault) : {len(false_positives)}")
print(f"  False negatives (fault missed as OK)    : {len(false_negatives)}")

if len(false_negatives) > 0:
    print(f"\n  Missed fault frames (false negatives):")
    for _, row in false_negatives.iterrows():
        print(f"    {row['image_path']}")

# ── 6. Summary JSON ───────────────────────────────────────────────────────────
summary = {
    "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_frames":        len(log),
    "good_frames":         len(good_rows),
    "fault_frames":        len(fault_rows),
    "accuracy_pct":        round(accuracy_score(y_true, y_pred) * 100, 2),
    "good_detection_pct":  round(100*good_correct/max(1,len(good_rows)),  2),
    "fault_detection_pct": round(100*fault_correct/max(1,len(fault_rows)),2),
    "false_positives":     len(false_positives),
    "false_negatives":     len(false_negatives),
    "fault_type_counts":   fault_type_counts,
    "signal_counts":       signal_counts.to_dict(),
}
summary_path = f"{OUTPUT_DIR}/test_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"\n  {PASS} Summary saved          → {summary_path}")

print(f"\n{'='*60}")
print(f"  Overall accuracy : {accuracy_score(y_true, y_pred)*100:.1f}%")
print(f"  Results saved in : {OUTPUT_DIR}/")
print(f"{'='*60}\n")
