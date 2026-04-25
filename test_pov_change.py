
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE_DIR    = "F:/Wildfire_Camera/Dataset_images/Mixed_images/POV_Sequence"
POV_INJECT_DIR  = "F:/Wildfire_Camera/Dataset_images/Mixed_images/POV_Change"
OUTPUT_DIR      = "pov_sequence_test_results"
TARGET_SIZE     = (1280, 960)

# Where to inject foreign frames in the sequence
# e.g. [10, 11, 12] injects 3 consecutive frames at positions 10, 11, 12
# These 3 consecutive faults should trigger SHUT_CAMERA
INJECT_POSITIONS = [10, 11, 12]

# How many good frames to process before and after injection
FRAMES_BEFORE_INJECT = 15   # process this many good frames first
FRAMES_AFTER_INJECT  = 5    # process this many after injection (should all be SHUT_CAMERA)

# POV threshold
POV_THRESHOLD = 0.4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
tests_passed = 0
tests_failed = 0


def assert_test(condition, name, detail=""):
    global tests_passed, tests_failed
    if condition:
        print(f"  {PASS} {name}")
        tests_passed += 1
    else:
        print(f"  {FAIL} {name} — {detail}")
        tests_failed += 1


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


def classify(frame, reference, consecutive_faults, is_shutdown,
             consecutive_limit=3):
    """
    Run POV detection on a single frame.
    Returns (signal, score, faults, new_consecutive, new_is_shutdown)
    """
    if is_shutdown:
        return "SHUT_CAMERA", None, ["CAMERA_OFFLINE"], consecutive_faults, True

    frame_r = cv2.resize(frame, TARGET_SIZE)
    score   = hsv_correlation(frame_r, reference)

    if score < POV_THRESHOLD:
        faults = [f"POV_CHANGE(score={score:.3f})"]
        consecutive_faults += 1
        if consecutive_faults >= consecutive_limit:
            signal      = "SHUT_CAMERA"
            is_shutdown = True
        elif consecutive_faults >= 2:
            signal = "FAULT_CRITICAL"
        else:
            signal = "FAULT_WARNING"
    else:
        faults             = []
        consecutive_faults = 0
        signal             = "CAMERA_OK"

    return signal, score, faults, consecutive_faults, is_shutdown


def load_images_ordered(folder):
    """Load images in filename order."""
    if not os.path.exists(folder):
        return []
    paths = sorted(
        list(Path(folder).glob("*.jpg")) +
        list(Path(folder).glob("*.png"))
    )
    result = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            result.append((p.name, frame))
    return result


# ── Load images ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SINGLE-CAMERA SEQUENCE POV TEST")
print(f"{'='*60}")

sequence_imgs = load_images_ordered(SEQUENCE_DIR)
inject_imgs   = load_images_ordered(POV_INJECT_DIR)

print(f"  Sequence images : {len(sequence_imgs)}")
print(f"  Inject images   : {len(inject_imgs)}")
print(f"  Inject positions: {INJECT_POSITIONS}")
print(f"  POV threshold   : {POV_THRESHOLD}")

if not sequence_imgs:
    print(f"\nERROR: No sequence images in {SEQUENCE_DIR}")
    print("Add ordered images from ONE camera node (name as 001.jpg, 002.jpg...)")
    exit(1)

if not inject_imgs:
    print(f"\nERROR: No inject images in {POV_INJECT_DIR}")
    print("Add foreign images (indoor shots, different scenes)")
    exit(1)

needed = FRAMES_BEFORE_INJECT + len(INJECT_POSITIONS) + FRAMES_AFTER_INJECT
if len(sequence_imgs) < needed:
    print(f"\nWARNING: Only {len(sequence_imgs)} sequence images but need {needed}.")
    print(f"Adjusting FRAMES_BEFORE_INJECT to fit.")
    FRAMES_BEFORE_INJECT = max(3, len(sequence_imgs) - len(INJECT_POSITIONS) - FRAMES_AFTER_INJECT)


# ── Set reference from first frame ────────────────────────────────────────────
ref_name, ref_raw = sequence_imgs[0]
reference         = cv2.resize(ref_raw, TARGET_SIZE)
self_score        = hsv_correlation(reference, reference)

print(f"\n  Reference frame : {ref_name}")
print(f"  Self-score (should be 1.0): {self_score:.4f}\n")


# ── Build the test sequence ───────────────────────────────────────────────────
# Interleave good frames with injected foreign frames at INJECT_POSITIONS
test_sequence = []

good_pool  = sequence_imgs[1:]    # skip reference frame
inject_pool= inject_imgs

good_idx   = 0
inject_idx = 0
total_positions = FRAMES_BEFORE_INJECT + len(INJECT_POSITIONS) + FRAMES_AFTER_INJECT

for pos in range(total_positions):
    if pos in INJECT_POSITIONS:
        # Inject a foreign frame
        inj_name, inj_frame = inject_pool[inject_idx % len(inject_pool)]
        test_sequence.append({
            "position": pos,
            "name":     inj_name,
            "frame":    inj_frame,
            "is_injected": True,
            "expected": "pov_fault",
        })
        inject_idx += 1
    else:
        # Use next good frame from sequence
        if good_idx < len(good_pool):
            g_name, g_frame = good_pool[good_idx]
            good_idx += 1
        else:
            # Wrap around if sequence is short
            g_name, g_frame = good_pool[good_idx % len(good_pool)]
            good_idx += 1
        test_sequence.append({
            "position":    pos,
            "name":        g_name,
            "frame":       g_frame,
            "is_injected": False,
            "expected":    "good",
        })

print(f"  Test sequence built: {len(test_sequence)} frames")
print(f"    Good frames    : {sum(1 for f in test_sequence if not f['is_injected'])}")
print(f"    Injected frames: {sum(1 for f in test_sequence if f['is_injected'])}\n")


# ── Run the sequence ──────────────────────────────────────────────────────────
print("── Processing sequence ──────────────────────────────────────────────")

consecutive_faults = 0
is_shutdown        = False
log                = []
scores             = []
signals            = []

for item in test_sequence:
    signal, score, faults, consecutive_faults, is_shutdown = classify(
        item["frame"], reference, consecutive_faults, is_shutdown
    )

    item["signal"]   = signal
    item["score"]    = score
    item["faults"]   = faults
    item["cons"]     = consecutive_faults

    log.append(item)
    scores.append(score)
    signals.append(signal)

    injected_marker = " ← INJECTED" if item["is_injected"] else ""
    score_str       = f"score={score:.3f}" if score is not None else "score=N/A"
    print(f"  Frame {item['position']:3d}  {signal:<16} {score_str}  "
          f"cons={consecutive_faults}  {item['name']}{injected_marker}")


# ── Validate results ──────────────────────────────────────────────────────────
print(f"\n── Validating results ───────────────────────────────────────────────")

good_frames      = [f for f in log if not f["is_injected"]]
injected_frames  = [f for f in log if f["is_injected"]]

# Good frames before injection should all be CAMERA_OK
good_before_inject = [
    f for f in good_frames
    if f["position"] < min(INJECT_POSITIONS)
]
ok_before = sum(1 for f in good_before_inject if f["signal"] == "CAMERA_OK")
assert_test(
    ok_before == len(good_before_inject),
    f"Test 01 — Good frames before injection all CAMERA_OK "
    f"({ok_before}/{len(good_before_inject)})",
    f"Some good frames were flagged"
)

# First injected frame should be FAULT_WARNING
first_injected = injected_frames[0] if injected_frames else None
if first_injected:
    assert_test(
        first_injected["signal"] in ["FAULT_WARNING", "FAULT_CRITICAL", "SHUT_CAMERA"],
        f"Test 02 — First injected frame triggers fault "
        f"({first_injected['signal']})",
        f"Expected fault, got CAMERA_OK"
    )

# After 3 consecutive injected frames, SHUT_CAMERA should trigger
if len(injected_frames) >= 3:
    third_injected = injected_frames[2]
    assert_test(
        third_injected["signal"] == "SHUT_CAMERA",
        f"Test 03 — SHUT_CAMERA triggers after 3 consecutive injected frames "
        f"({third_injected['signal']})",
        f"Expected SHUT_CAMERA"
    )

# Frames after injection should all be SHUT_CAMERA (camera stays locked)
good_after_inject = [
    f for f in good_frames
    if f["position"] > max(INJECT_POSITIONS)
]
if good_after_inject:
    shut_after = sum(1 for f in good_after_inject if f["signal"] == "SHUT_CAMERA")
    assert_test(
        shut_after == len(good_after_inject),
        f"Test 04 — All frames after SHUT_CAMERA stay locked "
        f"({shut_after}/{len(good_after_inject)})",
        f"Camera unlocked unexpectedly after shutdown"
    )

# All injected frames should have non-OK signal
injected_detected = sum(
    1 for f in injected_frames
    if f["signal"] != "CAMERA_OK"
)
assert_test(
    injected_detected == len(injected_frames),
    f"Test 05 — All injected frames flagged as fault "
    f"({injected_detected}/{len(injected_frames)})",
    f"{len(injected_frames)-injected_detected} injected frames missed"
)

# Score separation check
good_scores_list = [f["score"] for f in good_before_inject if f["score"] is not None]
inj_scores_list  = [f["score"] for f in injected_frames if f["score"] is not None]

if good_scores_list and inj_scores_list:
    gap = min(good_scores_list) - max(inj_scores_list)
    print(f"\n  Score separation:")
    print(f"    Good frames HSV  — min={min(good_scores_list):.3f}  "
          f"max={max(good_scores_list):.3f}  mean={np.mean(good_scores_list):.3f}")
    print(f"    Injected HSV     — min={min(inj_scores_list):.3f}  "
          f"max={max(inj_scores_list):.3f}  mean={np.mean(inj_scores_list):.3f}")
    print(f"    Gap (good_min - injected_max): {gap:.3f}")
    if gap > 0.1:
        print(f"    {PASS} Strong separation — threshold is well placed")
    elif gap > 0:
        print(f"    ! Marginal separation — consider tightening POV threshold")
    else:
        print(f"    {FAIL} Scores overlap — injected images too similar to scene")
        print(f"    Use more visually different images in pov_inject/")


# ── Plots ─────────────────────────────────────────────────────────────────────
print(f"\n── Saving results ───────────────────────────────────────────────────")

# Score timeline plot
fig, ax = plt.subplots(figsize=(14, 5))

all_positions = [f["position"] for f in log]
all_scores    = [f["score"] if f["score"] is not None else 0 for f in log]
all_injected  = [f["is_injected"] for f in log]
all_signals   = [f["signal"] for f in log]

colors = [
    "red"        if inj else
    "darkred"    if sig == "SHUT_CAMERA" else
    "darkorange" if sig == "FAULT_CRITICAL" else
    "orange"     if sig == "FAULT_WARNING" else
    "green"
    for inj, sig in zip(all_injected, all_signals)
]

ax.scatter(all_positions, all_scores, c=colors, s=100, zorder=3)
ax.plot(all_positions, all_scores, color="grey", alpha=0.4, linewidth=1)
ax.axhline(y=POV_THRESHOLD, color="orange", linestyle="--",
           linewidth=2, label=f"POV threshold={POV_THRESHOLD}")

# Mark injection zone
if INJECT_POSITIONS:
    ax.axvspan(min(INJECT_POSITIONS) - 0.5, max(INJECT_POSITIONS) + 0.5,
               alpha=0.15, color="red", label="Injection zone")

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green",      label="CAMERA_OK"),
    Patch(facecolor="orange",     label="FAULT_WARNING"),
    Patch(facecolor="darkorange", label="FAULT_CRITICAL"),
    Patch(facecolor="darkred",    label="SHUT_CAMERA"),
    Patch(facecolor="red",        label="Injected frame"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower left")
ax.set_title("POV Sequence Test — HSV Correlation Score Timeline\n"
             "Score drops below threshold at injection → escalates to SHUT_CAMERA")
ax.set_xlabel("Frame position in sequence")
ax.set_ylabel("HSV correlation score")
ax.set_ylim(-0.2, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
timeline_path = f"{OUTPUT_DIR}/sequence_timeline.png"
plt.savefig(timeline_path, dpi=150)
plt.close()
print(f"  {PASS} Timeline plot saved → {timeline_path}")

# Visual grid — reference + good samples + injected samples
fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(2, 6, figure=fig)

# Reference
ref_rgb = cv2.cvtColor(cv2.resize(ref_raw, (320, 240)), cv2.COLOR_BGR2RGB)
ax_ref  = fig.add_subplot(gs[:, 0])
ax_ref.imshow(ref_rgb)
ax_ref.set_title("REFERENCE\n(POV baseline)", fontsize=8, fontweight="bold")
ax_ref.axis("off")

# Good frames (row 0)
good_samples = good_before_inject[:5]
for col, item in enumerate(good_samples):
    rgb   = cv2.cvtColor(cv2.resize(item["frame"], (320, 240)), cv2.COLOR_BGR2RGB)
    score = item["score"] or 0
    ax    = fig.add_subplot(gs[0, col + 1])
    ax.imshow(rgb)
    ax.set_title(f"GOOD\n{item['signal']}\nscore={score:.3f}",
                 fontsize=7, color="green")
    ax.axis("off")

# Injected frames (row 1)
for col, item in enumerate(injected_frames[:5]):
    rgb   = cv2.cvtColor(cv2.resize(item["frame"], (320, 240)), cv2.COLOR_BGR2RGB)
    score = item["score"] or 0
    color = "red" if item["signal"] != "CAMERA_OK" else "orange"
    ax    = fig.add_subplot(gs[1, col + 1])
    ax.imshow(rgb)
    ax.set_title(f"INJECTED\n{item['signal']}\nscore={score:.3f}",
                 fontsize=7, color=color)
    ax.axis("off")

plt.suptitle("POV Sequence Test — Visual Comparison",
             fontsize=11, fontweight="bold")
plt.tight_layout()
grid_path = f"{OUTPUT_DIR}/visual_grid.png"
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  {PASS} Visual grid saved    → {grid_path}")

# Save JSON log
json_log = [
    {k: v for k, v in item.items() if k != "frame"}
    for item in log
]
with open(f"{OUTPUT_DIR}/sequence_log.json", "w") as f:
    json.dump(json_log, f, indent=2, default=str)
print(f"  {PASS} Log saved            → {OUTPUT_DIR}/sequence_log.json")


# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Results: {tests_passed} passed, {tests_failed} failed")
if tests_failed == 0:
    print(f"  \033[92m✓ POV sequence test passed.\033[0m")
else:
    print(f"  \033[91m✗ Some tests failed.\033[0m")
    if inj_scores_list and max(inj_scores_list) >= POV_THRESHOLD:
        print(f"\n  Fix: injected images score too high ({max(inj_scores_list):.3f})")
        print(f"  Use more visually different images in pov_inject/")
        print(f"  (indoor shots, city streets, close-up objects)")
print(f"{'='*60}\n")