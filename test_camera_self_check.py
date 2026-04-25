"""
Stage 1 — CameraSelfCheck Unit Tests
=====================================
Tests every fault detector and the signal state machine in complete isolation.
No RL environment, no dataset, no trained model needed.

Run:
    python test_camera_self_check.py
"""

import cv2
import numpy as np
import json
import copy
from wildfire_env import CameraSelfCheck

# ── Load config ───────────────────────────────────────────────────────────────
with open("config_setup_0_90036452TP_0_58399005FP_1dayReservedEnergy.json") as f:
    config = json.load(f)

check = CameraSelfCheck(config)

PASS  = "\033[92m✓\033[0m"
FAIL  = "\033[91m✗\033[0m"
tests_passed = 0
tests_failed = 0


def assert_test(condition, test_name, detail=""):
    global tests_passed, tests_failed
    if condition:
        print(f"  {PASS} {test_name}")
        tests_passed += 1
    else:
        print(f"  {FAIL} {test_name} — {detail}")
        tests_failed += 1


def make_frame(brightness=120, noise_level=2, tile_size=20):
    """
    Build a clean synthetic BGR frame that reliably passes ALL fault detectors.

    Strategy: checkerboard pattern alternating between (brightness) and
    (brightness + 80), clamped to [0, 255].  The sharp block edges give a
    Laplacian variance well above 100 regardless of brightness value.
    noise_level=0 is fully safe — np.random.randint is never called.

    Why not a gradient?
        A pure horizontal gradient has the same grey value at every pixel in
        the same column, so after BGR→GREY conversion the Laplacian (second
        derivative) is near-zero everywhere except at the very edge columns —
        giving a variance of only ~13.  A checkerboard produces many edges
        and a variance of ~2000+.
    """
    h, w   = 480, 640
    hi_val = int(np.clip(brightness + 80, 0, 255))
    lo_val = int(np.clip(brightness,      0, 255))

    # Build a checkerboard in grayscale then expand to BGR
    row_idx  = (np.arange(h) // tile_size) % 2          # 0 or 1 per row-block
    col_idx  = (np.arange(w) // tile_size) % 2          # 0 or 1 per col-block
    checker  = (row_idx[:, None] ^ col_idx[None, :])     # XOR → 0/1 grid
    gray     = np.where(checker, hi_val, lo_val).astype(np.uint8)
    frame    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if noise_level > 0:
        noise = np.random.randint(
            -noise_level, noise_level, frame.shape, dtype=np.int16
        )
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


# ── Pre-flight: confirm make_frame() passes blur / exposure / frozen checks ───
_test_frame = make_frame(brightness=120, noise_level=0)
_gray       = cv2.cvtColor(_test_frame, cv2.COLOR_BGR2GRAY)
_lap_var    = cv2.Laplacian(_gray, cv2.CV_64F).var()
_mean_b     = np.mean(_gray)
_over_r     = np.sum(_gray > config["Camera_SelfCheck"]["over_exposure_threshold"])  / _gray.size
_under_r    = np.sum(_gray < config["Camera_SelfCheck"]["under_exposure_threshold"]) / _gray.size
_blur_thr   = config["Camera_SelfCheck"]["blur_laplacian_threshold"]
_noise_thr  = config["Camera_SelfCheck"]["noise_laplacian_threshold"]
_black_thr  = config["Camera_SelfCheck"]["black_brightness_threshold"]
_exp_ratio  = config["Camera_SelfCheck"]["exposure_pixel_ratio"]

assert _lap_var  >= _blur_thr,   f"PREFLIGHT FAIL: Laplacian={_lap_var:.1f} < blur_threshold={_blur_thr}"
assert _lap_var  <= _noise_thr,  f"PREFLIGHT FAIL: Laplacian={_lap_var:.1f} > noise_threshold={_noise_thr}"
assert _mean_b   >= _black_thr,  f"PREFLIGHT FAIL: brightness={_mean_b:.1f} < black_threshold={_black_thr}"
assert _over_r   <= _exp_ratio,  f"PREFLIGHT FAIL: over_ratio={_over_r:.2f} > exposure_ratio={_exp_ratio}"
assert _under_r  <= _exp_ratio,  f"PREFLIGHT FAIL: under_ratio={_under_r:.2f} > exposure_ratio={_exp_ratio}"

print(f"  Pre-flight OK — Laplacian={_lap_var:.1f}  "
      f"brightness={_mean_b:.1f}  "
      f"over={_over_r:.3f}  under={_under_r:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Signal State Machine ─────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

clean = make_frame(brightness=120)
black = np.zeros((480, 640, 3), dtype=np.uint8)

# Test 1: Clean frame → CAMERA_OK
check.reset()
signal, faults, image_ok = check.check(clean, step=0)
assert_test(signal == "CAMERA_OK",
            "Test 01 — Clean frame → CAMERA_OK",
            f"got signal={signal}, faults={faults}")
assert_test(image_ok is True,
            "Test 01b — image_ok=True on CAMERA_OK")
assert_test(faults == [],
            "Test 01c — No faults on clean frame",
            f"unexpected faults: {faults}")

# Test 2: 1 fault → FAULT_WARNING
check.reset()
signal, faults, image_ok = check.check(black, step=0)
assert_test(signal == "FAULT_WARNING",
            "Test 02 — 1 fault → FAULT_WARNING")
assert_test(image_ok is False,
            "Test 02b — image_ok=False on FAULT_WARNING")

# Test 3: 2 consecutive faults → FAULT_CRITICAL
check.reset()
for s in range(2):
    signal, faults, image_ok = check.check(black, step=s)
assert_test(signal == "FAULT_CRITICAL",
            "Test 03 — 2 consecutive → FAULT_CRITICAL")
assert_test(image_ok is False,
            "Test 03b — image_ok=False on FAULT_CRITICAL")

# Test 4: 3 consecutive faults → SHUT_CAMERA
check.reset()
for s in range(3):
    signal, faults, image_ok = check.check(black, step=s)
assert_test(signal == "SHUT_CAMERA",
            "Test 04 — 3 consecutive → SHUT_CAMERA")
assert_test(image_ok is False,
            "Test 04b — image_ok=False on SHUT_CAMERA")
assert_test(check.is_shutdown is True,
            "Test 04c — is_shutdown=True after SHUT_CAMERA")

# Test 5: SHUT_CAMERA stays locked even on a perfect frame
signal, faults, image_ok = check.check(clean, step=3)
assert_test(signal == "SHUT_CAMERA",
            "Test 05 — SHUT_CAMERA stays locked on clean frame")
assert_test(check.is_shutdown is True,
            "Test 05b — is_shutdown remains True")

# Test 6: reset() fully clears shutdown state
check.reset()
assert_test(check.is_shutdown is False,
            "Test 06 — reset() clears is_shutdown")
assert_test(check.consecutive_faults == 0,
            "Test 06b — reset() clears consecutive_faults")
assert_test(check.signal == "CAMERA_OK",
            "Test 06c — reset() resets signal to CAMERA_OK")
signal, faults, image_ok = check.check(clean, step=0)
assert_test(signal == "CAMERA_OK",
            "Test 06d — first check after reset() → CAMERA_OK",
            f"got signal={signal}, faults={faults}")

# Test 7: Fault clears after a clean frame (consecutive counter resets)
check.reset()
check.check(black, step=0)      # 1 fault → WARNING
check.check(clean, step=1)      # clean   → counter resets
signal, faults, image_ok = check.check(clean, step=2)
assert_test(signal == "CAMERA_OK",
            "Test 07 — fault clears after good frame",
            f"got signal={signal}, faults={faults}")
assert_test(check.consecutive_faults == 0,
            "Test 07b — consecutive_faults=0 after recovery")

# Test 8: fault_log records correctly
check.reset()
check.check(black, step=10)
check.check(clean, step=11)
check.check(black, step=12)
assert_test(len(check.fault_log) == 3,
            "Test 08 — fault_log has 3 entries",
            f"got {len(check.fault_log)}")
assert_test(check.fault_log[0][0] == 10,
            "Test 08b — fault_log step index correct",
            f"got {check.fault_log[0][0]}")
assert_test(check.fault_log[1][1] == "CAMERA_OK",
            "Test 08c — fault_log middle entry is CAMERA_OK",
            f"got {check.fault_log[1][1]}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fault Detectors ──────────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 9: Black Frame
check.reset()
black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
signal, faults, image_ok = check.check(black_frame, step=0)
assert_test(
    any("BLACK_FRAME" in f for f in faults),
    "Test 09 — BLACK_FRAME detected on pure black image",
    f"faults={faults}"
)

# Test 10: Blur — flat uniform frame has near-zero Laplacian variance
check.reset()
blur_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
signal, faults, image_ok = check.check(blur_frame, step=0)
assert_test(
    any("BLUR" in f for f in faults),
    "Test 10 — BLUR detected on flat uniform frame",
    f"faults={faults}"
)

# Test 11: Noise — pure random pixels → very high Laplacian variance
check.reset()
noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
signal, faults, image_ok = check.check(noise_frame, step=0)
assert_test(
    any("NOISE" in f for f in faults),
    "Test 11 — NOISE detected on random pixel frame",
    f"faults={faults}"
)

# Test 12: Overexposure
check.reset()
over_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
signal, faults, image_ok = check.check(over_frame, step=0)
assert_test(
    any("OVEREXPOSED" in f for f in faults),
    "Test 12 — OVEREXPOSED detected on fully white frame",
    f"faults={faults}"
)

# Test 13: Underexposure
check.reset()
under_frame = np.full((480, 640, 3), 5, dtype=np.uint8)
signal, faults, image_ok = check.check(under_frame, step=0)
assert_test(
    any("UNDEREXPOSED" in f for f in faults),
    "Test 13 — UNDEREXPOSED detected on near-black frame",
    f"faults={faults}"
)

# Test 16: POV Change — completely different scene from reference
check.reset()
reference             = make_frame(brightness=120)
check.set_reference(reference)
very_different        = np.zeros((480, 640, 3), dtype=np.uint8)
very_different[:,:,2] = 255   # pure red — entirely different HSV from checkerboard
signal, faults, image_ok = check.check(very_different, step=0)
assert_test(
    any("POV_CHANGE" in f for f in faults),
    "Test 16 — POV_CHANGE detected when scene is completely different",
    f"faults={faults}"
)

# Test 17: POV Change NOT triggered on similar frame
check.reset()
check.set_reference(reference)
similar = make_frame(brightness=120, noise_level=2)   # same pattern, tiny noise
signal, faults, image_ok = check.check(similar, step=0)
assert_test(
    not any("POV_CHANGE" in f for f in faults),
    "Test 17 — POV_CHANGE not triggered on similar frame",
    f"faults={faults}"
)

# Test 18: No POV check when reference frame is None
check.reset()
check.reference_frame = None
signal, faults, image_ok = check.check(make_frame(), step=0)
assert_test(
    not any("POV_CHANGE" in f for f in faults),
    "Test 18 — POV_CHANGE skipped when no reference frame set",
    f"faults={faults}"
)


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Config Integration ───────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 19: enabled=False bypasses all checks
config_off = copy.deepcopy(config)
config_off["Camera_SelfCheck"]["enabled"] = False
check_off  = CameraSelfCheck(config_off)
signal, faults, image_ok = check_off.check(black_frame, step=0)
assert_test(signal == "CAMERA_OK",
            "Test 19 — enabled=False bypasses all checks")
assert_test(image_ok is True,
            "Test 19b — image_ok=True when disabled")
assert_test(faults == [],
            "Test 19c — no faults reported when disabled",
            f"got faults={faults}")

# Test 20: Custom consecutive_fault_threshold respected
config_strict = copy.deepcopy(config)
config_strict["Camera_SelfCheck"]["consecutive_fault_threshold"] = 2
check_strict  = CameraSelfCheck(config_strict)
for s in range(2):
    signal, faults, image_ok = check_strict.check(black_frame, step=s)
assert_test(
    signal == "SHUT_CAMERA",
    "Test 20 — consecutive_fault_threshold=2 triggers SHUT_CAMERA at 2 faults",
    f"got signal={signal}"
)


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Results: {tests_passed} passed, {tests_failed} failed")
if tests_failed == 0:
    print(f"  \033[92m✓ All CameraSelfCheck unit tests passed.\033[0m")
else:
    print(f"  \033[91m✗ {tests_failed} test(s) failed — fix before proceeding.\033[0m")
print(f"{'='*55}\n")