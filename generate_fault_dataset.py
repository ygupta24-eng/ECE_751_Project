"""
Fault Dataset Generator
========================
Works on top of your existing 2000 real images from the HPWREN dataset.

What this does:
  1. Augments your existing good/ images with realistic natural variations
     to increase the good class size (time of day, weather, season, night)
  2. Generates synthetic severity variants from your existing fault/ images
     to increase fault coverage across severity levels
  3. Leaves pov_change/ completely untouched — POV detection is camera-node
     specific and cannot be synthesized without a known reference frame

What this does NOT do:
  - Generate POV change images (camera-node specific property)
  - Generate blackout images (removed — night/blackout ambiguity)
  - Generate frozen frame images (removed — static camera false positives)

Input structure (your existing 2000 images):
    source_dataset/
        good/           ← real normal wildfire camera frames
        fault/          ← real faulty frames (blur, noise, over/underexposed)
        pov_change/     ← manually collected different-scene images (untouched)

Output structure:
    generated_test_dataset/
        good/           ← originals + augmented variants
        blur/           ← real blur faults + synthetic severity variants
        noise/          ← real noise faults + synthetic severity variants
        overexposed/    ← real overexposed + synthetic severity variants
        underexposed/   ← real underexposed + synthetic severity variants
        pov_change/     ← COPIED AS-IS from source (no modification)

Run:
    python generate_fault_dataset.py
"""

import cv2
import numpy as np
import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_DATASET_DIR  = "F:/Wildfire_Camera/Dataset_images/Mixed_images"
OUTPUT_DATASET_DIR  = "F:/Wildfire_Camera/Dataset_images/generated_test_dataset"
RANDOM_SEED         = 42

# ── Good image augmentation ───────────────────────────────────────────────────
# Each variant = one realistic variation of the same camera scene.
# NOT faults — must all pass every fault detector cleanly.
# Night variants deliberately included to verify underexposure detector
# does not false-positive on low-light scenes.

GOOD_AUGMENTATIONS = [
    # ── Time of day ───────────────────────────────────────────────────────────
    {"name": "dawn",         "brightness": 0.70, "gamma": 1.20,
     "haze": 0.10, "noise_std": 2},
    {"name": "morning",      "brightness": 0.85, "gamma": 1.10,
     "haze": 0.00, "noise_std": 1},
    {"name": "noon",         "brightness": 1.15, "gamma": 0.90,
     "haze": 0.00, "noise_std": 1},
    {"name": "afternoon",    "brightness": 1.00, "gamma": 1.00,
     "haze": 0.05, "noise_std": 1},
    {"name": "dusk",         "brightness": 0.60, "gamma": 1.30,
     "haze": 0.15, "noise_std": 3},
    {"name": "twilight",     "brightness": 0.40, "gamma": 1.50,
     "haze": 0.00, "noise_std": 4},

    # ── Night (must NOT trigger any fault detector) ───────────────────────────
    {"name": "night_1",      "brightness": 0.15, "gamma": 2.00,
     "haze": 0.00, "noise_std": 6},
    {"name": "night_2",      "brightness": 0.20, "gamma": 1.90,
     "haze": 0.00, "noise_std": 5},
    {"name": "night_moon",   "brightness": 0.25, "gamma": 1.70,
     "haze": 0.05, "noise_std": 4},
    {"name": "night_ir",     "brightness": 0.30, "gamma": 1.80,
     "haze": 0.00, "noise_std": 5},

    # ── Weather ───────────────────────────────────────────────────────────────
    {"name": "overcast",     "brightness": 0.80, "gamma": 1.10,
     "haze": 0.20, "noise_std": 1},
    {"name": "hazy",         "brightness": 0.90, "gamma": 1.05,
     "haze": 0.35, "noise_std": 2},
    {"name": "foggy",        "brightness": 0.75, "gamma": 1.10,
     "haze": 0.50, "noise_std": 2},
    {"name": "smoke_light",  "brightness": 0.85, "gamma": 1.00,
     "haze": 0.25, "noise_std": 1},

    # ── Seasonal colour shifts ────────────────────────────────────────────────
    {"name": "summer",       "brightness": 1.05, "gamma": 0.95,
     "haze": 0.00, "noise_std": 0, "color_shift": ( 0,  5,  -5)},
    {"name": "autumn",       "brightness": 0.95, "gamma": 1.00,
     "haze": 0.05, "noise_std": 0, "color_shift": ( 0, -10, 15)},
    {"name": "winter",       "brightness": 0.80, "gamma": 1.10,
     "haze": 0.10, "noise_std": 0, "color_shift": ( 5,   0, -10)},

    # ── Sensor / compression variation ───────────────────────────────────────
    {"name": "compress_high","brightness": 1.00, "gamma": 1.00,
     "haze": 0.00, "noise_std": 1},
    {"name": "compress_low", "brightness": 1.00, "gamma": 1.00,
     "haze": 0.00, "noise_std": 3},
    {"name": "slight_warm",  "brightness": 1.00, "gamma": 1.00,
     "haze": 0.00, "noise_std": 0, "color_shift": ( 0,  -5, 10)},
    {"name": "slight_cool",  "brightness": 1.00, "gamma": 1.00,
     "haze": 0.00, "noise_std": 0, "color_shift": (10,   0, -5)},
]

# ── Synthetic fault severity levels ──────────────────────────────────────────
# These are applied to your existing real fault images to extend severity coverage.
# Each generates additional variants beyond what you already have.

BLUR_EXTRA_KERNELS      = [11, 21, 31, 45, 61, 71]   # additional blur severities
NOISE_EXTRA_DENSITIES   = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.20]
OVEREXPOSE_EXTRA        = [1.6, 2.0, 2.4, 2.6, 2.8, 2.9]
UNDEREXPOSE_EXTRA       = [0.10, 0.20, 0.30, 0.40, 0.50]
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)


# ─── Utilities ────────────────────────────────────────────────────────────────
def load_images(folder: str) -> list:
    """Load all jpg/png from folder. Returns list of (stem, frame)."""
    if not os.path.exists(folder):
        return []
    paths = (sorted(Path(folder).glob("*.jpg")) +
             sorted(Path(folder).glob("*.png")))
    result = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            result.append((p.stem, frame))
    return result


def save(frame: np.ndarray, folder: str, filename: str):
    cv2.imwrite(os.path.join(folder, filename), frame)


# ─── Augmentation ─────────────────────────────────────────────────────────────
def augment_good(frame: np.ndarray, aug: dict) -> np.ndarray:
    out = frame.astype(np.float32)

    out = out * aug.get("brightness", 1.0)

    gamma = aug.get("gamma", 1.0)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)], dtype=np.float32)
        out = cv2.LUT(
            np.clip(out, 0, 255).astype(np.uint8),
            table.astype(np.uint8)
        ).astype(np.float32)

    haze = aug.get("haze", 0.0)
    if haze > 0:
        out = out * (1 - haze) + 255 * haze

    for i, shift in enumerate(aug.get("color_shift", (0, 0, 0))):
        if shift != 0:
            out[:, :, i] = np.clip(out[:, :, i] + shift, 0, 255)

    noise_std = aug.get("noise_std", 0)
    if noise_std > 0:
        out = out + np.random.normal(0, noise_std, out.shape).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


# ─── Fault generators ─────────────────────────────────────────────────────────
def apply_blur(frame: np.ndarray, k: int) -> np.ndarray:
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def apply_salt_pepper(frame: np.ndarray, density: float) -> np.ndarray:
    out    = frame.copy()
    n      = int(density / 2 * frame.size / 3)
    for val in [255, 0]:
        coords = [np.random.randint(0, i, n) for i in frame.shape[:2]]
        out[coords[0], coords[1]] = val
    return out


def apply_overexpose(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_underexpose(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ─── Main ─────────────────────────────────────────────────────────────────────
def generate_dataset():
    print(f"\n{'='*65}")
    print(f"  FAULT DATASET GENERATOR")
    print(f"  Working on top of your existing 2000 real images")
    print(f"{'='*65}")
    print(f"  Source  : {SOURCE_DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DATASET_DIR}")
    print(f"  Seed    : {RANDOM_SEED}")

    # ── Load source images ────────────────────────────────────────────────────
    print(f"\nLoading source images...")
    good_src  = load_images(os.path.join(SOURCE_DATASET_DIR, "good"))
    fault_src = load_images(os.path.join(SOURCE_DATASET_DIR, "fault"))
    pov_src   = load_images(os.path.join(SOURCE_DATASET_DIR, "pov_change"))

    print(f"  good/       : {len(good_src)} images")
    print(f"  fault/      : {len(fault_src)} images")
    print(f"  pov_change/ : {len(pov_src)} images  (will be copied as-is)")

    if not good_src:
        print(f"\nERROR: No good images found in "
              f"{SOURCE_DATASET_DIR}/good")
        return

    # ── Create output folders ─────────────────────────────────────────────────
    for cat in ["good", "blur", "noise", "overexposed", "underexposed", "pov_change"]:
        os.makedirs(os.path.join(OUTPUT_DATASET_DIR, cat), exist_ok=True)

    counts = {
        "good":         0,
        "blur":         0,
        "noise":        0,
        "overexposed":  0,
        "underexposed": 0,
        "pov_change":   0,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Good images: copy originals + augmented variants
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 1] Good images")
    print(f"  Copying {len(good_src)} originals + generating "
          f"{len(GOOD_AUGMENTATIONS)} variants each...")

    out_good = os.path.join(OUTPUT_DATASET_DIR, "good")

    for stem, frame in good_src:
        # Copy original
        save(frame, out_good, f"{stem}_original.jpg")
        counts["good"] += 1
        # Augmented variants
        for aug in GOOD_AUGMENTATIONS:
            save(augment_good(frame, aug), out_good,
                 f"{stem}_{aug['name']}.jpg")
            counts["good"] += 1

    night_count = len([a for a in GOOD_AUGMENTATIONS if "night" in a["name"]])
    print(f"  ✓ {counts['good']} good images total")
    print(f"    ({len(good_src)} originals + "
          f"{len(good_src) * len(GOOD_AUGMENTATIONS)} augmented)")
    print(f"    Night variants: {len(good_src) * night_count} "
          f"— must NOT trigger fault detectors")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Fault images: copy real faults + add synthetic severity variants
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] Fault images")
    print(f"  Copying {len(fault_src)} real fault images + "
          f"generating severity variants...")

    out_blur  = os.path.join(OUTPUT_DATASET_DIR, "blur")
    out_noise = os.path.join(OUTPUT_DATASET_DIR, "noise")
    out_over  = os.path.join(OUTPUT_DATASET_DIR, "overexposed")
    out_under = os.path.join(OUTPUT_DATASET_DIR, "underexposed")

    # Copy all real fault images into appropriate subfolders
    # Use filename to determine which fault type they belong to
    fault_originals = {"blur": 0, "noise": 0,
                       "overexposed": 0, "underexposed": 0, "other": 0}

    for stem, frame in fault_src:
        stem_lower = stem.lower()
        if any(k in stem_lower for k in ["blur", "blurry", "focus"]):
            save(frame, out_blur,  f"REAL_{stem}.jpg")
            counts["blur"] += 1
            fault_originals["blur"] += 1
        elif any(k in stem_lower for k in ["noise", "salt", "pepper", "sp_"]):
            save(frame, out_noise, f"REAL_{stem}.jpg")
            counts["noise"] += 1
            fault_originals["noise"] += 1
        elif any(k in stem_lower for k in ["over", "bright", "glare"]):
            save(frame, out_over,  f"REAL_{stem}.jpg")
            counts["overexposed"] += 1
            fault_originals["overexposed"] += 1
        elif any(k in stem_lower for k in ["under", "dark", "dim"]):
            save(frame, out_under, f"REAL_{stem}.jpg")
            counts["underexposed"] += 1
            fault_originals["underexposed"] += 1
        else:
            # Unclassified real fault — copy to all four fault folders
            # so nothing is lost
            for folder, key in [(out_blur,  "blur"),
                                 (out_noise, "noise"),
                                 (out_over,  "overexposed"),
                                 (out_under, "underexposed")]:
                save(frame, folder, f"REAL_unclassified_{stem}.jpg")
                counts[key] += 1
            fault_originals["other"] += 1

    print(f"  Real faults copied:")
    for k, v in fault_originals.items():
        if v > 0:
            print(f"    {k:<15} : {v}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Synthetic severity variants from good source images
    # Generate additional fault images at controlled severity levels
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Synthetic severity variants from good source images...")

    # Blur variants
    print(f"  Blur    : {len(good_src)} × {len(BLUR_EXTRA_KERNELS)} kernels...")
    for stem, frame in good_src:
        for k in BLUR_EXTRA_KERNELS:
            save(apply_blur(frame, k), out_blur,
                 f"SYNTH_blur_k{k:02d}_{stem}.jpg")
            counts["blur"] += 1

    # Noise variants
    print(f"  Noise   : {len(good_src)} × {len(NOISE_EXTRA_DENSITIES)} densities...")
    for stem, frame in good_src:
        for d in NOISE_EXTRA_DENSITIES:
            d_str = f"{d:.2f}".replace(".", "")
            save(apply_salt_pepper(frame, d), out_noise,
                 f"SYNTH_saltpepper_d{d_str}_{stem}.jpg")
            counts["noise"] += 1

    # Overexposed variants
    print(f"  Overexp : {len(good_src)} × {len(OVEREXPOSE_EXTRA)} factors...")
    for stem, frame in good_src:
        for f_val in OVEREXPOSE_EXTRA:
            f_str = f"{f_val:.1f}".replace(".", "")
            save(apply_overexpose(frame, f_val), out_over,
                 f"SYNTH_overexpose_f{f_str}_{stem}.jpg")
            counts["overexposed"] += 1

    # Underexposed variants
    print(f"  Underexp: {len(good_src)} × {len(UNDEREXPOSE_EXTRA)} factors...")
    for stem, frame in good_src:
        for f_val in UNDEREXPOSE_EXTRA:
            f_str = f"{f_val:.2f}".replace(".", "")
            save(apply_underexpose(frame, f_val), out_under,
                 f"SYNTH_underexpose_f{f_str}_{stem}.jpg")
            counts["underexposed"] += 1

    print(f"  ✓ Synthetic faults generated:")
    print(f"    blur         : {counts['blur']}")
    print(f"    noise        : {counts['noise']}")
    print(f"    overexposed  : {counts['overexposed']}")
    print(f"    underexposed : {counts['underexposed']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — POV change: copy as-is, no modification
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 4] POV change images — copying as-is (no modification)...")

    out_pov     = os.path.join(OUTPUT_DATASET_DIR, "pov_change")
    src_pov_dir = os.path.join(SOURCE_DATASET_DIR,  "pov_change")

    if pov_src:
        for stem, frame in pov_src:
            save(frame, out_pov, f"{stem}.jpg")
            counts["pov_change"] += 1
        print(f"  ✓ {counts['pov_change']} POV change images copied")
    else:
        print(f"  ⚠ No POV change images found in {src_pov_dir}")
        print(f"  Add manually — use genuinely different scenes")
        print(f"  (indoor, different landscapes, unrelated environments)")

    print(f"\n  NOTE: POV detection is camera-node specific.")
    print(f"  These images are used only in test_pov_change.py")
    print(f"  and test_fault_detectors.py Phase 6.")
    print(f"  They cannot be synthesized — each camera has its own")
    print(f"  reference frame and the correlation is relative to that.")

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    total       = sum(counts.values())
    fault_total = total - counts["good"] - counts["pov_change"]
    good_pct    = counts["good"]       / total * 100 if total > 0 else 0
    fault_pct   = fault_total          / total * 100 if total > 0 else 0
    pov_pct     = counts["pov_change"] / total * 100 if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*65}")
    print(f"  {'Category':<20} {'Count':>8}   {'%':>6}")
    print(f"  {'-'*40}")
    for cat, count in counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {cat:<20} {count:>8}   {pct:>5.1f}%")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<20} {total:>8}   100.0%")

    print(f"\n  Class distribution:")
    print(f"    Good frames  : {counts['good']:>6}  ({good_pct:.1f}%)  ← dominant class")
    print(f"    Fault frames : {fault_total:>6}  ({fault_pct:.1f}%)")
    print(f"    POV change   : {counts['pov_change']:>6}  ({pov_pct:.1f}%)  "
          f"← node-specific, test separately")

    print(f"\n  Detectors covered  : Blur, Noise, Overexposure, Underexposure")
    print(f"  POV change         : Tested separately per camera node")
    print(f"  Blackout           : Removed (night/blackout ambiguity)")
    print(f"  Frozen frame       : Removed (static camera false positives)")

    # Save manifest
    manifest = {
        "generated_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed":           RANDOM_SEED,
        "source_good_images":    len(good_src),
        "source_fault_images":   len(fault_src),
        "source_pov_images":     len(pov_src),
        "counts":                counts,
        "total":                 total,
        "good_augmentations":    [a["name"] for a in GOOD_AUGMENTATIONS],
        "night_variants":        [a["name"] for a in GOOD_AUGMENTATIONS
                                  if "night" in a["name"]],
        "blur_extra_kernels":    BLUR_EXTRA_KERNELS,
        "noise_extra_densities": NOISE_EXTRA_DENSITIES,
        "overexpose_extra":      OVEREXPOSE_EXTRA,
        "underexpose_extra":     UNDEREXPOSE_EXTRA,
        "detectors_included":    ["blur", "noise",
                                  "overexposure", "underexposure"],
        "detectors_excluded": {
            "blackout":     "Night images statistically indistinguishable "
                            "from hardware blackout via brightness threshold. "
                            "False positives suppress fire detection.",
            "frozen_frame": "Static wildfire cameras naturally produce "
                            "similar consecutive frames on calm days.",
        },
        "pov_note": (
            "POV detection is camera-node specific. The HSV correlation "
            "is computed against a reference frame belonging to a specific "
            "camera node. Cannot be synthesized without knowing which "
            "camera's reference frame to compare against. "
            "Test separately using test_pov_change.py."
        ),
    }

    manifest_path = os.path.join(OUTPUT_DATASET_DIR, "generation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n  Manifest → {manifest_path}")
    print(f"  Dataset  → {OUTPUT_DATASET_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    generate_dataset()