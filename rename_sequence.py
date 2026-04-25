"""
Image Renamer
==============
Renames all images in a folder to 001.jpg, 002.jpg, 003.jpg...
in alphabetical/chronological order of their original filename.

This is required by test_pov_sequence.py which loads images
in filename order to maintain the correct time sequence.

Run:
    python rename_images.py
"""

import os
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Folder containing images to rename
INPUT_FOLDER  = "F:/Wildfire_Camera/Dataset_images/Mixed_images/POV_Sequence"

# Set to True to preview without renaming
# Set to False to actually rename
DRY_RUN = False

# Set to True to make a backup of originals before renaming
MAKE_BACKUP = True
# ─────────────────────────────────────────────────────────────────────────────


def rename_images(folder: str, dry_run: bool, make_backup: bool):

    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    # Collect all image files sorted by original filename
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    images     = sorted([
        p for p in folder_path.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not images:
        print(f"No images found in {folder}")
        return

    print(f"\n{'='*55}")
    print(f"  IMAGE RENAMER")
    print(f"{'='*55}")
    print(f"  Folder  : {folder}")
    print(f"  Images  : {len(images)}")
    print(f"  Mode    : {'DRY RUN (preview only)' if dry_run else 'LIVE — renaming files'}")
    print(f"  Backup  : {'Yes' if make_backup and not dry_run else 'No'}")
    print(f"{'='*55}\n")

    # Make backup if requested
    if make_backup and not dry_run:
        backup_folder = folder_path.parent / (folder_path.name + "_backup")
        backup_folder.mkdir(exist_ok=True)
        for img in images:
            shutil.copy2(img, backup_folder / img.name)
        print(f"  Backup saved → {backup_folder}\n")

    # Rename
    print(f"  {'Original':<50} → {'New name'}")
    print(f"  {'-'*65}")

    for i, img_path in enumerate(images, start=1):
        new_name = f"{i:03d}.jpg"
        new_path = folder_path / new_name

        print(f"  {img_path.name:<50} → {new_name}")

        if not dry_run:
            img_path.rename(new_path)

    print(f"\n  {'='*55}")
    if dry_run:
        print(f"  DRY RUN complete — no files renamed.")
        print(f"  Set DRY_RUN = False to apply.")
    else:
        print(f"  {len(images)} images renamed successfully.")
    print(f"  {'='*55}\n")


rename_images(INPUT_FOLDER, DRY_RUN, MAKE_BACKUP)