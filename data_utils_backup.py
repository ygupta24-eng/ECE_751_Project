"""
Finds and fixes deprecated pandas frequency aliases in data_utils.py.
pandas 2.2+ replaced "min" with "min" and "h" with "h".

Run:
    python fix_data_utils.py
"""

import re

INPUT_FILE  = "data_utils.py"
BACKUP_FILE = "data_utils_backup.py"

# Read original file
with open(INPUT_FILE, "r") as f:
    original = f.read()

# Save a backup before making any changes
with open(BACKUP_FILE, "w") as f:
    f.write(original)
print(f"Backup saved → {BACKUP_FILE}")

# ── Replacement rules ─────────────────────────────────────────────────────────
# Order matters — replace longer patterns first so "1min" is caught before "min"
replacements = [
    # Minute aliases
    ('"1min"',   '"1min"'),
    ("'1min'",   "'1min'"),
    ('"1min"', '"1min"'),   # already correct — no-op
    ('"min"',    '"min"'),
    ("'min'",    "'min'"),
    # Hour aliases
    ('"1h"',   '"1h"'),
    ("'1h'",   "'1h'"),
    ('"h"',    '"h"'),
    ("'h'",    "'h'"),
    # Second aliases (fix while we are here)
    ('"1s"',   '"1s"'),
    ("'1s'",   "'1s'"),
    ('"s"',    '"s"'),
    ("'s'",    "'s'"),
]

fixed   = original
changes = []

for old, new in replacements:
    if old in fixed:
        count = fixed.count(old)
        fixed = fixed.replace(old, new)
        changes.append((old, new, count))

# ── Report what changed ───────────────────────────────────────────────────────
if changes:
    print("\nReplacements made:")
    for old, new, count in changes:
        print(f"  {old:10s}  →  {new:10s}   ({count} occurrence(s))")
    with open(INPUT_FILE, "w") as f:
        f.write(fixed)
    print(f"\ndata_utils.py updated successfully.")
else:
    print("\nNo deprecated frequency aliases found — data_utils.py is already up to date.")

# ── Print all date_range / resample / asfreq lines for visual confirmation ────
print("\nAll frequency-related lines in data_utils.py after fix:")
for i, line in enumerate(fixed.splitlines(), 1):
    if any(kw in line for kw in ["date_range", "resample", "asfreq", "freq=", "offset"]):
        print(f"  Line {i:3d}: {line.rstrip()}")