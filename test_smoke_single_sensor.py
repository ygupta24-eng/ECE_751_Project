"""
Stage 3 — Single Sensor End-to-End Smoke Test
==============================================
Runs one full episode on the first sensor in your real dataset using the
actual trained TD3 model. Validates the full pipeline works correctly
before launching the parallel batch inference.

Checks:
  - Episode completes without errors
  - All episode_data columns have consistent lengths
  - Camera fault .npy and summary .json are saved correctly
  - camera_signal and camera_faults columns are present and non-empty
  - Fleet summary aggregation produces a valid CSV

Run:
    python test_smoke_single_sensor.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd

# ── Explicit path check before any imports ────────────────────────────────────
# Guards against stale .pyc / wrong working directory causing ImportError
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# Confirm data_utils.py is visible and has prepare_dataset
try:
    import importlib
    import data_utils as _du
    importlib.reload(_du)                      # force reload — clears stale cache
    from data_utils import prepare_dataset
    print("  data_utils imported OK")
except ImportError as e:
    print(f"\nERROR importing data_utils: {e}")
    print(f"Working directory : {os.getcwd()}")
    print(f"sys.path          : {sys.path}")
    sys.exit(1)

from wildfire_env import WildfireEnv
from stable_baselines3 import TD3

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
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


# ── Config and dataset ────────────────────────────────────────────────────────
print("\nLoading config and dataset...")
with open("config_setup_0_90036452TP_0_58399005FP_1dayReservedEnergy.json") as f:
    config = json.load(f)

df        = prepare_dataset(config)
sensor_id = sorted(df["Sensor"].unique())[0]
sensor_df = df[df["Sensor"] == sensor_id].copy()

file_name = config["file_name"]
folder    = f"Inference/episode_plots_step_reward{file_name}"
os.makedirs(folder, exist_ok=True)

print(f"  Sensor selected : {sensor_id}")
print(f"  Rows in sensor  : {len(sensor_df)}")
print(f"  Output folder   : {folder}\n")


# ─────────────────────────────────────────────────────────────────────────────
print("── Running full episode ─────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

env = WildfireEnv(sensor_df, config, start_offset=0)
model = TD3.load(
    "wildfire_td3_20250622_041653_RL1to30min_beta0p9_0.90036452TP_0.58399005FP_noOffset_7daysReservedEng_50perLoss_37571840.zip",
    env=env,
    custom_objects={
        "action_noise":      None,
        "observation_space": env.observation_space,
        "action_space":      env.action_space,
    }
)

obs        = env.reset(sensor_override=sensor_id)
done       = False
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    step_count += 1

print(f"  Episode completed in {step_count} steps\n")


# ─────────────────────────────────────────────────────────────────────────────
print("── Validating episode_data ──────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

n_ts = len(env.episode_data["timestamps"])
assert_test(n_ts > 0, "Test 01 — episode_data has at least 1 step")

required_columns = [
    "timestamps", "battery_levels", "energy_budgets", "missed_fire_times",
    "sampling_time", "harvested_energy", "consumed_energy", "temperature",
    "humidity", "HDWI_score", "wind_speed", "ml_result", "take_a_picture",
    "label", "reward", "camera_signal", "camera_faults",
]

for col in required_columns:
    assert_test(
        col in env.episode_data,
        f"Test 02 — '{col}' key present in episode_data"
    )
    if col in env.episode_data:
        assert_test(
            len(env.episode_data[col]) == n_ts,
            f"Test 02b — '{col}' length matches timestamps ({n_ts})",
            f"got {len(env.episode_data[col])}"
        )

valid_signals = {"CAMERA_OK", "FAULT_WARNING", "FAULT_CRITICAL", "SHUT_CAMERA"}
invalid = [s for s in env.episode_data["camera_signal"] if s not in valid_signals]
assert_test(
    len(invalid) == 0,
    "Test 03 — All camera_signal values are valid states",
    f"Invalid: {set(invalid)}"
)

empty_faults = [f for f in env.episode_data["camera_faults"] if f == ""]
assert_test(
    len(empty_faults) == 0,
    "Test 04 — No empty strings in camera_faults (should be 'none' when clean)"
)


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Validating camera fault log ──────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

fault_log = env.camera_check.fault_log

assert_test(isinstance(fault_log, list), "Test 05 — fault_log is a list")

if len(fault_log) > 0:
    step_idx, sig, faults = fault_log[0]
    assert_test(
        isinstance(step_idx, (int, np.integer)),
        "Test 05b — fault_log entry step index is int",
        f"got type {type(step_idx)}"
    )
    assert_test(
        sig in valid_signals,
        "Test 05c — fault_log entry signal is valid",
        f"got {sig}"
    )
    assert_test(
        isinstance(faults, list),
        "Test 05d — fault_log entry faults is a list",
        f"got type {type(faults)}"
    )

n_ok       = sum(1 for _, s, _ in fault_log if s == "CAMERA_OK")
n_warn     = sum(1 for _, s, _ in fault_log if s == "FAULT_WARNING")
n_crit     = sum(1 for _, s, _ in fault_log if s == "FAULT_CRITICAL")
n_shut     = sum(1 for _, s, _ in fault_log if s == "SHUT_CAMERA")
n_degraded = n_warn + n_crit

print(f"\n  Camera health for sensor {sensor_id}:")
print(f"    Total steps          : {step_count}")
print(f"    Picture attempts     : {len(fault_log)}")
print(f"    CAMERA_OK            : {n_ok}")
print(f"    FAULT_WARNING        : {n_warn}  <- picture taken, ml_result=0")
print(f"    FAULT_CRITICAL       : {n_crit}  <- picture taken, ml_result=0")
print(f"    SHUT_CAMERA          : {n_shut}  <- picture suppressed")
print(f"    Degraded (WARN+CRIT) : {n_degraded}")
print(f"    Shutdown occurred    : {env.camera_check.is_shutdown}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Validating saved output files ────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

np.save(f"{folder}/rewards_{sensor_id}.npy",           env.cumulative_rewards_list)
np.save(f"{folder}/average_rewards_{sensor_id}.npy",   env.average_rewards_list)
np.save(f"{folder}/tensorboard_{sensor_id}.npy",       env.tensorboard_rewards_list)
np.save(f"{folder}/avg_sampling_time_{sensor_id}.npy", env.avg_sampling_time_list)
np.save(f"{folder}/detection_time_{sensor_id}.npy",    env.detection_time_list)

step_log_array  = np.array(sorted(env.step_reward_log,   key=lambda x: x[0]))
step_samp_array = np.array(sorted(env.step_sampling_log, key=lambda x: x[0]))
np.save(f"{folder}/step_rewards_{sensor_id}.npy",       step_log_array)
np.save(f"{folder}/step_sampling_time_{sensor_id}.npy", step_samp_array)

# Camera fault log
if fault_log:
    camera_array = np.array(
        [(s, sig, "; ".join(f) if f else "none") for s, sig, f in fault_log],
        dtype=object
    )
else:
    camera_array = np.array([], dtype=object)
np.save(f"{folder}/camera_faults_{sensor_id}.npy", camera_array)

# Camera summary JSON
summary = {
    "sensor_id":                sensor_id,
    "total_picture_attempts":   len(fault_log),
    "CAMERA_OK":                n_ok,
    "FAULT_WARNING":            n_warn,
    "FAULT_CRITICAL":           n_crit,
    "SHUT_CAMERA":              n_shut,
    "degraded_image_steps":     n_degraded,
    "camera_shutdown_occurred": env.camera_check.is_shutdown,
}
with open(f"{folder}/camera_summary_{sensor_id}.json", "w") as f:
    json.dump(summary, f, indent=4)

# Validate all expected files exist
expected_files = [
    f"rewards_{sensor_id}.npy",
    f"average_rewards_{sensor_id}.npy",
    f"tensorboard_{sensor_id}.npy",
    f"avg_sampling_time_{sensor_id}.npy",
    f"detection_time_{sensor_id}.npy",
    f"step_rewards_{sensor_id}.npy",
    f"step_sampling_time_{sensor_id}.npy",
    f"camera_faults_{sensor_id}.npy",
    f"camera_summary_{sensor_id}.json",
]

for fname in expected_files:
    assert_test(
        os.path.exists(f"{folder}/{fname}"),
        f"Test 06 — '{fname}' saved successfully"
    )

# Reload and validate camera_faults .npy
loaded_faults = np.load(
    f"{folder}/camera_faults_{sensor_id}.npy", allow_pickle=True
)
assert_test(
    isinstance(loaded_faults, np.ndarray),
    "Test 07 — camera_faults .npy reloads as numpy array"
)

# Reload and validate camera_summary .json
with open(f"{folder}/camera_summary_{sensor_id}.json") as f:
    loaded_summary = json.load(f)
expected_keys = [
    "sensor_id", "total_picture_attempts", "CAMERA_OK",
    "FAULT_WARNING", "FAULT_CRITICAL", "SHUT_CAMERA",
    "degraded_image_steps", "camera_shutdown_occurred",
]
for key in expected_keys:
    assert_test(key in loaded_summary, f"Test 08 — camera_summary has key '{key}'")

# Validate episode CSV has camera columns
episode_csv = f"{folder}/episode_1_{sensor_id}.csv"
if os.path.exists(episode_csv):
    df_ep = pd.read_csv(episode_csv)
    assert_test("camera_signal" in df_ep.columns,
                "Test 09 — episode CSV has camera_signal column")
    assert_test("camera_faults" in df_ep.columns,
                "Test 09b — episode CSV has camera_faults column")
else:
    print(f"  (skipping CSV column check — {episode_csv} not found yet)")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Validating fleet summary aggregation ─────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

all_summaries = []
spath = f"{folder}/camera_summary_{sensor_id}.json"
if os.path.exists(spath):
    with open(spath) as f:
        all_summaries.append(json.load(f))

summary_df = pd.DataFrame(all_summaries)
csv_path   = f"{folder}/camera_summary_smoke_test.csv"
summary_df.to_csv(csv_path, index=False)

assert_test(os.path.exists(csv_path),
            "Test 10 — fleet summary CSV saved")
assert_test(len(summary_df) == 1,
            "Test 10b — fleet summary has 1 row")
assert_test("degraded_image_steps" in summary_df.columns,
            "Test 10c — fleet summary has degraded_image_steps column")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Results: {tests_passed} passed, {tests_failed} failed")
if tests_failed == 0:
    print(f"  \033[92m✓ Smoke test passed — safe to run full inference.\033[0m")
else:
    print(f"  \033[91m✗ {tests_failed} test(s) failed — fix before full inference.\033[0m")
print(f"{'='*55}\n")