"""
Stage 2 — WildfireEnv Camera Wiring Tests
==========================================
Tests that take_picture, ml_result, energy, and episode_data are all wired
correctly for each camera signal state inside WildfireEnv.step().

Uses unittest.mock to control exactly what CameraSelfCheck.check() returns
so we can test each signal path in isolation without needing real images.

Run:
    python test_env_camera_wiring.py
"""

import numpy as np
import pandas as pd
import json
import copy
from unittest.mock import patch, MagicMock
from wildfire_env import WildfireEnv, CameraSelfCheck

# ── Load config ───────────────────────────────────────────────────────────────
with open("config_setup_0_90036452TP_0_58399005FP_1dayReservedEnergy.json") as f:
    config = json.load(f)

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


def make_fake_df(n=50, label=0, sensor_id=1):
    """Build a minimal sensor dataframe that WildfireEnv accepts."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({
        "Sensor":                         [sensor_id] * n,
        "Timestamp":                      timestamps,
        "Temperature_2m":                 [25.0] * n,
        "Temperature_2m_normalized":      [0.5]  * n,
        "Relative_Humidity_2m":           [40.0] * n,
        "Relative_Humidity_2m_normalized":[0.4]  * n,
        "Wind_Speed_10m":                 [0.3]  * n,
        "HDWI":                           [0.2]  * n,
        "solar_energy":                   [0.01] * n,
        "Time_of_Day":                    [0.5]  * n,
        "Season":                         [0.5]  * n,
        "Label":                          [label]* n,
    })


def make_env(cfg, label=0, sensor_id=1):
    """Create a WildfireEnv with a mocked DT model that always says take_picture=1."""
    df = make_fake_df(n=50, label=label, sensor_id=sensor_id)
    with patch("wildfire_env.load") as mock_load:
        mock_dt       = MagicMock()
        mock_dt.predict.return_value = [1]   # DT model always says take picture
        mock_load.return_value = mock_dt
        env = WildfireEnv(df, cfg)
    return env


def step_with_camera_signal(env, sensor_id, camera_return, action=5):
    """
    Reset env, patch camera check to return a fixed (signal, faults, image_ok),
    take one step, and return episode_data from that step.
    """
    env.reset(sensor_override=sensor_id)
    with patch.object(env.camera_check, "check", return_value=camera_return):
        env.step(np.array([action]))
    return env.episode_data


# ─────────────────────────────────────────────────────────────────────────────
print("\n── take_picture and ml_result per signal state ──────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

env = make_env(config)

# Test 1: CAMERA_OK → take_picture=1, ml_result runs (not forced to 0)
data = step_with_camera_signal(env, 1, ("CAMERA_OK", [], True))
assert_test(data["take_a_picture"][-1] == 1,
            "Test 01 — CAMERA_OK: take_picture=1")
assert_test(data["camera_signal"][-1] == "CAMERA_OK",
            "Test 01b — CAMERA_OK: camera_signal logged correctly")
# ml_result can be 0 or 1 (stochastic) — just confirm it is one of them
assert_test(data["ml_result"][-1] in [0, 1],
            "Test 01c — CAMERA_OK: ml_result is 0 or 1 (ML ran)")

# Test 2: FAULT_WARNING → take_picture=1, ml_result forced to 0
data = step_with_camera_signal(
    env, 1,
    ("FAULT_WARNING", ["BLACK_FRAME(brightness=2.0)"], False)
)
assert_test(data["take_a_picture"][-1] == 1,
            "Test 02 — FAULT_WARNING: take_picture=1 (camera still fires)")
assert_test(data["ml_result"][-1] == 0,
            "Test 02b — FAULT_WARNING: ml_result forced to 0")
assert_test(data["camera_signal"][-1] == "FAULT_WARNING",
            "Test 02c — FAULT_WARNING: camera_signal logged correctly")
assert_test("BLACK_FRAME" in data["camera_faults"][-1],
            "Test 02d — FAULT_WARNING: fault name logged in camera_faults")

# Test 3: FAULT_CRITICAL → take_picture=1, ml_result forced to 0
data = step_with_camera_signal(
    env, 1,
    ("FAULT_CRITICAL", ["BLUR(lap=10.0)"], False)
)
assert_test(data["take_a_picture"][-1] == 1,
            "Test 03 — FAULT_CRITICAL: take_picture=1 (camera still fires)")
assert_test(data["ml_result"][-1] == 0,
            "Test 03b — FAULT_CRITICAL: ml_result forced to 0")
assert_test(data["camera_signal"][-1] == "FAULT_CRITICAL",
            "Test 03c — FAULT_CRITICAL: camera_signal logged correctly")

# Test 4: SHUT_CAMERA → take_picture=0, ml_result=0, camera fully suppressed
data = step_with_camera_signal(
    env, 1,
    ("SHUT_CAMERA", ["CAMERA_OFFLINE"], False)
)
assert_test(data["take_a_picture"][-1] == 0,
            "Test 04 — SHUT_CAMERA: take_picture=0 (camera suppressed)")
assert_test(data["ml_result"][-1] == 0,
            "Test 04b — SHUT_CAMERA: ml_result=0")
assert_test(data["camera_signal"][-1] == "SHUT_CAMERA",
            "Test 04c — SHUT_CAMERA: camera_signal logged correctly")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── Energy accounting per signal state ───────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 5: CAMERA_OK → camera + selfcheck energy charged
env2 = make_env(config)
env2.reset(sensor_override=1)
battery_before = env2.battery_energy

with patch.object(env2.camera_check, "check", return_value=("CAMERA_OK", [], True)):
    env2.step(np.array([5]))

energy_ok = env2.episode_data["consumed_energy"][-1]

# Test 6: SHUT_CAMERA → camera + selfcheck energy NOT charged
env3 = make_env(config)
env3.reset(sensor_override=1)

with patch.object(env3.camera_check, "check",
                  return_value=("SHUT_CAMERA", ["CAMERA_OFFLINE"], False)):
    env3.step(np.array([5]))

energy_shut = env3.episode_data["consumed_energy"][-1]

# CAMERA_OK consumes more energy than SHUT_CAMERA (camera + selfcheck + ML)
assert_test(
    energy_ok > energy_shut,
    "Test 05/06 — CAMERA_OK consumes more energy than SHUT_CAMERA",
    f"CAMERA_OK={energy_ok:.8f}, SHUT_CAMERA={energy_shut:.8f}"
)

# Test 7: FAULT_WARNING → camera energy charged (same as CAMERA_OK, picture was taken)
env4 = make_env(config)
env4.reset(sensor_override=1)
with patch.object(env4.camera_check, "check",
                  return_value=("FAULT_WARNING", ["BLACK_FRAME"], False)):
    env4.step(np.array([5]))
energy_warn = env4.episode_data["consumed_energy"][-1]

# WARNING and OK should cost the same camera energy (both take picture)
# They differ only if OK triggers ml comm energy — for label=0, FP may or may not fire
# So just confirm WARNING costs more than SHUT
assert_test(
    energy_warn > energy_shut,
    "Test 07 — FAULT_WARNING charges more energy than SHUT_CAMERA",
    f"FAULT_WARNING={energy_warn:.8f}, SHUT_CAMERA={energy_shut:.8f}"
)


# ─────────────────────────────────────────────────────────────────────────────
print("\n── episode_data integrity ───────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 8: camera_signal and camera_faults keys exist in episode_data
env5 = make_env(config)
env5.reset(sensor_override=1)
assert_test("camera_signal" in env5.episode_data,
            "Test 08 — camera_signal key exists in episode_data")
assert_test("camera_faults" in env5.episode_data,
            "Test 08b — camera_faults key exists in episode_data")

# Test 9: After multiple steps, all lists are same length
with patch.object(env5.camera_check, "check",
                  return_value=("CAMERA_OK", [], True)):
    for _ in range(5):
        env5.step(np.array([1]))

n_ts = len(env5.episode_data["timestamps"])
assert_test(
    len(env5.episode_data["camera_signal"]) == n_ts,
    "Test 09 — camera_signal list length matches timestamps"
)
assert_test(
    len(env5.episode_data["camera_faults"]) == n_ts,
    "Test 09b — camera_faults list length matches timestamps"
)

# Test 10: No faults → camera_faults logged as "none"
env6 = make_env(config)
env6.reset(sensor_override=1)
with patch.object(env6.camera_check, "check",
                  return_value=("CAMERA_OK", [], True)):
    env6.step(np.array([5]))
assert_test(env6.episode_data["camera_faults"][-1] == "none",
            "Test 10 — No faults logged as 'none' string")

# Test 11: Faults logged as semicolon-separated string
env7 = make_env(config)
env7.reset(sensor_override=1)
with patch.object(env7.camera_check, "check",
                  return_value=("FAULT_WARNING",
                                ["BLACK_FRAME(brightness=2.0)", "BLUR(lap=50.0)"],
                                False)):
    env7.step(np.array([5]))
faults_str = env7.episode_data["camera_faults"][-1]
assert_test("BLACK_FRAME" in faults_str and "BLUR" in faults_str,
            "Test 11 — Multiple faults logged as semicolon-separated string")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── DT model says take_picture=0 (no camera check should run) ────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 12: When DT model returns 0, camera check should never be called
df_no = make_fake_df(n=50, label=0, sensor_id=1)
with patch("wildfire_env.load") as mock_load:
    mock_dt = MagicMock()
    mock_dt.predict.return_value = [0]   # DT says do NOT take picture
    mock_load.return_value = mock_dt
    env8 = WildfireEnv(df_no, config)

env8.reset(sensor_override=1)
with patch.object(env8.camera_check, "check") as mock_check:
    env8.step(np.array([5]))
    assert_test(
        mock_check.call_count == 0,
        "Test 12 — CameraSelfCheck.check() not called when DT says take_picture=0"
    )
assert_test(env8.episode_data["camera_signal"][-1] == "CAMERA_OK",
            "Test 12b — camera_signal defaults to CAMERA_OK when no picture attempted")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── enabled=False bypasses self-check ────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 13: enabled=False — check() never touches detectors
cfg_off = copy.deepcopy(config)
cfg_off["Camera_SelfCheck"]["enabled"] = False
env9 = make_env(cfg_off)
env9.reset(sensor_override=1)
env9.step(np.array([5]))
assert_test(env9.episode_data["camera_signal"][-1] == "CAMERA_OK",
            "Test 13 — enabled=False → camera_signal always CAMERA_OK")
assert_test(env9.episode_data["take_a_picture"][-1] == 1,
            "Test 13b — enabled=False → take_picture not suppressed")


# ─────────────────────────────────────────────────────────────────────────────
print("\n── reset() between episodes ─────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

# Test 14: Camera state cleared on reset — shutdown from ep1 doesn't leak into ep2
env10 = make_env(config)
env10.reset(sensor_override=1)

# Force camera into SHUT_CAMERA during episode 1
with patch.object(env10.camera_check, "check",
                  return_value=("SHUT_CAMERA", ["CAMERA_OFFLINE"], False)):
    env10.step(np.array([5]))

assert env10.camera_check.is_shutdown is True, "Pre-condition: camera shut during ep1"

# Reset for episode 2 — shutdown should be cleared
env10.reset(sensor_override=1)
assert_test(env10.camera_check.is_shutdown is False,
            "Test 14 — reset() clears shutdown state between episodes")
assert_test(env10.camera_check.consecutive_faults == 0,
            "Test 14b — reset() clears consecutive_faults between episodes")
assert_test(len(env10.camera_check.fault_log) == 0,
            "Test 14c — reset() clears fault_log between episodes")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Results: {tests_passed} passed, {tests_failed} failed")
if tests_failed == 0:
    print(f"  \033[92m✓ All environment wiring tests passed.\033[0m")
else:
    print(f"  \033[91m✗ {tests_failed} test(s) failed — fix before proceeding.\033[0m")
print(f"{'='*55}\n")
