import numpy as np
import json
from datetime import datetime

from stable_baselines3 import TD3

import torch

from multiprocessing import Pool
import os
import sys

from data_utils import prepare_dataset
from wildfire_env import WildfireEnv, Logger

# Set deterministic CPU behavior
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Reduces nondeterminism from GPU kernel choices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run_inference_with_sensor_id(sensor_id, sensor_df, start_offset, config):

    file_name = config["file_name"]
    folder    = f"Inference/episode_plots_step_reward{file_name}"

    # ── Recreate environment and load trained TD3 model ───────────────────────
    env = WildfireEnv(sensor_df, config, start_offset=start_offset)
    model = TD3.load(
        "wildfire_td3_20250622_041653_RL1to30min_beta0p9_0.90036452TP_0.58399005FP_noOffset_7daysReservedEng_50perLoss_37571840.zip",
        env=env,
        custom_objects={
            "action_noise":      None,
            "observation_space": env.observation_space,
            "action_space":      env.action_space,
        }
    )

    obs  = env.reset(sensor_override=sensor_id)
    done = False

    # ── Run full episode ───────────────────────────────────────────────────────
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    # ── Save existing reward / sampling outputs ────────────────────────────────
    np.save(f"{folder}/rewards_{sensor_id}.npy",            env.cumulative_rewards_list)
    np.save(f"{folder}/average_rewards_{sensor_id}.npy",    env.average_rewards_list)
    np.save(f"{folder}/tensorboard_{sensor_id}.npy",        env.tensorboard_rewards_list)
    np.save(f"{folder}/avg_sampling_time_{sensor_id}.npy",  env.avg_sampling_time_list)
    np.save(f"{folder}/detection_time_{sensor_id}.npy",     env.detection_time_list)

    step_log_array          = np.array(sorted(env.step_reward_log,   key=lambda x: x[0]))
    step_sampling_log_array = np.array(sorted(env.step_sampling_log, key=lambda x: x[0]))
    np.save(f"{folder}/step_rewards_{sensor_id}.npy",       step_log_array)
    np.save(f"{folder}/step_sampling_time_{sensor_id}.npy", step_sampling_log_array)

    # ── NEW: Save full camera fault log ───────────────────────────────────────
    # Each entry: (step, signal, fault_string)
    # Stored as object array — mixed int/str types require dtype=object.
    #
    # The fault log records every step where take_picture_attempted == 1,
    # covering all four signals:
    #   CAMERA_OK      → image was good, ML ran normally
    #   FAULT_WARNING  → image taken but degraded, ml_result forced to 0
    #   FAULT_CRITICAL → image taken but degraded, ml_result forced to 0
    #   SHUT_CAMERA    → camera suppressed, take_picture set to 0
    if env.camera_check.fault_log:
        camera_fault_array = np.array(
            [
                (step, signal, "; ".join(faults) if faults else "none")
                for step, signal, faults in env.camera_check.fault_log
            ],
            dtype=object
        )
    else:
        camera_fault_array = np.array([], dtype=object)

    np.save(f"{folder}/camera_faults_{sensor_id}.npy", camera_fault_array)

    # ── NEW: Save per-sensor camera health summary ─────────────────────────────
    # Counts each signal level that appeared during this episode.
    # camera_shutdown_occurred=True means the camera hit SHUT_CAMERA at some
    # point and was locked out for the rest of the episode.
    fault_log    = env.camera_check.fault_log
    n_ok         = sum(1 for _, sig, _ in fault_log if sig == "CAMERA_OK")
    n_warning    = sum(1 for _, sig, _ in fault_log if sig == "FAULT_WARNING")
    n_critical   = sum(1 for _, sig, _ in fault_log if sig == "FAULT_CRITICAL")
    n_shutdown   = sum(1 for _, sig, _ in fault_log if sig == "SHUT_CAMERA")

    # Steps where picture was attempted but image quality was too poor for ML
    # (WARNING + CRITICAL) — the picture was taken and energy was charged,
    # but ml_result was forced to 0 due to degraded image.
    n_degraded_image = n_warning + n_critical

    summary = {
        "sensor_id":                    sensor_id,
        "total_picture_attempts":       len(fault_log),
        "CAMERA_OK":                    n_ok,
        "FAULT_WARNING":                n_warning,
        "FAULT_CRITICAL":               n_critical,
        "SHUT_CAMERA":                  n_shutdown,
        "degraded_image_steps":         n_degraded_image,
        "camera_shutdown_occurred":     env.camera_check.is_shutdown,
    }

    with open(f"{folder}/camera_summary_{sensor_id}.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(
        f"Inference finished for sensor {sensor_id} | "
        f"Camera — OK: {n_ok}, WARN: {n_warning}, "
        f"CRIT: {n_critical}, SHUT: {n_shutdown}, "
        f"Degraded (WARN+CRIT): {n_degraded_image}, "
        f"Shutdown occurred: {env.camera_check.is_shutdown}"
    )


# ── Config and dataset loading ─────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python inference_main.py <config_file.json>")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, "r") as json_file:
    config = json.load(json_file)

max_offset_per_sensor = config["max_offset_per_sensor"]
batch_size            = config["parallel_batch_size"]

df = prepare_dataset(config)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    file_name = config["file_name"]
    folder    = f"Inference/episode_plots_step_reward{file_name}"
    os.makedirs(folder, exist_ok=True)

    log_file_path = (
        f"{folder}/wildfire_rl_log_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_name}.txt"
    )
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    all_sensors = sorted(df["Sensor"].unique())

    start_offset_per_sensor = {
        sensor: np.random.randint(0, max_offset_per_sensor - 1)
        for sensor in all_sensors
    }

    def run_batch(sensor_list):
        args = [
            (
                sensor,
                df[df["Sensor"] == sensor].copy(),
                start_offset_per_sensor[sensor],
                config
            )
            for sensor in sensor_list
        ]
        with Pool(processes=len(sensor_list)) as pool:
            pool.starmap(run_inference_with_sensor_id, args)

    # ── Run all sensors in parallel batches ────────────────────────────────────
    for i in range(0, len(all_sensors), batch_size):
        current_batch = all_sensors[i:i + batch_size]
        print(f"\nRunning batch {i // batch_size + 1} with {len(current_batch)} sensors...")
        run_batch(current_batch)
        print(f"Batch {i // batch_size + 1} complete.")

    # ── NEW: Aggregate camera summaries across all sensors ─────────────────────
    # Reads back per-sensor camera_summary JSONs → single CSV + fleet report.
    print("\nAggregating camera self-check summaries across all sensors...")
    import pandas as pd

    all_summaries = []
    for sensor in all_sensors:
        path = f"{folder}/camera_summary_{sensor}.json"
        if os.path.exists(path):
            with open(path) as f:
                all_summaries.append(json.load(f))
        else:
            print(f"  Warning: no camera summary found for sensor {sensor}")

    if all_summaries:
        summary_df       = pd.DataFrame(all_summaries)
        summary_csv_path = f"{folder}/camera_summary_all_sensors.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        total_sensors      = len(summary_df)
        shutdown_sensors   = int(summary_df["camera_shutdown_occurred"].sum())
        total_attempts     = int(summary_df["total_picture_attempts"].sum())
        total_ok           = int(summary_df["CAMERA_OK"].sum())
        total_warning      = int(summary_df["FAULT_WARNING"].sum())
        total_critical     = int(summary_df["FAULT_CRITICAL"].sum())
        total_shut         = int(summary_df["SHUT_CAMERA"].sum())
        total_degraded     = int(summary_df["degraded_image_steps"].sum())

        print(f"\n{'='*60}")
        print(f"  CAMERA SELF-CHECK — FLEET SUMMARY")
        print(f"{'='*60}")
        print(f"  Total sensors evaluated      : {total_sensors}")
        print(f"  Sensors with SHUT_CAMERA     : {shutdown_sensors} "
              f"({100 * shutdown_sensors / total_sensors:.1f}%)")
        print(f"  Total picture attempts       : {total_attempts}")
        print(f"  Total CAMERA_OK              : {total_ok}")
        print(f"  Total FAULT_WARNING          : {total_warning}  "
            f"(picture taken, ml_result zeroed)")
        print(f"  Total FAULT_CRITICAL         : {total_critical}  "
            f"(picture taken, ml_result zeroed)")
        print(f"  Total SHUT_CAMERA            : {total_shut}  "
            f"(picture suppressed)")
        print(f"  Total degraded-image steps   : {total_degraded}  "
              f"(WARNING + CRITICAL combined)")
        print(f"  Fleet summary saved          : {summary_csv_path}")
        print(f"{'='*60}\n")