import gymnasium as gym
import numpy as np
import pandas as pd
import json
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gymnasium import spaces
from joblib import load
from pathlib import Path

from data_utils import normalize_feature


class Logger(object):
    def __init__(self, filename):
        import sys
        self.terminal = sys.__stdout__
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ─── Camera Self-Check ────────────────────────────────────────────────────────
class CameraSelfCheck:
    """
    Detects 6 camera fault conditions and escalates through a signal state machine.

    State machine:
    ┌──────────────────┬───────────────────────────────────────────────────────┐
    │ Signal           │ Behaviour in WildfireEnv.step()                       │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ CAMERA_OK        │ take_picture=1, ML runs normally                      │
    │ FAULT_WARNING    │ take_picture=1, image faulty → ml_result forced to 0  │
    │ FAULT_CRITICAL   │ take_picture=1, image faulty → ml_result forced to 0  │
    │ SHUT_CAMERA      │ take_picture=0, camera suppressed for rest of episode  │
    └──────────────────┴───────────────────────────────────────────────────────┘
    """

    # Standard resolution — all frames resized to this before any check
    # Handles mixed camera types / resolutions in the dataset
    TARGET_SIZE = (1280, 960)

    def __init__(self, config):
        cfg = config["Camera_SelfCheck"]

        self.enabled            = cfg["enabled"]
        self.consecutive_limit  = cfg["consecutive_fault_threshold"]
        self.blur_thresh        = cfg["blur_laplacian_threshold"]
        self.noise_thresh       = cfg["noise_laplacian_threshold"]
        self.over_exp_ratio    = cfg.get("over_exposure_pixel_ratio",  0.10)
        self.under_exp_ratio   = cfg.get("under_exposure_pixel_ratio", 0.02)
        self.over_exp_thresh    = cfg["over_exposure_threshold"]
        self.under_exp_thresh   = cfg["under_exposure_threshold"]
        self.pov_corr_thresh    = cfg["pov_correlation_threshold"]

        self.reference_frame    = None
        self.consecutive_faults = 0
        self.is_shutdown        = False
        self.signal             = "CAMERA_OK"
        self.fault_log          = []

    def reset(self):
        """
        Called at the start of every episode.
        Clears per-episode state but keeps reference_frame.
        """
        self.consecutive_faults = 0
        self.is_shutdown        = False
        self.signal             = "CAMERA_OK"
        self.fault_log          = []

    def check(self, frame: np.ndarray, step: int):
        """
        Run all fault detectors on a frame.

        Returns:
            signal    str   "CAMERA_OK" | "FAULT_WARNING" |
                            "FAULT_CRITICAL" | "SHUT_CAMERA"
            faults    list  names of faults detected
            image_ok  bool  True only when signal == "CAMERA_OK"
        """
        if not self.enabled:
            return "CAMERA_OK", [], True

        if self.is_shutdown:
            return "SHUT_CAMERA", ["CAMERA_OFFLINE"], False

        # Normalize resolution — handles mixed camera types
        frame = cv2.resize(frame, self.TARGET_SIZE)

        faults = []
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Fault 2 & 3: Blur / Noise
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < self.blur_thresh:
            faults.append(f"BLUR(lap={lap_var:.1f})")
        elif lap_var > self.noise_thresh:
            faults.append(f"NOISE(lap={lap_var:.1f})")

        # Fault 4: Exposure
        total_pixels = gray.size
        over_ratio   = np.sum(gray > self.over_exp_thresh)  / total_pixels
        under_ratio  = np.sum(gray < self.under_exp_thresh) / total_pixels
        if over_ratio > self.over_exp_ratio:
            faults.append(f"OVEREXPOSED(ratio={over_ratio:.2f})")
        elif under_ratio > self.under_exp_ratio:
            faults.append(f"UNDEREXPOSED(ratio={under_ratio:.2f})")

        # Fault 5: POV Change
        if self.reference_frame is not None:
            pov_score = self._hsv_correlation(frame, self.reference_frame)
            if pov_score < self.pov_corr_thresh:
                faults.append(f"POV_CHANGE(score={pov_score:.2f})")

        # Signal state machine
        if faults:
            self.consecutive_faults += 1
            if self.consecutive_faults >= self.consecutive_limit:
                self.signal      = "SHUT_CAMERA"
                self.is_shutdown = True
            elif self.consecutive_faults >= 2:
                self.signal = "FAULT_CRITICAL"
            else:
                self.signal = "FAULT_WARNING"
        else:
            self.consecutive_faults = 0
            self.signal             = "CAMERA_OK"

        self.fault_log.append((step, self.signal, faults))

        image_ok = (self.signal == "CAMERA_OK")
        return self.signal, faults, image_ok

    def set_reference(self, frame: np.ndarray):
        self.reference_frame = cv2.resize(frame, self.TARGET_SIZE).copy()

    def _hsv_correlation(self, f1: np.ndarray, f2: np.ndarray) -> float:
        score = 0.0
        hsv1  = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
        hsv2  = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
        for ch in range(3):
            h1 = cv2.calcHist([hsv1], [ch], None, [64], [0, 256])
            h2 = cv2.calcHist([hsv2], [ch], None, [64], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            score += cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        return score / 3.0


# ─── Image Dataset Loader ─────────────────────────────────────────────────────
class ImageDatasetLoader:
    """
    Loads real camera images from the Mixed_images dataset to feed into
    CameraSelfCheck during inference.

    Matching strategy:
      Given a sensor timestamp, find the closest image in the dataset
      by filename timestamp. Falls back to a random good image if no
      close match is found.

    In production (real deployed sensor):
      Replace this class with a direct camera capture call.
    """

    def __init__(self, dataset_dir: str, fault_injection_rate: float = 0.20):
        self.dataset_dir = dataset_dir
        self.fault_injection_rate = fault_injection_rate
        self.good_paths  = []
        self.fault_paths = []
        self._load_paths()

    def _load_paths(self):
        good_dir  = os.path.join(self.dataset_dir, "good")
        fault_dir = os.path.join(self.dataset_dir, "fault")

        if os.path.exists(good_dir):
            self.good_paths = [
                str(p) for p in Path(good_dir).glob("*.jpg")
            ]
        if os.path.exists(fault_dir):
            self.fault_paths = [
                str(p) for p in Path(fault_dir).glob("*.jpg")
            ]

        total = len(self.good_paths) + len(self.fault_paths)
        self.all_paths = self.good_paths + self.fault_paths

        print(
            f"[ImageLoader] Loaded {len(self.good_paths)} good + "
            f"{len(self.fault_paths)} fault = {total} total images "
            f"from {self.dataset_dir}"
        )

    def get_frame(self, timestamp=None, label=None):
        """
        Returns a BGR frame for the current step.
        
        Fault injection is independent of fire label — camera hardware
        degrades regardless of whether a fire is occurring.
        
        FAULT_INJECTION_RATE fraction of all picture attempts receive
        a fault image. The rest receive good images.
        Uses timestamp as seed for reproducibility across runs.
        """
        FAULT_INJECTION_RATE = self.fault_injection_rate  # from config, default 0.20

        if self.fault_paths:
            seed = int(timestamp.timestamp()) if timestamp is not None \
                else np.random.randint(0, 10000)
            rng  = np.random.RandomState(seed % (2**31))
            if rng.random() < FAULT_INJECTION_RATE:
                idx   = seed % len(self.fault_paths)
                frame = cv2.imread(self.fault_paths[idx])
                if frame is not None:
                    return frame

        # Serve good image
        pool = self.good_paths or self.all_paths
        if not pool:
            return None
        idx   = int(timestamp.timestamp()) % len(pool) \
                if timestamp is not None else np.random.randint(0, len(pool))
        frame = cv2.imread(pool[idx])
        return frame

# ─── RL Environment ───────────────────────────────────────────────────────────
class WildfireEnv(gym.Env):

    def __init__(self, df, config, start_offset=0):
        super(WildfireEnv, self).__init__()
        self.df           = df
        self.config       = config
        self.start_offset = start_offset

        self.dt_model = load("weather_fire_detection_model.pkl")
        self.sensor_selection_count = {sensor: 0 for sensor in df["Sensor"].unique()}
        self.sensor_data    = None
        self.current_sensor = None
        self.current_step   = 0
        self.last_image_timestamp = None
        self.last_sampling_time   = None

        battery_energy_dict = self.config["Initial_Battery_Levels"]
        battery_choice      = np.random.choice(list(battery_energy_dict.keys()))
        initial_energy      = battery_energy_dict[battery_choice]
        self.battery_energy     = initial_energy
        self.max_battery_energy = initial_energy
        self.energy_budget      = initial_energy - self.config["Energy_Constraints"]["reserved_energy"]

        self.previous_ml_result = 0
        self.missed_fire        = 0
        self.missed_fire_time   = 0

        self.episode_counter   = 0
        self.step_rewards_list = []

        self.total_env_steps   = 0
        self.step_reward_log   = []
        self.step_sampling_log = []

        self.cumulative_rewards_list  = []
        self.average_rewards_list     = []
        self.tensorboard_rewards_list = []
        self.avg_sampling_time_list   = []
        self.detection_time_list      = []

        self.reward      = 0
        self.data_length = 0

        self.fire_start_time        = None
        self.fire_detection_time    = None
        self.battery_depletion_time = None

        # ── Camera Self-Check ─────────────────────────────────────────────────
        self.camera_check = CameraSelfCheck(config)

        # ── Image Dataset Loader ──────────────────────────────────────────────
        # Loads real images from dataset if path is provided in config.
        # Falls back to synthesized frames if path is missing or invalid.
        image_dir = config.get("Camera_SelfCheck", {}).get("image_dataset_dir", "")
        if image_dir and os.path.exists(image_dir):
            fault_rate = config.get("Camera_SelfCheck", {}).get("fault_injection_rate", 0.20)
            self.image_loader = ImageDatasetLoader(image_dir, fault_injection_rate=fault_rate)
            # Set reference frame from first good image for POV detection
            if self.image_loader.good_paths:
                ref = cv2.imread(self.image_loader.good_paths[0])
                if ref is not None:
                    self.camera_check.set_reference(ref)
                    print(f"[CameraCheck] Reference frame set from dataset.")
        else:
            self.image_loader = None
            if image_dir:
                print(
                    f"[CameraCheck] WARNING: image_dataset_dir '{image_dir}' "
                    f"not found. Falling back to synthesized frames."
                )
            else:
                print(
                    f"[CameraCheck] image_dataset_dir not set in config. "
                    f"Using synthesized frames."
                )

        self.episode_data = {
            "timestamps":        [],
            "battery_levels":    [],
            "energy_budgets":    [],
            "missed_fire_times": [],
            "sampling_time":     [],
            "harvested_energy":  [],
            "consumed_energy":   [],
            "temperature":       [],
            "humidity":          [],
            "HDWI_score":        [],
            "wind_speed":        [],
            "ml_result":         [],
            "take_a_picture":    [],
            "label":             [],
            "reward":            [],
            "camera_signal":     [],
            "camera_faults":     [],
        }

        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.action_space      = spaces.Box(
            low=np.array([self.config["TD3_params"]["min_sampling_time"]]),
            high=np.array([self.config["TD3_params"]["max_sampling_time"]]),
            dtype=np.float32
        )

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, sensor_override=None):
        if sensor_override is None:
            print("Sensor override must be provided for sequential execution.")
            return None

        self.current_sensor = sensor_override
        self.sensor_data    = self.df[
            self.df["Sensor"] == self.current_sensor
        ].reset_index(drop=True)
        self.sensor_selection_count[self.current_sensor] += 1

        self.last_sampling_time = 0

        battery_energy_dict = self.config["Initial_Battery_Levels"]
        battery_choice      = np.random.choice(list(battery_energy_dict.keys()))
        initial_energy      = battery_energy_dict[battery_choice]

        self.battery_energy     = initial_energy
        self.max_battery_energy = initial_energy
        self.energy_budget      = initial_energy - self.config["Energy_Constraints"]["reserved_energy"]
        self.previous_ml_result = 0
        self.missed_fire        = 0
        self.missed_fire_time   = 0
        self.step_rewards_list  = []
        self.reward             = 0

        fire_rows            = self.sensor_data[self.sensor_data["Label"] == 1]
        self.fire_start_time = fire_rows["Timestamp"].iloc[0] if not fire_rows.empty else None

        if not fire_rows.empty:
            fire_start_time = fire_rows["Timestamp"].iloc[0]
            allowed_indices = self.sensor_data[
                self.sensor_data["Timestamp"] <= (fire_start_time - pd.Timedelta(days=7))
            ].index
        else:
            allowed_indices = self.sensor_data.index

        if len(self.sensor_data) <= self.start_offset:
            self.current_step = 0
        else:
            self.current_step = 0

        self.last_image_timestamp   = self.sensor_data["Timestamp"].iloc[self.current_step]
        self.data_length            = max(1, len(self.sensor_data) - (self.current_step + 1))
        self.fire_detection_time    = None
        self.battery_depletion_time = None

        # Reset camera self-check for new episode
        self.camera_check.reset()

        # Re-set reference frame for new episode if loader is available
        if self.image_loader and self.image_loader.good_paths:
            ref = cv2.imread(self.image_loader.good_paths[0])
            if ref is not None:
                self.camera_check.set_reference(ref)

        self.episode_data = {
            "timestamps":        [],
            "battery_levels":    [],
            "energy_budgets":    [],
            "missed_fire_times": [],
            "sampling_time":     [],
            "harvested_energy":  [],
            "consumed_energy":   [],
            "temperature":       [],
            "humidity":          [],
            "HDWI_score":        [],
            "wind_speed":        [],
            "ml_result":         [],
            "take_a_picture":    [],
            "label":             [],
            "reward":            [],
            "camera_signal":     [],
            "camera_faults":     [],
        }

        return self.get_state()

    # ─────────────────────────────────────────────────────────────────────────
    def get_state(self):
        row = self.sensor_data.iloc[self.current_step]
        time_since_last_image = (
            row["Timestamp"] - self.last_image_timestamp
        ).total_seconds() / 60

        return np.array([
            row["Temperature_2m_normalized"],
            row["Relative_Humidity_2m_normalized"],
            row["Wind_Speed_10m"],
            row["HDWI"],
            row["solar_energy"],
            normalize_feature(
                self.last_sampling_time, 1,
                self.config["TD3_params"]["max_sampling_time"]
            ),
            normalize_feature(
                self.energy_budget, 0,
                self.max_battery_energy - self.config["Energy_Constraints"]["reserved_energy"]
            ),
            normalize_feature(time_since_last_image, 0, 120),
            row["Time_of_Day"],
            row["Season"],
            self.previous_ml_result,
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_frame(self, row) -> np.ndarray:
        """
        Returns a BGR frame for the current step.

        Priority:
          1. Real image from ImageDatasetLoader (if configured and available)
          2. Synthesized frame from sensor data (fallback)

        ── PRODUCTION NOTE ──────────────────────────────────────────────────
        In a real deployed sensor node, replace this entire method with:
            return capture_image_from_camera()
        ─────────────────────────────────────────────────────────────────────
        """
        if self.image_loader is not None:
            frame = self.image_loader.get_frame(
                timestamp=row["Timestamp"],
                label=int(row["Label"])
            )
            if frame is not None:
                return frame
            # Frame load failed — fall through to synthesized

        return self._synthesize_frame(row)

    def _synthesize_frame(self, row) -> np.ndarray:
        """
        Fallback: builds a synthetic BGR frame from sensor readings.
        Used when no image dataset is configured or image load fails.
        """
        brightness = int(np.clip(row["Temperature_2m_normalized"] * 200 + 30, 0, 255))
        frame      = np.full((960, 1280, 3), brightness, dtype=np.uint8)
        frame[:, :, 0] = np.clip(brightness + int(row["Time_of_Day"] * 20), 0, 255)
        frame[:, :, 2] = np.clip(brightness - int(row["HDWI"] * 30),        0, 255)
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        self.last_sampling_time = int(action)

        next_timestamp = (
            self.sensor_data.iloc[self.current_step]["Timestamp"]
            + pd.Timedelta(minutes=self.last_sampling_time)
        )

        next_steps   = self.sensor_data[
            self.sensor_data["Timestamp"] >= next_timestamp
        ].index
        skipped_data = self.sensor_data[
            (self.sensor_data["Timestamp"] > self.sensor_data.iloc[self.current_step]["Timestamp"]) &
            (self.sensor_data["Timestamp"] < next_timestamp)
        ]

        if not next_steps.empty:
            self.current_step = next_steps[0]
        else:
            self.current_step = len(self.sensor_data) - 1

        row = self.sensor_data.iloc[self.current_step]

        features    = {
            "avgtempC": row["Temperature_2m"],
            "humid":    row["Relative_Humidity_2m"]
        }
        df_features = pd.DataFrame([features])

        # ── Step 1: DT model decides whether to take a picture ────────────────
        take_picture = int(self.dt_model.predict(df_features)[0])

        # ── Step 2: Camera Self-Check ─────────────────────────────────────────
        camera_signal = "CAMERA_OK"
        camera_faults = []
        image_ok      = True

        if take_picture:
            # Get real image from dataset (or synthesized fallback)
            frame = self._get_frame(row)

            camera_signal, camera_faults, image_ok = self.camera_check.check(
                frame, self.current_step
            )

            if camera_signal == "SHUT_CAMERA":
                # Hardware failure — suppress picture entirely
                take_picture = 0
                print(
                    f"[CameraCheck] SHUT_CAMERA at step {self.current_step} | "
                    f"Faults: {camera_faults} | Camera suppressed."
                )
            elif not image_ok:
                # WARNING or CRITICAL — picture taken but image too poor for ML
                print(
                    f"[CameraCheck] {camera_signal} at step {self.current_step} | "
                    f"Faults: {camera_faults} | ml_result forced to 0."
                )

        fire_rows = pd.concat([skipped_data, self.sensor_data.iloc[[self.current_step]]])

        # ── Step 3: ML fire detection ─────────────────────────────────────────
        ml_result = 0

        if take_picture:
            if image_ok:
                # Camera healthy — run stochastic TP/FP simulation
                ml_result = (
                    np.random.choice(
                        [1, 0],
                        p=[
                            self.config["ML_Performance"]["TP_rate"],
                            1 - self.config["ML_Performance"]["TP_rate"]
                        ]
                    ) if row["Label"] == 1 else
                    np.random.choice(
                        [1, 0],
                        p=[
                            self.config["ML_Performance"]["FP_rate"],
                            1 - self.config["ML_Performance"]["FP_rate"]
                        ]
                    )
                )
            else:
                # Degraded image — cannot trust ML result
                ml_result = 0

            if row["Label"] == 1 and ml_result == 1:
                self.fire_detection_time = row["Timestamp"]

            self.last_image_timestamp = row["Timestamp"]

        neighbor_comm_energy = (
            self.config["Neighborhood_Communication"]["num_neighbors"] *
            self.config["Neighborhood_Communication"]["E_comm_neighbor"]
        )

        if not skipped_data.empty and "Label" in skipped_data.columns:
            self.missed_fire = any(
                (skipped_data["Label"] == 1) &
                (
                    (row["Timestamp"] - skipped_data["Timestamp"]).dt.total_seconds() / 60
                    > self.config["max_missing_fire_min"]
                )
            )

        if row["Label"] == 1:
            self.missed_fire_time = (
                row["Timestamp"] - self.fire_start_time
            ).total_seconds() / 60

        total_harvested_energy = (
            (fire_rows["solar_energy"] * self.config["harvested_energy_loss"]).sum()
            if not fire_rows.empty else 0
        )

        time_skipped_hours = self.last_sampling_time / 60
        standby_power_used = (
            self.config["Standby_Power_Components"]["P_temp_humidity_standby"] +
            self.config["Standby_Power_Components"]["P_anemometer_standby"]    +
            self.config["Standby_Power_Components"]["P_camera_standby"]        +
            self.config["Standby_Power_Components"]["P_comm_standby"]
        ) * time_skipped_hours

        # ── Step 4: Energy accounting ─────────────────────────────────────────
        energy_used = (
            self.config["Energy_Constraints"]["E_proc_rl"]             +
            self.config["Energy_Constraints"]["E_temp_humidity_sensor"] +
            self.config["Energy_Constraints"]["E_anemometer_sensor"]    +
            standby_power_used                                          +
            (
                self.config["Energy_Constraints"]["E_proc_ml"]          +
                self.config["Energy_Constraints"]["E_camera_host"]      +
                self.config["Energy_Constraints"]["E_camera_selfcheck"]
                if take_picture else 0
            )                                                           +
            (
                self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy
                if ml_result and take_picture else 0
            )
        )

        if self.battery_depletion_time is None:
            current_batt    = self.battery_energy
            max_batt_energy = self.max_battery_energy
            E_used = (
                self.config["Energy_Constraints"]["E_proc_rl"]              +
                self.config["Energy_Constraints"]["E_temp_humidity_sensor"] +
                self.config["Energy_Constraints"]["E_anemometer_sensor"]    +
                (
                    self.config["Energy_Constraints"]["E_proc_ml"]          +
                    self.config["Energy_Constraints"]["E_camera_host"]      +
                    self.config["Energy_Constraints"]["E_camera_selfcheck"]
                    if take_picture else 0
                )                                                           +
                (
                    self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy
                    if ml_result and take_picture else 0
                )
            )
            for i, ts in enumerate(fire_rows["Timestamp"]):
                harvested    = fire_rows.iloc[i]["solar_energy"] * self.config["harvested_energy_loss"]
                standby_used = standby_power_used / len(fire_rows)
                net_energy   = harvested - standby_used

                current_batt    += net_energy
                current_batt    *= (1 - self.config["E_battery_leakage_percentage"])
                max_batt_energy *= (1 - self.config["E_battery_leakage_percentage"])
                current_batt     = max(0, min(max_batt_energy, current_batt))

                if current_batt - self.config["Energy_Constraints"]["reserved_energy"] <= 0:
                    self.battery_depletion_time = ts
                    break

            if (
                current_batt - self.config["Energy_Constraints"]["reserved_energy"] > 0 and
                current_batt - E_used - self.config["Energy_Constraints"]["reserved_energy"] <= 0
            ):
                self.battery_depletion_time = row["Timestamp"]

        self.battery_energy     += total_harvested_energy - energy_used
        self.battery_energy     *= (1 - self.config["E_battery_leakage_percentage"])
        self.max_battery_energy *= (1 - self.config["E_battery_leakage_percentage"])
        self.battery_energy      = max(0, min(self.max_battery_energy, self.battery_energy))
        self.energy_budget       = max(
            0, self.battery_energy - self.config["Energy_Constraints"]["reserved_energy"]
        )

        self.previous_ml_result = ml_result

        done = self.current_step >= len(self.sensor_data) - 1

        if (
            (row["Label"] == 1 and ml_result == 1) or
            self.energy_budget <= 0 or
            self.battery_depletion_time is not None
        ):
            print(
                f'Next sensor time: {self.last_sampling_time}, '
                f'Stopping Condition Met: '
                f'Fire Detected: {row["Label"] == 1 and ml_result == 1}, '
                f'Missed Fire for {self.missed_fire_time} min: {self.missed_fire_time >= 30}, '
                f'Energy Depleted: {self.energy_budget <= 0}, '
                f'Battery Depletion Time: {self.battery_depletion_time}'
            )
            self.detection_time_list.append(self.missed_fire_time)
            done = True

        # Store episode data
        self.episode_data["timestamps"].append(row["Timestamp"])
        self.episode_data["battery_levels"].append(self.battery_energy)
        self.episode_data["energy_budgets"].append(self.energy_budget)
        self.episode_data["missed_fire_times"].append(self.missed_fire_time)
        self.episode_data["sampling_time"].append(self.last_sampling_time)
        self.episode_data["harvested_energy"].append(row["solar_energy"])
        self.episode_data["consumed_energy"].append(energy_used)
        self.episode_data["temperature"].append(row["Temperature_2m_normalized"])
        self.episode_data["humidity"].append(row["Relative_Humidity_2m_normalized"])
        self.episode_data["HDWI_score"].append(row["HDWI"])
        self.episode_data["wind_speed"].append(row["Wind_Speed_10m"])
        self.episode_data["ml_result"].append(ml_result)
        self.episode_data["take_a_picture"].append(take_picture)
        self.episode_data["label"].append(row["Label"])
        self.episode_data["camera_signal"].append(camera_signal)
        self.episode_data["camera_faults"].append(
            "; ".join(camera_faults) if camera_faults else "none"
        )

        self.reward = (
            self.config["Reward_Params"]["beta"] * self.last_sampling_time +
            (1 - self.config["Reward_Params"]["beta"]) * self.reward
        )

        step_reward = self.last_sampling_time * (1 - 2 * take_picture) / self.data_length

        self.episode_data["reward"].append(step_reward)
        self.step_sampling_log.append((self.total_env_steps, self.last_sampling_time))

        if done:
            self.episode_counter += 1
            avg_sampling_time    = np.mean(self.episode_data["sampling_time"])
            final_reward, reason = self.calculate_final_reward()
            print("final_reward", final_reward)
            self.step_rewards_list.append(final_reward)
            self.step_reward_log.append((self.total_env_steps, final_reward))
            self.total_env_steps += 1
            self.average_rewards_list.append(
                sum(self.step_rewards_list) / len(self.step_rewards_list)
            )
            self.cumulative_rewards_list.append(sum(self.step_rewards_list))
            self.tensorboard_rewards_list.append(
                sum(self.cumulative_rewards_list) / len(self.cumulative_rewards_list)
            )
            self.avg_sampling_time_list.append(avg_sampling_time)
            print("avg_sampling_time", avg_sampling_time)

            self.plot_episode_metrics(reason, final_reward)
            return self.get_state(), final_reward, done, {}

        self.step_rewards_list.append(step_reward)
        self.step_reward_log.append((self.total_env_steps, step_reward))
        self.total_env_steps += 1

        return self.get_state(), step_reward, done, {}

    # ─────────────────────────────────────────────────────────────────────────
    def calculate_final_reward(self):
        if self.battery_depletion_time is not None and (
            not self.fire_start_time or
            self.battery_depletion_time < self.fire_start_time
        ):
            t_deplete_minus_t_start = (
                self.battery_depletion_time - self.sensor_data["Timestamp"].iloc[0]
            ).total_seconds() / 60
            reward = (
                -self.config["Reward_Params"]["alpha_B"] * (1 / t_deplete_minus_t_start)
                - self.config["Reward_Params"]["R_min"]
            )
            reason = "case1"
            print('reward case1 ', reward, "t_deplete_minus_t_start", t_deplete_minus_t_start)
        else:
            reward = -self.config["Reward_Params"]["k1"] * self.reward
            reason = "case2/3"
            print('reward case2/3 ', reward)

        return reward, reason

    # ─────────────────────────────────────────────────────────────────────────
    def plot_episode_metrics(self, reason, final_reward):
        file_name = self.config["file_name"]
        folder    = f"Inference/episode_plots_step_reward{file_name}"
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "config_setup.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        df2 = pd.DataFrame(self.episode_data)
        df2.to_csv(
            f"{folder}/episode_{self.episode_counter}_{self.current_sensor}.csv",
            index=False
        )

        fig, axs = plt.subplots(7, 2, figsize=(15, 20), sharex=True)
        axs = axs.flatten()

        axs[0].scatter(self.episode_data["timestamps"],  self.episode_data["harvested_energy"],   label="Harvested Energy (Wh)",  color='green')
        axs[1].scatter(self.episode_data["timestamps"],  self.episode_data["consumed_energy"],    label="Consumed Energy (Wh)",   color='green')
        axs[2].scatter(self.episode_data["timestamps"],  self.episode_data["energy_budgets"],     label="Energy Budget (Wh)",     color='green')
        axs[3].scatter(self.episode_data["timestamps"],  self.episode_data["battery_levels"],     label="Battery Energy (Wh)",    color='green')
        axs[4].scatter(self.episode_data["timestamps"],  self.episode_data["temperature"],        label="Temperature (C)",        color='green')
        axs[5].scatter(self.episode_data["timestamps"],  self.episode_data["humidity"],           label="Humidity (%)",           color='green')
        axs[6].scatter(self.episode_data["timestamps"],  self.episode_data["HDWI_score"],         label="HDWI score",             color='green')
        axs[7].scatter(self.episode_data["timestamps"],  self.episode_data["wind_speed"],         label="Wind Speed (km/h)",      color='green')
        axs[8].scatter(self.episode_data["timestamps"],  self.episode_data["sampling_time"],      label="Sampling Time (min)",    color='green')
        axs[9].scatter(self.episode_data["timestamps"],  self.episode_data["take_a_picture"],     label="Take a picture [0, 1]",  color='green')
        axs[10].scatter(self.episode_data["timestamps"], self.episode_data["ml_result"],          label="ML result [0, 1]",       color='green')
        axs[11].scatter(self.episode_data["timestamps"], self.episode_data["missed_fire_times"],  label="Missed Fire Time (min)", color='green')
        axs[12].scatter(self.episode_data["timestamps"], self.episode_data["reward"],
                        label=f"Step Reward {reason} {final_reward}",                            color='green')

        # Camera signal panel
        signal_map   = {"CAMERA_OK": 0, "FAULT_WARNING": 1, "FAULT_CRITICAL": 2, "SHUT_CAMERA": 3}
        signal_nums  = [signal_map.get(s, 0) for s in self.episode_data["camera_signal"]]
        point_colors = [
            "green" if s == 0 else
            "orange" if s == 1 else
            "darkorange" if s == 2 else
            "red"
            for s in signal_nums
        ]
        axs[13].scatter(
            self.episode_data["timestamps"], signal_nums,
            c=point_colors, label="Camera Signal"
        )
        axs[13].set_yticks([0, 1, 2, 3])
        axs[13].set_yticklabels(["OK", "WARN", "CRIT", "SHUT"])
        axs[13].set_ylabel("Camera State")

        axs[-1].set_xlabel("Timestamp (min)", fontsize=16, fontweight='bold')

        if self.fire_start_time is not None:
            for ax in axs:
                ax.axvline(
                    x=self.fire_start_time, color='red',
                    linestyle='--', label='First Fire Detected'
                )

        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y\n%I:%M %p'))
            ax.tick_params(axis='x', labelsize=12, width=2, rotation=45)
            ax.tick_params(axis='y', labelsize=12, width=2)
            ax.legend()
            ax.grid(True)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{folder}/episode_{self.episode_counter}_{self.current_sensor}.png",
            dpi=300
        )
        plt.close()

        print(f"Episode {self.episode_counter} plot saved.")