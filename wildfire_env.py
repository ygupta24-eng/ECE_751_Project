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

    State machine (mirrors notebook CONSECUTIVE_THRESH logic):
    ┌──────────────────┬───────────────────────────────────────────────────────┐
    │ Signal           │ Behaviour in WildfireEnv.step()                       │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ CAMERA_OK        │ take_picture=1, ML runs normally                      │
    │ FAULT_WARNING    │ take_picture=1 (picture attempted), but image is      │
    │                  │ flagged as faulty → ml_result forced to 0             │
    │ FAULT_CRITICAL   │ take_picture=1 (picture attempted), but image is      │
    │                  │ flagged as faulty → ml_result forced to 0             │
    │ SHUT_CAMERA      │ take_picture=0 (camera fully suppressed),             │
    │                  │ no energy for camera/ML, locked for rest of episode   │
    └──────────────────┴───────────────────────────────────────────────────────┘

    Key distinction from previous version:
      - WARNING / CRITICAL do NOT suppress take_picture.
        The camera still fires — but the image quality is too poor for
        reliable fire detection, so ml_result is zeroed out.
      - Only SHUT_CAMERA suppresses take_picture entirely.

    Fault detectors:
      1. Black Frame        — mean brightness below threshold
      2. Blur               — Laplacian variance below threshold
      3. Noise              — Laplacian variance above threshold
      4. Overexposure /
         Underexposure      — saturated/dark pixel ratio above threshold
      5. Frozen Frame       — mean pixel diff vs previous frame below threshold
      6. POV Change         — HSV histogram correlation vs reference below threshold
    """

    def __init__(self, config):
        cfg = config["Camera_SelfCheck"]

        # All thresholds driven from config — no magic numbers in code
        self.enabled            = cfg["enabled"]
        self.consecutive_limit  = cfg["consecutive_fault_threshold"]
        self.black_thresh       = cfg["black_brightness_threshold"]
        self.blur_thresh        = cfg["blur_laplacian_threshold"]
        self.noise_thresh       = cfg["noise_laplacian_threshold"]
        self.exposure_ratio     = cfg["exposure_pixel_ratio"]
        self.over_exp_thresh    = cfg["over_exposure_threshold"]
        self.under_exp_thresh   = cfg["under_exposure_threshold"]
        self.frozen_diff_thresh = cfg["frozen_diff_threshold"]
        self.pov_corr_thresh    = cfg["pov_correlation_threshold"]

        # Runtime state
        self.prev_frame         = None
        self.reference_frame    = None
        self.consecutive_faults = 0
        self.is_shutdown        = False
        self.signal             = "CAMERA_OK"
        self.fault_log          = []   # list of (step, signal, [faults])

    def reset(self):
        """
        Called at the start of every episode in WildfireEnv.reset().
        Clears per-episode state but keeps reference_frame — the physical
        camera does not change between episodes.
        """
        self.prev_frame         = None
        self.consecutive_faults = 0
        self.is_shutdown        = False
        self.signal             = "CAMERA_OK"
        self.fault_log          = []

    def check(self, frame: np.ndarray, step: int):
        """
        Run all fault detectors and update the signal state machine.

        Called from WildfireEnv.step() every time take_picture_attempted == 1.

        Args:
            frame : BGR numpy array (real captured image or synthesized frame)
            step  : current env step index — stored in fault_log for traceability

        Returns:
            signal      str   "CAMERA_OK" | "FAULT_WARNING" |
                              "FAULT_CRITICAL" | "SHUT_CAMERA"
            faults      list  names of all faults detected this frame
            image_ok    bool  True only when signal == "CAMERA_OK"
                              (False means image quality is too poor for ML —
                               but does NOT mean take_picture is suppressed,
                               unless signal == "SHUT_CAMERA")
        """
        if not self.enabled:
            return "CAMERA_OK", [], True

        # Once shutdown, camera stays offline for the rest of the episode
        if self.is_shutdown:
            return "SHUT_CAMERA", ["CAMERA_OFFLINE"], False

        faults = []
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Fault 1: Black Frame ──────────────────────────────────────────────
        mean_brightness = np.mean(gray)
        if mean_brightness < self.black_thresh:
            faults.append(f"BLACK_FRAME(brightness={mean_brightness:.1f})")

        # ── Fault 2 & 3: Blur / Noise (same metric, mutually exclusive) ───────
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < self.blur_thresh:
            faults.append(f"BLUR(lap={lap_var:.1f})")
        elif lap_var > self.noise_thresh:
            faults.append(f"NOISE(lap={lap_var:.1f})")

        # ── Fault 4: Exposure ─────────────────────────────────────────────────
        total_pixels = gray.size
        over_ratio   = np.sum(gray > self.over_exp_thresh)  / total_pixels
        under_ratio  = np.sum(gray < self.under_exp_thresh) / total_pixels
        if over_ratio > self.exposure_ratio:
            faults.append(f"OVEREXPOSED(ratio={over_ratio:.2f})")
        elif under_ratio > self.exposure_ratio:
            faults.append(f"UNDEREXPOSED(ratio={under_ratio:.2f})")

        # ── Fault 5: Frozen Frame ─────────────────────────────────────────────
        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(
                self.prev_frame, cv2.COLOR_BGR2GRAY
            ).astype(np.float32)
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray))
            if diff < self.frozen_diff_thresh:
                faults.append(f"FROZEN_FRAME(diff={diff:.2f})")

        # ── Fault 6: POV Change ───────────────────────────────────────────────
        if self.reference_frame is not None:
            pov_score = self._hsv_correlation(frame, self.reference_frame)
            if pov_score < self.pov_corr_thresh:
                faults.append(f"POV_CHANGE(score={pov_score:.2f})")

        # Always update prev_frame for next frozen-frame check
        self.prev_frame = frame.copy()

        # ── Signal state machine (mirrors notebook CONSECUTIVE_THRESH logic) ──
        if faults:
            self.consecutive_faults += 1
            if self.consecutive_faults >= self.consecutive_limit:
                # 3+ consecutive faults → camera is hardware-failed, shut down
                self.signal      = "SHUT_CAMERA"
                self.is_shutdown = True
            elif self.consecutive_faults >= 2:
                self.signal = "FAULT_CRITICAL"
            else:
                self.signal = "FAULT_WARNING"
        else:
            # Fault cleared → reset consecutive counter
            self.consecutive_faults = 0
            self.signal             = "CAMERA_OK"

        self.fault_log.append((step, self.signal, faults))

        # image_ok is True only on CAMERA_OK — used by step() to decide
        # whether to trust the ML result from this frame
        image_ok = (self.signal == "CAMERA_OK")
        return self.signal, faults, image_ok

    def set_reference(self, frame: np.ndarray):
        """Store a clean reference frame for POV-change detection."""
        self.reference_frame = frame.copy()

    def _hsv_correlation(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """
        Average Pearson correlation of HSV histograms across all 3 channels.
        Returns value in [0, 1]. Score below pov_corr_thresh = scene changed.
        """
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

        # ── Camera Self-Check instance ────────────────────────────────────────
        self.camera_check = CameraSelfCheck(config)

        # Episode data — includes two new camera health columns
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
            "camera_signal":     [],   # NEW
            "camera_faults":     [],   # NEW
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

        # ── Reset camera self-check state for new episode ─────────────────────
        self.camera_check.reset()

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
            "camera_signal":     [],   # NEW
            "camera_faults":     [],   # NEW
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
    def _synthesize_frame(self, row) -> np.ndarray:
        """
        Builds a synthetic BGR frame from sensor readings for simulation mode.

        ── PRODUCTION NOTE ──────────────────────────────────────────────────
        Replace this method body with your real image capture call, e.g.:
            return capture_image_from_camera()
        Everything else in step() stays identical.
        ─────────────────────────────────────────────────────────────────────

        Encoding:
          Base brightness  ← normalized temperature
          Blue channel     ← time-of-day offset  (avoids greyscale flatness)
          Red channel      ← HDWI fire-risk offset
          Small random noise ensures frozen-frame detector does not
          false-positive on consecutive identical sensor readings.
        """
        brightness = int(np.clip(row["Temperature_2m_normalized"] * 200 + 30, 0, 255))
        frame      = np.full((480, 640, 3), brightness, dtype=np.uint8)
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

        # ── Step 1: DT model decides whether weather warrants a picture ───────
        take_picture = int(self.dt_model.predict(df_features)[0])

        # ── Step 2: Camera Self-Check state machine ───────────────────────────
        #
        # The state machine runs ONLY when take_picture == 1 (i.e. the DT
        # model already decided a picture should be taken).
        #
        # Corrected behaviour:
        #
        #   CAMERA_OK      → picture taken, ML runs normally
        #
        #   FAULT_WARNING  → picture IS taken (camera still physically fires),
        #   FAULT_CRITICAL   but image quality is flagged as too poor to trust.
        #                    ml_result is forced to 0 — fire cannot be detected
        #                    from a degraded image. Energy for camera IS charged
        #                    because the hardware still activated.
        #
        #   SHUT_CAMERA    → camera is hardware-failed (3+ consecutive faults).
        #                    take_picture is set to 0 — camera does NOT fire.
        #                    No camera/ML energy charged. Locked for episode.
        #
        # image_ok is the flag that separates WARNING/CRITICAL from OK:
        #   True  → trust the ML result
        #   False → zero out ml_result (degraded image, unreliable inference)
        # ─────────────────────────────────────────────────────────────────────
        camera_signal   = "CAMERA_OK"
        camera_faults   = []
        image_ok        = True          # True → ML result is trustworthy

        if take_picture:
            sim_frame = self._synthesize_frame(row)   # swap for real frame in production
            camera_signal, camera_faults, image_ok = self.camera_check.check(
                sim_frame, self.current_step
            )

            if camera_signal == "SHUT_CAMERA":
                # Hardware failure — suppress the picture entirely
                take_picture = 0
                print(
                    f"[CameraCheck] SHUT_CAMERA at step {self.current_step} | "
                    f"Faults: {camera_faults} | Camera suppressed for rest of episode."
                )
            else:
                # WARNING or CRITICAL: picture was taken but image is faulty.
                # Log the degraded state without suppressing take_picture.
                if not image_ok:
                    print(
                        f"[CameraCheck] {camera_signal} at step {self.current_step} | "
                        f"Faults: {camera_faults} | "
                        f"Picture taken but image quality too poor for ML — "
                        f"ml_result forced to 0."
                    )
        # ─────────────────────────────────────────────────────────────────────

        fire_rows = pd.concat([skipped_data, self.sensor_data.iloc[[self.current_step]]])

        # ── Step 3: ML fire detection ─────────────────────────────────────────
        ml_result = 0

        if take_picture:
            if image_ok:
                # Camera is healthy — run stochastic TP/FP simulation normally
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
                # FAULT_WARNING or FAULT_CRITICAL:
                # Image was captured but quality is degraded — cannot trust
                # the ML inference output. Force ml_result = 0.
                ml_result = 0

            if row["Label"] == 1 and ml_result == 1:
                self.fire_detection_time = row["Timestamp"]

            self.last_image_timestamp = row["Timestamp"]
        # ─────────────────────────────────────────────────────────────────────

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
        #
        # E_camera_selfcheck: charged when take_picture_attempted == 1,
        #   even if the check returned WARNING/CRITICAL — the processor ran.
        #   NOT charged when SHUT_CAMERA suppressed take_picture to 0.
        #
        # E_proc_ml + E_camera_host: charged when take_picture == 1 after the
        #   state machine. This means:
        #   - CAMERA_OK      → charged (picture taken, ML ran)
        #   - FAULT_WARNING  → charged (picture taken, ML ran but result zeroed)
        #   - FAULT_CRITICAL → charged (picture taken, ML ran but result zeroed)
        #   - SHUT_CAMERA    → NOT charged (take_picture was set to 0)
        #
        # E_comm: charged only when ml_result == 1 AND take_picture == 1,
        #   which naturally cannot happen when image_ok is False (ml_result=0).
        # ─────────────────────────────────────────────────────────────────────
        energy_used = (
            self.config["Energy_Constraints"]["E_proc_rl"]             +
            self.config["Energy_Constraints"]["E_temp_humidity_sensor"] +
            self.config["Energy_Constraints"]["E_anemometer_sensor"]    +
            standby_power_used                                          +
            (
                self.config["Energy_Constraints"]["E_proc_ml"] +
                self.config["Energy_Constraints"]["E_camera_host"] +
                self.config["Energy_Constraints"]["E_camera_selfcheck"]
                if take_picture else 0
            )                                                           +
            (
                self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy
                if ml_result and take_picture else 0
            )
        )

        # Minute-by-minute battery depletion simulation over skipped time
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

        # ── Store episode data ────────────────────────────────────────────────
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
        self.episode_data["camera_signal"].append(camera_signal)      # NEW
        self.episode_data["camera_faults"].append(                     # NEW
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
        """Calculate the reward at the end of the episode."""
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
        """Generates and saves episode-specific plots inside episode_plots/."""
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
                        label=f"Step Reward {reason} {final_reward}",                                                             color='green')

        # ── NEW: Camera signal panel (axs[13] — was unused in original 7×2 grid)
        signal_map    = {"CAMERA_OK": 0, "FAULT_WARNING": 1, "FAULT_CRITICAL": 2, "SHUT_CAMERA": 3}
        signal_nums   = [signal_map.get(s, 0) for s in self.episode_data["camera_signal"]]
        point_colors  = [
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