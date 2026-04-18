import gym
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gym import spaces
from joblib import load

from data_utils import normalize_feature

class TrustFactorAlgorithm:
    def __init__(self, params):
        self.window_size = params.get("window_size", 60)
        self.ema_alpha = params.get("ema_alpha", 0.04)
        self.min_val = params.get("min_threshold", 0)
        self.max_val = params.get("max_threshold", 100)
        self.penalty_bounds = params.get("penalty_out_of_bounds", 50)
        self.penalty_flatline = params.get("penalty_flatline", 30)
        self.z_score_thresh = params.get("z_score_threshold", 3)
        self.penalty_z_score = params.get("penalty_high_z_score", 10)
        self.healthy_thresh = params.get("state_threshold_healthy", 90)
        self.warning_thresh = params.get("state_threshold_warning", 70)

        self.history = []
        self.ema_trust = 100.0

    def process_reading(self, value):
        penalty = 0
        z_score = 0.0
        
        # Phase 1: Ingestion & Fast Filters
        if value < self.min_val or value > self.max_val:
            penalty = self.penalty_bounds
        else:
            # Phase 2: Statistical Filters (Calculate stats BEFORE adding the new value)
            if len(self.history) == self.window_size:
                variance = np.var(self.history)
                if variance == 0: # Flatline check
                    penalty = self.penalty_flatline
                else: # Z-score check
                    std_dev = np.sqrt(variance)
                    mean_val = np.mean(self.history)
                    z_score = abs(value - mean_val) / std_dev

                    # Lightweight Statistical Anomaly Trigger
                    if z_score > self.z_score_thresh:
                        penalty = self.penalty_z_score

        # Keep a rolling history for the NEXT reading
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Phase 3: Continuous Trust Decay
        target_trust = max(0, self.ema_trust - penalty) 
        
        # If the sensor is behaving normally, set the target back to 100
        if penalty == 0:
            target_trust = 100
            
        # Apply exponential moving average
        self.ema_trust = (self.ema_alpha * target_trust) + ((1 - self.ema_alpha) * self.ema_trust)

        # Evaluate States
        if self.ema_trust >= self.healthy_thresh:
            state = "Healthy"
        elif self.ema_trust >= self.warning_thresh:
            state = "Warning"
        else:
            state = "Critical"

        return self.ema_trust, state, z_score
    
    def reset(self):
        self.history = []
        self.ema_trust = 100.0


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

# Define RL Environment
class WildfireEnv(gym.Env):

    def __init__(self, df, config, start_offset=0, shared_states=None):
        super(WildfireEnv, self).__init__()
        self.df = df  # This df is now for a SINGLE sensor
        self.config = config
        self.start_offset = start_offset

        # Use the managed dictionary passed from the main process
        if shared_states is not None:
            self.sensor_shared_states = shared_states
            # --- Change: Get all sensor IDs from the shared state ---
            all_sensor_ids = list(shared_states.keys())
        else:
            # Fallback for non-parallel execution
            self.sensor_shared_states = {sensor_id: 0 for sensor_id in df["Sensor"].unique()}
            all_sensor_ids = df["Sensor"].unique()

        # Read trust factor params from config and initialize algorithms for each metric
        trust_factor_configs = self.config.get("TrustFactor", {})
        self.temp_trust_algo = TrustFactorAlgorithm(trust_factor_configs.get("Temperature", {}))
        self.humidity_trust_algo = TrustFactorAlgorithm(trust_factor_configs.get("Humidity", {}))
        self.wind_trust_algo = TrustFactorAlgorithm(trust_factor_configs.get("Wind", {}))

        self.temp_trust_score = 100.0
        self.humidity_trust_score = 100.0
        self.wind_trust_score = 100.0

        self.dt_model = load("weather_fire_detection_model.pkl")
        self.sensor_selection_count = {sensor: 0 for sensor in all_sensor_ids} # Change: Initialize count for ALL sensors
        self.sensor_data = None  # Will be set in reset()
        self.current_sensor = None  # Track the current sensor
        self.current_step = 0
        self.last_image_timestamp = None
        self.last_sampling_time = None
        # Randomly choose one of the battery levels each episode
        battery_energy_dict = self.config["Initial_Battery_Levels"]
        battery_choice = np.random.choice(list(battery_energy_dict.keys()))
        initial_energy = battery_energy_dict[battery_choice]
        self.battery_energy = initial_energy  # Start with full battery capacity
        self.max_battery_energy = initial_energy  # Full battery capacity
        self.energy_budget = initial_energy - self.config["Energy_Constraints"]["reserved_energy"]
        self.previous_ml_result = 0  # No fire detected at start
        self.missed_fire = 0
        self.missed_fire_time = 0  # Initialize missed fire time tracking
        
        self.episode_counter = 0  # Track episode number
        
        self.step_rewards_list = []
        
        self.total_env_steps = 0  # Track global step count for logging
        self.step_reward_log = []  # Store (step, reward)
        self.step_sampling_log = []  # Store (step, sampling)

        self.cumulative_rewards_list = []
        self.average_rewards_list = []
        self.tensorboard_rewards_list = []
        self.avg_sampling_time_list = []
        self.detection_time_list = []
        
        
        self.reward = 0
        self.data_length = 0
        
        self.fire_start_time = None
        self.fire_detection_time = None
        self.battery_depletion_time = None

        # Store episode data for plotting
        self.episode_data = {
            "timestamps": [],
            "battery_levels": [],
            "energy_budgets": [],
            "missed_fire_times": [],
            "sampling_time": [],
            "harvested_energy": [],
            "consumed_energy": [],
            "temperature": [],
            "humidity": [],
            "HDWI_score": [],
            "wind_speed": [],
            "ml_result": [],
            "take_a_picture": [],
            "label": [],
            "reward": [],
            "temp_trust_score": [],
            "temp_z_score": [],
            "humidity_trust_score": [],
            "humidity_z_score": [],
            "wind_trust_score": [],
            "wind_z_score": [],
            "used_neighbor_risk": []
        }

        # Observation space (state)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        # Action space (continuous: [1 min, 60 min])
        self.action_space = spaces.Box(low=np.array([self.config["TD3_params"]["min_sampling_time"]]), high=np.array([self.config["TD3_params"]["max_sampling_time"]]), dtype=np.float32)

    def reset(self, sensor_override=None):
        if sensor_override is None:
            print("Sensor override must be provided for sequential execution.")
            return None  
        self.current_sensor = sensor_override

        # Identify neighbor based on the full list of sensors in the shared dict
        all_sensors = list(self.sensor_shared_states.keys())
        if len(all_sensors) == 2:
            self.current_neighbor_sensor = [s for s in all_sensors if s != self.current_sensor][0]
        else:
            self.current_neighbor_sensor = None # No neighbor if not exactly 2 sensors

        # Set sensor data dynamically
        self.sensor_data = self.df.reset_index(drop=True) # df is already filtered in main
        
        self.sensor_selection_count[self.current_sensor] += 1  # Update visit count
        
        self.last_sampling_time = 0

        # Reset trust algorithms
        self.temp_trust_algo.reset()
        self.humidity_trust_algo.reset()
        self.wind_trust_algo.reset()
        self.temp_trust_score = 100.0
        self.humidity_trust_score = 100.0
        self.wind_trust_score = 100.0
        
        # Randomly choose one of the battery levels each episode
        battery_energy_dict = self.config["Initial_Battery_Levels"]
        battery_choice = np.random.choice(list(battery_energy_dict.keys()))
        initial_energy = battery_energy_dict[battery_choice]

        self.battery_energy = initial_energy  # Reset to full capacity
        self.max_battery_energy = initial_energy  # Full battery capacity
        self.energy_budget = initial_energy - self.config["Energy_Constraints"]["reserved_energy"]
        self.previous_ml_result = 0
        self.missed_fire = 0
        self.missed_fire_time = 0 # Reset missed fire time
        
        self.step_rewards_list = []
        
        self.reward = 0
        
        # Initialize fire start time as the first Label == 1 timestamp
        fire_rows = self.sensor_data[self.sensor_data["Label"] == 1]
        self.fire_start_time = fire_rows["Timestamp"].iloc[0] if not fire_rows.empty else None
        
        if not fire_rows.empty:
            fire_start_time = fire_rows["Timestamp"].iloc[0]
            allowed_indices = self.sensor_data[self.sensor_data["Timestamp"] <= (fire_start_time - pd.Timedelta(days=7))].index
        else:
            # No fire at all → just pick from the whole dataset
            allowed_indices = self.sensor_data.index

        # Find the start timestamp index + offset
        if len(self.sensor_data) <= self.start_offset:
            self.current_step = 0
        else:
            self.current_step = 0 # self.start_offset
        
        self.last_image_timestamp = self.sensor_data["Timestamp"].iloc[self.current_step]
          
        self.data_length = max(1, len(self.sensor_data) - (self.current_step + 1))
        self.fire_detection_time = None
        self.battery_depletion_time = None
        
        # Reset episode data
        self.episode_data = {
            "timestamps": [],
            "battery_levels": [],
            "energy_budgets": [],
            "missed_fire_times": [],
            "sampling_time": [],
            "harvested_energy": [],
            "consumed_energy": [],
            "temperature": [],
            "humidity": [],
            "HDWI_score": [],
            "wind_speed": [],
            "ml_result": [],
            "take_a_picture": [],
            "label": [],
            "reward": [],
            "temp_trust_score": [],
            "temp_z_score": [],
            "humidity_trust_score": [],
            "humidity_z_score": [],
            "wind_trust_score": [],
            "wind_z_score": [],
            "used_neighbor_risk": []
        }
        
        return self.get_state()

    def get_state(self):
        row = self.sensor_data.iloc[self.current_step]
        time_since_last_image = (row["Timestamp"] - self.last_image_timestamp).total_seconds() / 60

        return np.array([
            row["Temperature_2m_normalized"],
            row["Relative_Humidity_2m_normalized"],
            row["Wind_Speed_10m"],
            # row["Rain"], # This line is commented, so the shape is 11
            row["HDWI"],
            row["solar_energy"],
            normalize_feature(self.last_sampling_time, 1, self.config["TD3_params"]["max_sampling_time"]),
            normalize_feature(self.energy_budget, 0, self.max_battery_energy - self.config["Energy_Constraints"]["reserved_energy"]),
            normalize_feature(time_since_last_image, 0, 120),
            row["Time_of_Day"],
            row["Season"],
            self.previous_ml_result,
        ], dtype=np.float32) # Total = 11 elements

    def step(self, action):
        self.last_sampling_time = int(action) # Weather Sensor Read Interval

        # Find the next timestamp based on the RL decision
        next_timestamp = self.sensor_data.iloc[self.current_step]["Timestamp"] + pd.Timedelta(minutes=self.last_sampling_time)
        
        next_steps = self.sensor_data[self.sensor_data["Timestamp"] >= next_timestamp].index
        
        # Get all rows within the skipped time range
        skipped_data = self.sensor_data[
            (self.sensor_data["Timestamp"] > self.sensor_data.iloc[self.current_step]["Timestamp"]) &
            (self.sensor_data["Timestamp"] < next_timestamp)
        ]

        if not next_steps.empty:
            self.current_step = next_steps[0]
        else:
            self.current_step = len(self.sensor_data) - 1  # Stop at last timestamp

        row = self.sensor_data.iloc[self.current_step]

        # Update trust scores for all metrics
        self.temp_trust_score, temp_state, temp_z_score = self.temp_trust_algo.process_reading(row["Temperature_2m"])
        self.humidity_trust_score, humidity_state, humidity_z_score = self.humidity_trust_algo.process_reading(row["Relative_Humidity_2m"])
        self.wind_trust_score, wind_state, wind_z_score = self.wind_trust_algo.process_reading(row["Wind_Speed_10m"])

        unreliable_sensor_penalty = 0
        used_neighbor_risk = 0

        is_critical = temp_state == "Critical" or humidity_state == "Critical" or wind_state == "Critical"

        if is_critical and self.current_neighbor_sensor:
            # Fetch the last known value from the neighbor via the shared state
            take_picture = self.sensor_shared_states[self.current_neighbor_sensor]
            
            # Apply energy penalty for communication
            unreliable_sensor_penalty += self.config["Energy_Constraints"].get("E_unreliable_sensor", 0.05)
            used_neighbor_risk = 1
        else:
            # Data is reliable, so calculate our own value
            features = {
                "avgtempC": row["Temperature_2m"],
                "humid": row["Relative_Humidity_2m"]
            }
            df_features = pd.DataFrame([features])
            take_picture = int(self.dt_model.predict(df_features)[0])
            
            # "Post" our new status to the shared state for others to see
            self.sensor_shared_states[self.current_sensor] = take_picture
        
        # Combine skipped rows and the RL-decided row
        fire_rows = pd.concat([skipped_data, self.sensor_data.iloc[[self.current_step]]])

        """ # Check if fire exists and update fire_start_time if not set
        if self.fire_start_time is None and (fire_rows["Label"] == 1).any():
            self.fire_start_time = fire_rows.loc[fire_rows["Label"] == 1, "Timestamp"].iloc[0] """

        ml_result = 0

        if take_picture:
            ml_result = np.random.choice([1, 0], p=[self.config["ML_Performance"]["TP_rate"], 1 - self.config["ML_Performance"]["TP_rate"]]) if row["Label"] == 1 else \
                        np.random.choice([1, 0], p=[self.config["ML_Performance"]["FP_rate"], 1 - self.config["ML_Performance"]["FP_rate"]])

            if row["Label"] == 1 and ml_result == 1:
                self.fire_detection_time = row["Timestamp"] # Fire detection tracking
                    
            self.last_image_timestamp = row["Timestamp"]
        
        neighbor_comm_energy = self.config["Neighborhood_Communication"]["num_neighbors"] * self.config["Neighborhood_Communication"]["E_comm_neighbor"]

        # Check for missed fires safely
        if not skipped_data.empty and "Label" in skipped_data.columns:
            self.missed_fire = any((skipped_data["Label"] == 1) &
                            ((row["Timestamp"] - skipped_data["Timestamp"]).dt.total_seconds() / 60 > self.config["max_missing_fire_min"]))

        if row["Label"] == 1:
            self.missed_fire_time = (row["Timestamp"] - self.fire_start_time).total_seconds() / 60

        # Sum up harvested energy for every minute in the skipped time
        total_harvested_energy = (fire_rows["solar_energy"] * self.config["harvested_energy_loss"]).sum() if not fire_rows.empty else 0
        
        time_skipped_hours = self.last_sampling_time / 60  # Convert to hours
        standby_power_used = (
            self.config["Standby_Power_Components"]["P_temp_humidity_standby"] + self.config["Standby_Power_Components"]["P_anemometer_standby"] +
            self.config["Standby_Power_Components"]["P_camera_standby"] + self.config["Standby_Power_Components"]["P_comm_standby"]
        ) * time_skipped_hours  # Multiply by time skipped

        # Energy management with standby power
        energy_used = (
            self.config["Energy_Constraints"]["E_proc_rl"] + self.config["Energy_Constraints"]["E_temp_humidity_sensor"] + self.config["Energy_Constraints"]["E_anemometer_sensor"] + (self.config["Energy_Constraints"]["E_proc_ml"] + self.config["Energy_Constraints"]["E_camera_host"] if take_picture else 0) +
            standby_power_used + (self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy if ml_result and take_picture else 0) +
            unreliable_sensor_penalty
        )
        
        # Simulate minute-by-minute depletion during skipped time
        if self.battery_depletion_time is None:
            current_batt = self.battery_energy
            max_batt_energy = self.max_battery_energy
            E_used = (
                self.config["Energy_Constraints"]["E_proc_rl"] + self.config["Energy_Constraints"]["E_temp_humidity_sensor"] + self.config["Energy_Constraints"]["E_anemometer_sensor"] + (self.config["Energy_Constraints"]["E_proc_ml"] + self.config["Energy_Constraints"]["E_camera_host"] if take_picture else 0) +
                (self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy if ml_result and take_picture else 0)
            )
            for i, ts in enumerate(fire_rows["Timestamp"]):
                harvested = fire_rows.iloc[i]["solar_energy"] * self.config["harvested_energy_loss"]
                standby_used = standby_power_used / len(fire_rows) # Distribute standby cost equally
                net_energy = harvested - standby_used 
                # No sensing/ML/comm costs since device is idle, just standby and harvested

                current_batt += net_energy
                current_batt *= (1 - self.config["E_battery_leakage_percentage"])
                max_batt_energy *= (1 - self.config["E_battery_leakage_percentage"])
                current_batt = max(0, min(max_batt_energy, current_batt))

                if current_batt - self.config["Energy_Constraints"]["reserved_energy"] <= 0:
                    self.battery_depletion_time = ts
                    break  # First time battery hits 0
                
            if  current_batt - self.config["Energy_Constraints"]["reserved_energy"] > 0 and current_batt - E_used - self.config["Energy_Constraints"]["reserved_energy"] <= 0:
                self.battery_depletion_time = row["Timestamp"]
            
        # Update battery energy first
        self.battery_energy += total_harvested_energy - energy_used
        self.battery_energy *= (1 - self.config["E_battery_leakage_percentage"])  # Apply battery leakage loss
        self.max_battery_energy *= (1 - self.config["E_battery_leakage_percentage"])  # Apply battery leakage loss
        self.battery_energy = max(0, min(self.max_battery_energy, self.battery_energy))  # Ensure valid range

        # Update energy budget after battery energy is updated
        self.energy_budget = max(0, self.battery_energy - self.config["Energy_Constraints"]["reserved_energy"])  

        self.previous_ml_result = ml_result
        
        # Default: Episode continues
        done = self.current_step >= len(self.sensor_data) - 1  # Stops when dataset is finished

        # Stop the episode if fire is detected, missed, or energy is 0
        # if (row["Label"] == 1 and ml_result == 1) or self.missed_fire_time >= 15 or self.energy_budget <= 0:
        if (row["Label"] == 1 and ml_result == 1) or self.energy_budget <= 0 or self.battery_depletion_time is not None:
            print(f'Next sensor time: {self.last_sampling_time}, Stopping Condition Met: Fire Detected: {row["Label"] == 1 and ml_result == 1}, Missed Fire for {self.missed_fire_time} min: {self.missed_fire_time >= 30}, Energy Depleted: {self.energy_budget <= 0}, Battery Depletion Time: {self.battery_depletion_time}')
            self.detection_time_list.append(self.missed_fire_time)
            done = True # Stops the episode
            
        # Store episode data for plotting
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
        self.episode_data["temp_trust_score"].append(self.temp_trust_score)
        self.episode_data["temp_z_score"].append(temp_z_score)
        self.episode_data["humidity_trust_score"].append(self.humidity_trust_score)
        self.episode_data["humidity_z_score"].append(humidity_z_score)
        self.episode_data["wind_trust_score"].append(self.wind_trust_score)
        self.episode_data["wind_z_score"].append(wind_z_score)
        self.episode_data["used_neighbor_risk"].append(used_neighbor_risk)

        """ if self.battery_energy > 5:
            self.reward = - k1 * self.last_sampling_time
        else:
            self.reward = k1 * self.last_sampling_time """
        
        self.reward  = self.config["Reward_Params"]["beta"] * self.last_sampling_time + (1 - self.config["Reward_Params"]["beta"]) * self.reward
        
        step_reward = self.last_sampling_time * (1 - 2 * take_picture) / self.data_length
        
        self.episode_data["reward"].append(step_reward)
        
        self.step_sampling_log.append((self.total_env_steps, self.last_sampling_time))
        

        # If episode ends, generate plot
        if done:
            self.episode_counter += 1  # Increment episode counter
            avg_sampling_time = np.mean(self.episode_data["sampling_time"]) 
            final_reward, reason = self.calculate_final_reward() # -k1 * self.reward, "xx" #self.calculate_final_reward()
            print("final_reward", final_reward)
            self.step_rewards_list.append(final_reward)  # Store final reward globally
            
            self.step_reward_log.append((self.total_env_steps, final_reward))
            
            self.total_env_steps += 1
            
            self.average_rewards_list.append(sum(self.step_rewards_list) / len(self.step_rewards_list))
            self.cumulative_rewards_list.append(sum(self.step_rewards_list))
            self.tensorboard_rewards_list.append(sum(self.cumulative_rewards_list) / len(self.cumulative_rewards_list))
        
            self.avg_sampling_time_list.append(avg_sampling_time)
            print("avg_sampling_time", avg_sampling_time)

            self.plot_episode_metrics(reason, final_reward)
            return self.get_state(), final_reward, done, {}
        
        self.step_rewards_list.append(step_reward)  # Store step reward globally
        self.step_reward_log.append((self.total_env_steps, step_reward))
        self.total_env_steps += 1

        return self.get_state(), step_reward, done, {}
    
    def calculate_final_reward(self):
        """Calculate the reward at the end of the episode."""
        if self.battery_depletion_time is not None and (not self.fire_start_time or self.battery_depletion_time < self.fire_start_time):
            # Case 1: Battery depletes before fire
            t_deplete_minus_t_start = (self.battery_depletion_time - self.sensor_data["Timestamp"].iloc[0]).total_seconds() / 60
            reward = -self.config["Reward_Params"]["alpha_B"] * (1 / (t_deplete_minus_t_start)) - self.config["Reward_Params"]["R_min"]
            reason = "case1"
            print('reward case1 ', reward, "t_deplete_minus_t_start", t_deplete_minus_t_start)
        else:
            reward = -self.config["Reward_Params"]["k1"] * self.reward  
            
            reason = "case2/3"
            print('reward case2/3 ', reward)

        return reward, reason
    
    def plot_episode_metrics(self, reason, final_reward):
        """ if self.episode_counter % 10 != 0:
            return  # Skip saving unless episode number is a multiple of 100 """
        """Generates and saves episode-specific plots inside `episode_plots/`."""
        file_name = self.config["file_name"]
        folder = f"Inference/episode_plots_step_reward{file_name}"
        os.makedirs(folder, exist_ok=True)
        
        # Save the loaded config into the folder
        with open(os.path.join(folder, "config_setup.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        df2 = pd.DataFrame(self.episode_data)

        # Save it to a CSV file
        df2.to_csv(f"{folder}/episode_{self.episode_counter}_{self.current_sensor}.csv", index=False)  # index=False avoids adding an extra index column

        fig, axs = plt.subplots(10, 2, figsize=(15, 28), sharex=True)
        axs = axs.flatten()

        axs[0].scatter(self.episode_data["timestamps"], self.episode_data["harvested_energy"], label="Harvested Energy (Wh)", color='green')
        axs[1].scatter(self.episode_data["timestamps"], self.episode_data["consumed_energy"], label="Consumed Energy (Wh)", color='green')
        axs[2].scatter(self.episode_data["timestamps"], self.episode_data["energy_budgets"], label="Energy Budget (Wh)", color='green')
        axs[3].scatter(self.episode_data["timestamps"], self.episode_data["battery_levels"], label="Battery Energy (Wh)", color='green')
        
        
        axs[4].scatter(self.episode_data["timestamps"], self.episode_data["temperature"], label="Temperature (C)", color='green')
        axs[5].scatter(self.episode_data["timestamps"], self.episode_data["humidity"], label="Humidity (%)", color='green')
        axs[6].scatter(self.episode_data["timestamps"], self.episode_data["HDWI_score"], label="HDWI score", color='green')
        axs[7].scatter(self.episode_data["timestamps"], self.episode_data["wind_speed"], label="Wind Speed (km/h)", color='green')
        
        axs[8].scatter(self.episode_data["timestamps"], self.episode_data["sampling_time"], label="Sampling Time (min)", color='green')
        axs[9].scatter(self.episode_data["timestamps"], self.episode_data["take_a_picture"], label="Take a picture [0, 1]", color='green')
        axs[10].scatter(self.episode_data["timestamps"], self.episode_data["ml_result"], label="ML result [0, 1]", color='green')
        
        axs[11].scatter(self.episode_data["timestamps"], self.episode_data["missed_fire_times"], label="Missed Fire Time (min)", color='green')
        
        axs[12].scatter(self.episode_data["timestamps"], self.episode_data["reward"], label=f"Step Reward {reason} {final_reward}", color='green')
        
        axs[13].scatter(self.episode_data["timestamps"], self.episode_data["temp_trust_score"], label="Temperature Trust Score", color='purple')
        axs[14].scatter(self.episode_data["timestamps"], self.episode_data["temp_z_score"], label="Temperature Z-Score", color='purple')
        
        axs[15].scatter(self.episode_data["timestamps"], self.episode_data["humidity_trust_score"], label="Humidity Trust Score", color='blue')
        axs[16].scatter(self.episode_data["timestamps"], self.episode_data["humidity_z_score"], label="Humidity Z-Score", color='blue')

        axs[17].scatter(self.episode_data["timestamps"], self.episode_data["wind_trust_score"], label="Wind Trust Score", color='orange')
        axs[18].scatter(self.episode_data["timestamps"], self.episode_data["wind_z_score"], label="Wind Z-Score", color='orange')

        axs[19].scatter(self.episode_data["timestamps"], self.episode_data["used_neighbor_risk"], label="Used Neighbor Risk [0, 1]", color='black')

        axs[-1].set_xlabel("Timestamp (min)", fontsize=16, fontweight='bold')
        
        if self.fire_start_time is not None:
            for ax in axs:
                ax.axvline(x=self.fire_start_time, color='red', linestyle='--', label='First Fire Detected')

        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y\n%I:%M %p'))  # Format timestamps
            ax.tick_params(axis='x', labelsize=12, width=2, rotation=45)  # Rotate & set label size
            ax.tick_params(axis='y', labelsize=12, width=2)
            ax.legend()
            ax.grid(True)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/episode_{self.episode_counter}_{self.current_sensor}.png", dpi=300)
        plt.close()

        print(f"Episode {self.episode_counter} plot saved.")
