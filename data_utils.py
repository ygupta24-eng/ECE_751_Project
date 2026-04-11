import pandas as pd
import numpy as np


def filter_last_month(group, episode_length_days):
    last_ts  = group["Timestamp"].max()
    start_ts = last_ts - pd.Timedelta(days=episode_length_days)
    return group[group["Timestamp"] >= start_ts]


def interpolate_group(group, sensor_val, location_val):
    """
    Interpolate one sensor group to 1-minute resolution.

    sensor_val and location_val are passed in explicitly because pandas 2.x
    drops the groupby key column from the group before calling apply(),
    making it impossible to recover them from inside the function.
    We avoid that problem entirely by not using groupby().apply() —
    prepare_dataset() loops over sensors manually instead.
    """
    # ── Dedup: fire rows win, then sort by time ───────────────────────────────
    if "Label" in group.columns:
        group = group.sort_values(by="Label", ascending=False)
    group = group.drop_duplicates(subset="Timestamp", keep="first")
    group = group.sort_values(by="Timestamp")

    # ── Build full 1-minute index ─────────────────────────────────────────────
    minutely_index = pd.date_range(
        start=group["Timestamp"].min(),
        end=(
            group["Timestamp"].max()
            + pd.Timedelta(hours=1)
            - pd.Timedelta(minutes=1)
        ),
        freq="1min",      # pandas 2.2+ (replaces deprecated "1T")
    )

    # ── Reindex to 1-minute resolution ───────────────────────────────────────
    group = group.set_index("Timestamp").reindex(minutely_index)

    # ── Interpolate numeric weather columns ───────────────────────────────────
    numeric_cols = [
        "Temperature_2m",
        "Relative_Humidity_2m",
        "Wind_Speed_10m",
        "HDWI",
        "Rain",
    ]
    existing = [c for c in numeric_cols if c in group.columns]
    group[existing] = group[existing].interpolate(method="linear")

    # ── Solar energy: distribute hourly value across 60 minutes ──────────────
    if "solar_energy" in group.columns:
        group["solar_energy"] = group["solar_energy"].ffill() / 60

    # ── Labels ────────────────────────────────────────────────────────────────
    if "Label" in group.columns:
        group["Label"] = group["Label"].ffill().bfill()

    # ── Restore identity columns using explicitly passed scalars ─────────────
    # This is the reliable fix for pandas 2.x dropping the groupby key column.
    group["Sensor"]   = sensor_val
    group["Location"] = location_val

    # ── Reset index so Timestamp is a regular column again ───────────────────
    group = group.reset_index().rename(columns={"index": "Timestamp"})
    return group


def get_season(month):
    if 3 <= month <= 5:
        return 0.25   # Spring
    elif 6 <= month <= 8:
        return 0.50   # Summer
    elif 9 <= month <= 11:
        return 0.75   # Fall
    else:
        return 1.00   # Winter


def normalize_feature(x, min_val, max_val):
    return np.clip((x - min_val) / (max_val - min_val), 0, 1)


def normalize(data):
    range_val = data.max() - data.min()
    return (data - data.min()) / range_val if range_val != 0 else data * 0


def prepare_dataset(config):
    """
    Loads and preprocesses the sensor dataset.
    Called by both inference_main.py and the smoke test.
    """
    episode_length_days = config["episode_length_days"]

    # ── Load raw CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv("test_set.csv")
    df["Timestamp"] = (
        pd.to_datetime(df["Timestamp"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Etc/GMT+8")
    )
    df = df.sort_values(by="Timestamp")

    # ── Interpolate to 1-minute resolution — manual loop per sensor ───────────
    # We deliberately avoid groupby().apply() here because pandas 2.x drops
    # the groupby key column ("Sensor") from the group passed to the function,
    # making it impossible to restore it inside the function.
    # Looping manually sidesteps this entirely.
    interpolated_parts = []
    for sensor_val, group in df.groupby("Sensor", sort=False):
        # Grab location before the group is modified
        location_val = group["Location"].iloc[0] if "Location" in group.columns else ""
        part = interpolate_group(group.copy(), sensor_val, location_val)
        interpolated_parts.append(part)

    df = pd.concat(interpolated_parts, ignore_index=True)

    # ── Sanity check ─────────────────────────────────────────────────────────
    if "Sensor" not in df.columns:
        raise RuntimeError(
            "Sensor column missing after interpolation. "
            "Check interpolate_group is setting group['Sensor'] = sensor_val."
        )

    # ── Keep only the last N days per sensor ──────────────────────────────────
    filtered_parts = []
    for _, group in df.groupby("Sensor", sort=False):
        filtered_parts.append(filter_last_month(group, episode_length_days))
    df = pd.concat(filtered_parts, ignore_index=True)

    # ── Remove known duplicate / bad sensors ─────────────────────────────────
    duplicates_to_remove = [
        "bh-s-mobo-c1", "bh-w-mobo-c2", "bl-s-mobo-c1", "bm-w-mobo-c1",
        "hp-w-mobo-c1", "lp-n-mobo-c3", "mlo-s-mobo-c1", "om-w-mobo-c1",
        "pi-w-mobo-c1", "so-w-mobo-c1", "sp-w-mobo-c1",  "lp-n-mobo1",
        "om-n-mobo-c1", "lp-s-mobo-c1", "bh-n-mobo-c2",  "sdsc-e-mobo-c1",
        "hp-w-mobo-c2",
    ]
    df = df[~df["Sensor"].isin(duplicates_to_remove)].copy()

    # ── Normalise features ────────────────────────────────────────────────────
    df["Temperature_2m_normalized"]       = normalize(df["Temperature_2m"])
    df["Relative_Humidity_2m_normalized"] = normalize(df["Relative_Humidity_2m"])
    df["Wind_Speed_10m"]                  = normalize(df["Wind_Speed_10m"])
    df["HDWI"]                            = normalize(df["HDWI"])
    df["Rain"]                            = normalize(df["Rain"])

    # ── Temporal features ─────────────────────────────────────────────────────
    df["Season"]      = df["Timestamp"].dt.month.apply(get_season)
    df["Time_of_Day"] = df["Timestamp"].dt.hour / 23

    return df