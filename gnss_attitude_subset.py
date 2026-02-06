import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- small helpers ----------
def _append_suffix(path_str, suffix):
    p = Path(path_str)
    return str(p.with_name(p.stem + suffix + p.suffix)) if p.suffix else str(p.with_name(p.name + suffix))

def _find_name(cols, candidates):
    """
    Return the first matching column name using:
      1) case-insensitive exact match on any 'candidates'
      2) token fallback: every token in cand.lower().split(':') must appear in col.lower()
    """
    # exact
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    # tokens
    low = [c.lower() for c in cols]
    for cand in candidates:
        toks = [t for t in cand.lower().split(":") if t]
        for i, name in enumerate(low):
            if all(tok in name for tok in toks):
                return cols[i]
    return None

def _find_first_containing(cols, token_lists):
    """Find the first column whose lowercased name contains all tokens in ANY of token_lists (checked in order)."""
    low = [c.lower() for c in cols]
    for tokens in token_lists:
        for i, name in enumerate(low):
            if all(tok in name for tok in tokens):
                return cols[i]
    return None

def _find_angle_col(cols, primary_lists, avoid_tokens=("rate","gyro")):
    """Pick an angle column (e.g., roll) while avoiding names that look like rates/gyros."""
    low = [c.lower() for c in cols]
    for tokens in primary_lists:
        for i, name in enumerate(low):
            if all(tok in name for tok in tokens) and not any(bad in name for bad in avoid_tokens):
                return cols[i]
    return None

def _find_with_tokens(cols, tokens):
    return _find_first_containing(cols, [tokens])

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Build an attitude subset CSV with GNSS-derived Unix timestamps.")
    parser.add_argument("input_csv", help="CSV exported from DatCon (e.g., 2025-12-09_17-30-39_FLY075.csv)")
    args = parser.parse_args()

    """
    Outputs <input>_gnss_att.csv with:
      tick, unix, fractional_seconds, gps_date, gps_time, gps_num_sats,
      latitude, longitude, altitude, relative_height, roll, pitch, yaw.
    NOTE: roll/pitch/yaw are **degrees** (DJI/DatCon IMU attitude fields are in deg).
    """
    infile = args.input_csv
    outfile = _append_suffix(infile, "_gnss_att")

    # Read as strings to preserve untouched formatting
    df = pd.read_csv(infile, dtype=str, keep_default_na=False)

    # ---- 1) TIME HANDLING (GNSS -> unix; ticks -> fractional_seconds) ----
    date_col = _find_name(df.columns, ["GPS:Date","gps:date","GPS:date"])
    time_col = _find_name(df.columns, ["GPS:Time","gps:time","GPS:time"])
    if not date_col or not time_col:
        raise ValueError("Required columns GPS:Date and/or GPS:Time not found.")

    tick_col = _find_name(df.columns, ["Clock:Tick#","Clock:Tick","clock:tick#","clock:tick","Tick#","tick#","tick"])

    dt_str = (df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()).replace({"": np.nan})
    dt = pd.to_datetime(dt_str, utc=True, errors="coerce")
    unix_int = np.floor(dt.view("int64") / 1e9).astype("float")  # keep NaN for bad rows
    valid_unix = ~np.isnan(unix_int)

    frac = np.full(len(df), np.nan, dtype=float)
    if tick_col:
        tick = pd.to_numeric(df[tick_col], errors="coerce").to_numpy()
        unix_series = pd.Series(unix_int)
        uSec = unix_series.dropna().astype(np.int64).drop_duplicates(keep="first")

        first_idx, last_idx = {}, {}
        for sec in uSec:
            rows = np.where(unix_int == sec)[0]
            first_idx[sec] = rows[0]; last_idx[sec] = rows[-1]

        denoms = {}
        for sec in uSec:
            i_first = first_idx[sec]; i_last = last_idx[sec]
            next_first = first_idx.get(sec+1, None)
            if next_first is not None and np.isfinite(tick[next_first]) and np.isfinite(tick[i_first]):
                d = tick[next_first] - tick[i_first]
            else:
                d = tick[i_last] - tick[i_first] if np.isfinite(tick[i_last]) and np.isfinite(tick[i_first]) else np.nan
            denoms[sec] = d if (np.isfinite(d) and d > 0) else np.nan

        denom_vals = np.array([v for v in denoms.values() if np.isfinite(v) and v > 0], dtype=float)
        denom_med = np.median(denom_vals) if denom_vals.size else np.nan

        for sec in uSec:
            rows = np.where(unix_int == sec)[0]
            i_first = first_idx[sec]
            if not np.isfinite(tick[i_first]): 
                continue
            d = denoms.get(sec, np.nan)
            if not (np.isfinite(d) and d > 0):
                d = denom_med
            if not (np.isfinite(d) and d > 0):
                continue
            frac[rows] = (tick[rows] - tick[i_first]) / d

    frac = np.clip(frac, 0.0, 0.999999)
    frac6 = np.round(frac, 6)

    unix_str = pd.Series(unix_int).where(valid_unix, other=np.nan)
    unix_str = unix_str.dropna().astype("Int64").astype(str).reindex(range(len(df))).fillna("")
    frac_str = pd.Series(frac6).map(lambda v: f"{float(v):.6f}" if np.isfinite(v) else "0.000000")

    # ---- 2) PICK COLUMNS ----
    cols = list(df.columns)

    # Lat / Lon
    lat_col = _find_first_containing(cols, [["gps","latitude"], ["latitude"], ["lat"]])
    lon_col = _find_first_containing(cols, [["gps","longitude"], ["longitude"], ["lon"]])

    # Absolute altitude: prefer your column first
    alt_abs_col = (
        _find_name(cols, ["IMU_ATTI(0):alti:D[meters]"])  # explicit
        or _find_first_containing(cols, [["gps","alt"]])
        or _find_first_containing(cols, [["absoluteheight"]])
        or _find_first_containing(cols, [["alti"]])
        or _find_first_containing(cols, [["altitude"]])
        or _find_first_containing(cols, [["height"]])
    )

    # Relative height
    rel_h_col = (
        _find_first_containing(cols, [["relativeheight"]])
        or _find_first_containing(cols, [["relative","height"]])
        or _find_first_containing(cols, [["height","relative"]])
        or _find_first_containing(cols, [["height","takeoff"]])
        or _find_first_containing(cols, [["height","home"]])
        or _find_first_containing(cols, [["agl"]])
        or _find_first_containing(cols, [["relative","alt"]])
        or _find_first_containing(cols, [["rel","height"]])
    )

    # Attitude angles (degrees). Avoid rate/gyro columns.
    roll_col  = _find_angle_col(cols, [["roll"], ["atti","roll"], ["imu","atti","roll"]])
    pitch_col = _find_angle_col(cols, [["pitch"], ["atti","pitch"], ["imu","atti","pitch"]])
    yaw_col   = _find_angle_col(cols, [["yaw"], ["atti","yaw"], ["imu","atti","yaw"]])

    # GPS num sats
    sats_col = (
        _find_first_containing(cols, [["gps","numsats"]])
        or _find_first_containing(cols, [["numsats"]])
        or _find_first_containing(cols, [["satellites"]])
        or _find_first_containing(cols, [["num","sat"]])
    )

    # ---- 2b) RFI-RELEVANT STATUS/POWER SIGNALS ----
    # Controller signal level
    ctrl_sig_col = _find_name(cols, ["Controller:sig_level:D"]) or _find_first_containing(cols, [["controller","sig_level"]])

    # GPS quality / DOP
    gps_hdop_col = _find_name(cols, ["GPS:hDOP"]) or _find_first_containing(cols, [["gps","hdop"]])
    gps_pdop_col = _find_name(cols, ["GPS:pDOP"]) or _find_first_containing(cols, [["gps","pdop"]])
    gps_numgps_col = _find_name(cols, ["GPS:numGPS"]) or _find_first_containing(cols, [["gps","numgps"]])
    gps_numglnas_col = _find_name(cols, ["GPS:numGLNAS"]) or _find_first_containing(cols, [["gps","numglnas"]])
    gps_numsv_col = _find_name(cols, ["GPS:numSV"]) or _find_first_containing(cols, [["gps","numsv"]])

    # GPS hardware status (noise/jamming/spoof)
    gps_hw_noise_col = _find_name(cols, ["gps_hw_status:noiseperms:D"]) or _find_with_tokens(cols, ["gps_hw_status","noiseperms"])
    gps_hw_agc_col = _find_name(cols, ["gps_hw_status:agccnt:D"]) or _find_with_tokens(cols, ["gps_hw_status","agccnt"])
    gps_hw_jamind_col = _find_name(cols, ["gps_hw_status:jamind:D"]) or _find_with_tokens(cols, ["gps_hw_status","jamind"])
    gps_hw_flag_col = _find_name(cols, ["gps_hw_status:flag:D"]) or _find_with_tokens(cols, ["gps_hw_status","flag"])
    gps_hw_jamstate_col = _find_name(cols, ["gps_hw_status:jammingState:D"]) or _find_with_tokens(cols, ["gps_hw_status","jammingstate"])
    gps_hw_spoof_col = _find_name(cols, ["gps_hw_status:spoofState:D"]) or _find_with_tokens(cols, ["gps_hw_status","spoofstate"])

    # RTK (if present)
    rtk_gpsstate_col = _find_name(cols, ["raw_rtk_data:GpsState:D"]) or _find_with_tokens(cols, ["raw_rtk_data","gpsstate"])
    rtk_hdop_col = _find_name(cols, ["raw_rtk_data:hdop:D"]) or _find_with_tokens(cols, ["raw_rtk_data","hdop"])
    rtk_posflg_cols = []
    for i in range(6):
        c = _find_name(cols, [f"raw_rtk_data:posFlg_{i}:D"]) or _find_with_tokens(cols, ["raw_rtk_data", f"posflg_{i}"])
        rtk_posflg_cols.append(c)

    # Motors / ESC / power
    motor_speed_cols = {
        "rfront": _find_name(cols, ["Motor:Speed:RFront[rpm]"]) or _find_with_tokens(cols, ["motor","speed","rfront"]),
        "lfront": _find_name(cols, ["Motor:Speed:LFront[rpm]"]) or _find_with_tokens(cols, ["motor","speed","lfront"]),
        "lback": _find_name(cols, ["Motor:Speed:LBack[rpm]"]) or _find_with_tokens(cols, ["motor","speed","lback"]),
        "rback": _find_name(cols, ["Motor:Speed:RBack[rpm]"]) or _find_with_tokens(cols, ["motor","speed","rback"]),
    }
    motor_pwm_cols = {
        "rfront": _find_name(cols, ["MotorCtrl:PWM:RFront[%]"]) or _find_with_tokens(cols, ["motorctrl","pwm","rfront"]),
        "lfront": _find_name(cols, ["MotorCtrl:PWM:LFront[%]"]) or _find_with_tokens(cols, ["motorctrl","pwm","lfront"]),
        "lback": _find_name(cols, ["MotorCtrl:PWM:LBack[%]"]) or _find_with_tokens(cols, ["motorctrl","pwm","lback"]),
        "rback": _find_name(cols, ["MotorCtrl:PWM:RBack[%]"]) or _find_with_tokens(cols, ["motorctrl","pwm","rback"]),
    }
    motor_ppm_cols = {
        "rfront": _find_name(cols, ["Motor:PPMrecv:RFront[%]"]) or _find_with_tokens(cols, ["motor","ppmrecv","rfront"]),
        "lfront": _find_name(cols, ["Motor:PPMrecv:LFront[%]"]) or _find_with_tokens(cols, ["motor","ppmrecv","lfront"]),
        "lback": _find_name(cols, ["Motor:PPMrecv:LBack[%]"]) or _find_with_tokens(cols, ["motor","ppmrecv","lback"]),
        "rback": _find_name(cols, ["Motor:PPMrecv:RBack[%]"]) or _find_with_tokens(cols, ["motor","ppmrecv","rback"]),
    }
    motor_current_cols = {
        "rfront": _find_name(cols, ["Motor:Current:RFront[Amperes]"]) or _find_with_tokens(cols, ["motor","current","rfront"]),
        "lfront": _find_name(cols, ["Motor:Current:LFront[Amperes]"]) or _find_with_tokens(cols, ["motor","current","lfront"]),
        "lback": _find_name(cols, ["Motor:Current:LBack[Amperes]"]) or _find_with_tokens(cols, ["motor","current","lback"]),
        "rback": _find_name(cols, ["Motor:Current:RBack[Amperes]"]) or _find_with_tokens(cols, ["motor","current","rback"]),
    }
    motor_volts_cols = {
        "rfront": _find_name(cols, ["Motor:Volts:RFront[volts]"]) or _find_with_tokens(cols, ["motor","volts","rfront"]),
        "lfront": _find_name(cols, ["Motor:Volts:LFront[volts]"]) or _find_with_tokens(cols, ["motor","volts","lfront"]),
        "lback": _find_name(cols, ["Motor:Volts:LBack[volts]"]) or _find_with_tokens(cols, ["motor","volts","lback"]),
        "rback": _find_name(cols, ["Motor:Volts:RBack[volts]"]) or _find_with_tokens(cols, ["motor","volts","rback"]),
    }
    motor_vout_cols = {
        "rfront": _find_name(cols, ["Motor:V_out:RFront[volts]"]) or _find_with_tokens(cols, ["motor","v_out","rfront"]),
        "lfront": _find_name(cols, ["Motor:V_out:LFront[volts]"]) or _find_with_tokens(cols, ["motor","v_out","lfront"]),
        "lback": _find_name(cols, ["Motor:V_out:LBack[volts]"]) or _find_with_tokens(cols, ["motor","v_out","lback"]),
        "rback": _find_name(cols, ["Motor:V_out:RBack[volts]"]) or _find_with_tokens(cols, ["motor","v_out","rback"]),
    }
    motor_esct_cols = {
        "rfront": _find_name(cols, ["Motor:EscTemp:RFront[degrees]"]) or _find_with_tokens(cols, ["motor","esctemp","rfront"]),
        "lfront": _find_name(cols, ["Motor:EscTemp:LFront[degrees]"]) or _find_with_tokens(cols, ["motor","esctemp","lfront"]),
        "lback": _find_name(cols, ["Motor:EscTemp:LBack[degrees]"]) or _find_with_tokens(cols, ["motor","esctemp","lback"]),
        "rback": _find_name(cols, ["Motor:EscTemp:RBack[degrees]"]) or _find_with_tokens(cols, ["motor","esctemp","rback"]),
    }

    # Battery (for power/ripple context)
    batt_voltage_col = _find_name(cols, ["battery_info:Voltage:D"]) or _find_with_tokens(cols, ["battery_info","voltage"])
    batt_current_col = _find_name(cols, ["battery_info:Current:D"]) or _find_with_tokens(cols, ["battery_info","current"])
    batt_pct_col = _find_name(cols, ["battery_info:CapPercentage:D"]) or _find_with_tokens(cols, ["battery_info","cappercentage"])
    batt_temp_col = _find_name(cols, ["battery_info:BatTemperature:D"]) or _find_with_tokens(cols, ["battery_info","battemperature"])

    # ---- 3) BUILD OUTPUT (tick first, then the rest) ----
    N = len(df)
    def blank(): return pd.Series([""]*N)

    out_cols = {}
    out_cols["tick"] = df[tick_col] if tick_col else blank()
    out_cols["unix"] = unix_str
    out_cols["fractional_seconds"] = frac_str
    out_cols["gps_date"] = df[date_col]
    out_cols["gps_time"] = df[time_col]
    out_cols["gps_num_sats"] = df[sats_col] if sats_col else blank()
    out_cols["latitude"]  = df[lat_col] if lat_col else blank()
    out_cols["longitude"] = df[lon_col] if lon_col else blank()
    out_cols["altitude"]        = df[alt_abs_col] if alt_abs_col else blank()
    out_cols["relative_height"] = df[rel_h_col]  if rel_h_col  else blank()
    out_cols["roll"]  = df[roll_col]  if roll_col  else blank()
    out_cols["pitch"] = df[pitch_col] if pitch_col else blank()
    out_cols["yaw"]   = df[yaw_col]   if yaw_col   else blank()

    # Controller signal
    out_cols["controller_signal_level"] = df[ctrl_sig_col] if ctrl_sig_col else blank()

    # GPS quality
    out_cols["gps_hdop"] = df[gps_hdop_col] if gps_hdop_col else blank()
    out_cols["gps_pdop"] = df[gps_pdop_col] if gps_pdop_col else blank()
    out_cols["gps_num_gps"] = df[gps_numgps_col] if gps_numgps_col else blank()
    out_cols["gps_num_glnas"] = df[gps_numglnas_col] if gps_numglnas_col else blank()
    out_cols["gps_num_sv"] = df[gps_numsv_col] if gps_numsv_col else blank()

    # GPS HW status (jamming/spoof indicators)
    out_cols["gps_hw_noiseperms"] = df[gps_hw_noise_col] if gps_hw_noise_col else blank()
    out_cols["gps_hw_agccnt"] = df[gps_hw_agc_col] if gps_hw_agc_col else blank()
    out_cols["gps_hw_jamind"] = df[gps_hw_jamind_col] if gps_hw_jamind_col else blank()
    out_cols["gps_hw_flag"] = df[gps_hw_flag_col] if gps_hw_flag_col else blank()
    out_cols["gps_hw_jamming_state"] = df[gps_hw_jamstate_col] if gps_hw_jamstate_col else blank()
    out_cols["gps_hw_spoof_state"] = df[gps_hw_spoof_col] if gps_hw_spoof_col else blank()

    # RTK status
    out_cols["rtk_gps_state"] = df[rtk_gpsstate_col] if rtk_gpsstate_col else blank()
    out_cols["rtk_hdop"] = df[rtk_hdop_col] if rtk_hdop_col else blank()
    for i, c in enumerate(rtk_posflg_cols):
        out_cols[f"rtk_posflg_{i}"] = df[c] if c else blank()

    # Motor/ESC/power
    for key, col in motor_speed_cols.items():
        out_cols[f"motor_speed_{key}"] = df[col] if col else blank()
    for key, col in motor_pwm_cols.items():
        out_cols[f"motor_pwm_{key}"] = df[col] if col else blank()
    for key, col in motor_ppm_cols.items():
        out_cols[f"motor_ppm_{key}"] = df[col] if col else blank()
    for key, col in motor_current_cols.items():
        out_cols[f"motor_current_{key}"] = df[col] if col else blank()
    for key, col in motor_volts_cols.items():
        out_cols[f"motor_volts_{key}"] = df[col] if col else blank()
    for key, col in motor_vout_cols.items():
        out_cols[f"motor_vout_{key}"] = df[col] if col else blank()
    for key, col in motor_esct_cols.items():
        out_cols[f"motor_esctemp_{key}"] = df[col] if col else blank()

    # Battery
    out_cols["battery_voltage"] = df[batt_voltage_col] if batt_voltage_col else blank()
    out_cols["battery_current"] = df[batt_current_col] if batt_current_col else blank()
    out_cols["battery_percent"] = df[batt_pct_col] if batt_pct_col else blank()
    out_cols["battery_temp"] = df[batt_temp_col] if batt_temp_col else blank()

    order = [
        "tick",
        "unix", "fractional_seconds",
        "gps_date", "gps_time", "gps_num_sats",
        "latitude", "longitude",
        "altitude", "relative_height",
        "roll", "pitch", "yaw",
        "controller_signal_level",
        "gps_hdop", "gps_pdop",
        "gps_num_gps", "gps_num_glnas", "gps_num_sv",
        "gps_hw_noiseperms", "gps_hw_agccnt", "gps_hw_jamind",
        "gps_hw_flag", "gps_hw_jamming_state", "gps_hw_spoof_state",
        "rtk_gps_state", "rtk_hdop",
        "rtk_posflg_0", "rtk_posflg_1", "rtk_posflg_2",
        "rtk_posflg_3", "rtk_posflg_4", "rtk_posflg_5",
        "motor_speed_rfront", "motor_speed_lfront", "motor_speed_lback", "motor_speed_rback",
        "motor_pwm_rfront", "motor_pwm_lfront", "motor_pwm_lback", "motor_pwm_rback",
        "motor_ppm_rfront", "motor_ppm_lfront", "motor_ppm_lback", "motor_ppm_rback",
        "motor_current_rfront", "motor_current_lfront", "motor_current_lback", "motor_current_rback",
        "motor_volts_rfront", "motor_volts_lfront", "motor_volts_lback", "motor_volts_rback",
        "motor_vout_rfront", "motor_vout_lfront", "motor_vout_lback", "motor_vout_rback",
        "motor_esctemp_rfront", "motor_esctemp_lfront", "motor_esctemp_lback", "motor_esctemp_rback",
        "battery_voltage", "battery_current", "battery_percent", "battery_temp",
    ]
    out = pd.DataFrame({k: out_cols[k] for k in order})

    # ---- 4) WRITE FILE ----
    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile} with columns: {', '.join(out.columns)}")

if __name__ == "__main__":
    main()
