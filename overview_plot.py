import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from simulation import load_flight, resample_uniform, plot_overview, INPUT_CSV, FPS


def _find_name(cols, candidates):
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    low = [c.lower() for c in cols]
    for cand in candidates:
        toks = [t for t in cand.lower().split(":") if t]
        for i, name in enumerate(low):
            if all(tok in name for tok in toks):
                return cols[i]
    return None


def _find_first_containing(cols, token_lists):
    low = [c.lower() for c in cols]
    for tokens in token_lists:
        for i, name in enumerate(low):
            if all(tok in name for tok in tokens):
                return cols[i]
    return None


def _series_as_float(df, col):
    if not col:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy()


def _make_rfi_plots(time_s, data, out_path, start_unix=None):
    if time_s.size == 0:
        return False
    t_rel = time_s - time_s[0]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    title = "Flight RFI Telemetry"
    if start_unix is not None and np.isfinite(start_unix):
        start_dt = pd.to_datetime(start_unix, unit="s", utc=True)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        title = f"Flight RFI Telemetry – Start UTC {start_str} (seconds since this instant)"
    fig.suptitle(title)

    def plot_or_note(ax, y, label, color="tab:blue"):
        if y is None or y.size != time_s.size or not np.any(np.isfinite(y)):
            ax.text(0.5, 0.5, f"Missing: {label}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        ax.plot(t_rel, y, color=color, linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # 1) Sats in view
    plot_or_note(axes[0], data.get("gps_num_sv"), "Satellites in View")

    # 2) GPS noise performance metric
    plot_or_note(axes[1], data.get("gps_hw_noiseperms"), "GPS NoisePerMS")

    # 3) GPS AGC
    plot_or_note(axes[2], data.get("gps_hw_agccnt"), "GPS AGC")

    # 4) Jam indicator + jamming state bands
    ax_jam = axes[3]
    jamind = data.get("gps_hw_jamind")
    jamstate = data.get("gps_hw_jamming_state")
    if jamind is None or jamind.size != time_s.size or not np.any(np.isfinite(jamind)):
        ax_jam.text(0.5, 0.5, "Missing: gps_hw_jamind", ha="center", va="center", transform=ax_jam.transAxes)
        ax_jam.set_axis_off()
    else:
        ax_jam.plot(t_rel, jamind, color="tab:red", linewidth=1.0)
        if jamstate is not None and jamstate.size == time_s.size and np.any(np.isfinite(jamstate)):
            state = np.nan_to_num(jamstate, nan=0.0)
            ax_jam.fill_between(
                t_rel, ax_jam.get_ylim()[0], ax_jam.get_ylim()[1],
                where=state >= 2, color="red", alpha=0.15, label="jamming state 2"
            )
            ax_jam.fill_between(
                t_rel, ax_jam.get_ylim()[0], ax_jam.get_ylim()[1],
                where=state == 1, color="gold", alpha=0.15, label="jamming state 1"
            )
        ax_jam.set_ylabel("Jamming Indicator")
        ax_jam.grid(True, alpha=0.3)

    # 5) Battery voltage
    ax_cur = axes[4]
    batt_v = data.get("battery_voltage")
    if batt_v is None or batt_v.size != time_s.size or not np.any(np.isfinite(batt_v)):
        ax_cur.text(0.5, 0.5, "Missing: battery_voltage", ha="center", va="center", transform=ax_cur.transAxes)
        ax_cur.set_axis_off()
    else:
        ax_cur.plot(t_rel, batt_v, color="tab:blue", linewidth=1.2)
        ax_cur.set_ylabel("Battery Voltage (V)")
        ax_cur.grid(True, alpha=0.3)

    # 6) Battery current
    ax_v = axes[5]
    batt_cur = data.get("battery_current")
    if batt_cur is None or batt_cur.size != time_s.size or not np.any(np.isfinite(batt_cur)):
        ax_v.text(0.5, 0.5, "Missing: battery_current", ha="center", va="center", transform=ax_v.transAxes)
        ax_v.set_axis_off()
    else:
        ax_v.plot(t_rel, batt_cur, color="tab:green", linewidth=1.2)
        ax_v.set_ylabel("Battery Current (A)")
        ax_v.grid(True, alpha=0.3)

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.tick_params(axis="x", labelbottom=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return True


def _make_motor_plots(time_s, data, out_path, start_unix=None):
    if time_s.size == 0:
        return False
    t_rel = time_s - time_s[0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    title = "Motor Telemetry"
    if start_unix is not None and np.isfinite(start_unix):
        start_dt = pd.to_datetime(start_unix, unit="s", utc=True)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        title = f"Motor Telemetry – Start UTC {start_str} (seconds since this instant)"
    fig.suptitle(title)

    def plot_group(ax, series, ylabel, missing_label, linewidth=0.8, prefix=""):
        if series is None or series.shape[0] != time_s.size:
            ax.text(0.5, 0.5, f"Missing: {missing_label}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        for name, _ in series.dtype.fields.items():
            short = name.replace(prefix, "") if prefix else name
            ax.plot(t_rel, series[name], label=short, linewidth=linewidth)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    plot_group(axes[0], data.get("motor_current"), "Motor Current (A)", "motor_current_*", prefix="motor_current_")
    plot_group(axes[1], data.get("motor_pwm"), "Motor PWM (%)", "motor_pwm_*", prefix="motor_pwm_")
    plot_group(axes[2], data.get("motor_speed"), "Motor Speed (RPM)", "motor_speed_*", prefix="motor_speed_")

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.tick_params(axis="x", labelbottom=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate the flight overview PNG (altitude/yaw/track) without animation.")
    parser.add_argument("input_csv", nargs="?", default=INPUT_CSV,
                        help="CSV file (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=FPS,
                        help="Resampling rate used for plotting (frames per second).")
    parser.add_argument("--output", type=str, default=None,
                        help="Override overview PNG path/name.")
    parser.add_argument("--rfi-output", type=str, default=None,
                        help="Override RFI PNG path/name.")
    parser.add_argument("--motor-output", type=str, default=None,
                        help="Override motor telemetry PNG path/name.")
    parser.add_argument("--show", action="store_true",
                        help="Show the figure window after saving.")
    args = parser.parse_args()

    t, lat, lon, alt_agl, alt_msl, roll, pitch, yaw = load_flight(args.input_csv)
    arrays = {
        "alt_msl": alt_msl,
        "yaw": yaw,
        "lat": lat,
        "lon": lon,
    }
    t_uni, arr = resample_uniform(t, arrays, fps=args.fps)
    fig = plot_overview(t_uni, arr, start_unix=float(t[0]) if t.size else None)
    if fig is None:
        raise SystemExit("No valid data to plot.")

    base = Path(args.input_csv)
    out_path = args.output or str(base.with_name(base.stem + "_overview.png"))
    # add time tick labels for overview plot
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="x", labelbottom=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved overview plot to {out_path}")

    # ---- RFI telemetry overview ----
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    unix_col = _find_name(cols, ["unix"])
    frac_col = _find_name(cols, ["fractional_seconds"])
    if unix_col and frac_col:
        t_unix = _series_as_float(df, unix_col)
        t_frac = _series_as_float(df, frac_col)
        time_s = t_unix + np.nan_to_num(t_frac, nan=0.0)
        time_s = np.where(time_s > 0.0, time_s, np.nan)
        valid = np.isfinite(time_s)
        time_s = time_s[valid]

        def col(name):
            return _series_as_float(df, _find_name(cols, [name]))[valid]

        gps_num_sv = _series_as_float(df, _find_name(cols, ["gps_num_sv"]))[valid]
        gps_noise = _series_as_float(df, _find_name(cols, ["gps_hw_noiseperms"]))[valid]
        gps_agc = _series_as_float(df, _find_name(cols, ["gps_hw_agccnt"]))[valid]
        gps_jamind = _series_as_float(df, _find_name(cols, ["gps_hw_jamind"]))[valid]
        gps_jamstate = _series_as_float(df, _find_name(cols, ["gps_hw_jamming_state"]))[valid]
        batt_v = _series_as_float(df, _find_name(cols, ["battery_voltage"]))[valid] / 1000.0
        batt_cur = _series_as_float(df, _find_name(cols, ["battery_current"]))[valid] / 1000.0

        # motor currents/volts
        mc_names = ["motor_current_rfront", "motor_current_lfront", "motor_current_lback", "motor_current_rback"]
        mp_names = ["motor_pwm_rfront", "motor_pwm_lfront", "motor_pwm_lback", "motor_pwm_rback"]
        ms_names = ["motor_speed_rfront", "motor_speed_lfront", "motor_speed_lback", "motor_speed_rback"]
        mc = np.zeros(time_s.size, dtype=[(n, "f8") for n in mc_names])
        mp = np.zeros(time_s.size, dtype=[(n, "f8") for n in mp_names])
        ms = np.zeros(time_s.size, dtype=[(n, "f8") for n in ms_names])
        for n in mc_names:
            mc[n] = _series_as_float(df, _find_name(cols, [n]))[valid]
        for n in mp_names:
            mp[n] = _series_as_float(df, _find_name(cols, [n]))[valid]
        for n in ms_names:
            ms[n] = _series_as_float(df, _find_name(cols, [n]))[valid]

        rfi_data = {
            "gps_num_sv": gps_num_sv,
            "gps_hw_noiseperms": gps_noise,
            "gps_hw_agccnt": gps_agc,
            "gps_hw_jamind": gps_jamind,
            "gps_hw_jamming_state": gps_jamstate,
            "battery_voltage": batt_v,
            "battery_current": batt_cur,
        }
        rfi_out = args.rfi_output or str(base.with_name(base.stem + "_rfi_overview.png"))
        ok = _make_rfi_plots(time_s, rfi_data, rfi_out, start_unix=float(time_s[0]) if time_s.size else None)
        if ok:
            print(f"Saved RFI overview plot to {rfi_out}")

        motor_data = {
            "motor_current": mc,
            "motor_pwm": mp,
            "motor_speed": ms,
        }
        motor_out = args.motor_output or str(base.with_name(base.stem + "_motor_overview.png"))
        ok2 = _make_motor_plots(time_s, motor_data, motor_out, start_unix=float(time_s[0]) if time_s.size else None)
        if ok2:
            print(f"Saved motor telemetry plot to {motor_out}")
    else:
        print("[note] Missing unix/fractional_seconds columns; skipping RFI overview plot.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
