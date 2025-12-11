# animate_drone_3d.py
# Visualize a flight as a 3D animation (position + attitude).
# Requires: pandas, numpy, matplotlib (with ffmpeg installed only if SAVE_MP4=True)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# -------------------------- configuration -------------------------- #
INPUT_CSV = "2025-12-09_17-30-39_FLY075_gnss_att.csv"  # produced by gnss_attitude_subset.py
FPS = 20                               # playback fps
SAVE_MP4 = True                       # set True to save an MP4 next to the CSV
MP4_NAME = None                        # None => derive from INPUT_CSV with _anim.mp4
START_OFFSET_S = 930.0                 # <-- start the animation 5 minutes (300 s) in
# If your yaw is a compass heading (0°=North, +CW), leave True.
# If yaw is ENU yaw (0°=East, +CCW), set False.
YAW_IS_HEADING_CW_FROM_NORTH = True
# Drone drawing size (arm half-length in meters)
DRONE_SIZE_M = 30
AUTO_SCALE_DRONE = True             # scale drone arms to flight volume
DRONE_SIZE_SCALE = 0.05             # fraction of max span when auto-scaling
DRONE_SIZE_MIN = 1.0                # meters
# ------------------------------------------------------------------- #

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

def _find_angle_col(cols, primary_lists, avoid_tokens=("rate","gyro")):
    low = [c.lower() for c in cols]
    for tokens in primary_lists:
        for i, name in enumerate(low):
            if all(tok in name for tok in tokens) and not any(bad in name for bad in avoid_tokens):
                return cols[i]
    return None

def _series_as_float(df, col):
    if not col:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy()

def load_flight(csv_path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    cols = list(df.columns)

    unix_col = _find_name(cols, ["unix"])
    frac_col = _find_name(cols, ["fractional_seconds"])
    if not unix_col or not frac_col:
        raise ValueError("Input must contain 'unix' and 'fractional_seconds' columns. Run one of the GNSS scripts first.")

    lat_col = _find_first_containing(cols, [["latitude"], ["gps","latitude"], ["lat"]])
    lon_col = _find_first_containing(cols, [["longitude"], ["gps","longitude"], ["lon"]])
    alt_abs_col = (
        _find_name(cols, ["altitude"]) or
        _find_first_containing(cols, [["gps","alt"]]) or
        _find_first_containing(cols, [["height"]])
    )
    rel_h_col = (
        _find_first_containing(cols, [["relative","height"]]) or
        _find_first_containing(cols, [["relativeheight"]]) or
        _find_first_containing(cols, [["height","takeoff"]]) or
        _find_first_containing(cols, [["height","home"]])
    )

    roll_col  = _find_angle_col(cols, [["roll"], ["atti","roll"], ["imu","atti","roll"]])
    pitch_col = _find_angle_col(cols, [["pitch"], ["atti","pitch"], ["imu","atti","pitch"]])
    yaw_col   = _find_angle_col(cols, [["yaw"], ["atti","yaw"], ["imu","atti","yaw"]])

    # Time
    t_unix = _series_as_float(df, unix_col)
    t_frac = _series_as_float(df, frac_col)
    time_s = t_unix + np.nan_to_num(t_frac, nan=0.0)

    # Position
    lat = _series_as_float(df, lat_col)
    lon = _series_as_float(df, lon_col)
    alt_abs = _series_as_float(df, alt_abs_col)
    rel_h = _series_as_float(df, rel_h_col)
    alt = np.where(np.isfinite(alt_abs), alt_abs, rel_h)

    # Attitude (degrees); default to zeros if missing (mag subset)
    roll  = _series_as_float(df, roll_col)
    pitch = _series_as_float(df, pitch_col)
    yaw   = _series_as_float(df, yaw_col)
    if not np.any(np.isfinite(roll)):
        roll = np.zeros(len(df), dtype=float)
    if not np.any(np.isfinite(pitch)):
        pitch = np.zeros(len(df), dtype=float)
    if not np.any(np.isfinite(yaw)):
        yaw = np.zeros(len(df), dtype=float)

    m = np.isfinite(time_s) & np.isfinite(lat) & np.isfinite(lon)
    alt = np.where(np.isfinite(alt), alt, 0.0)

    filtered = [arr[m] for arr in (time_s, lat, lon, alt, roll, pitch, yaw)]
    time_s, lat, lon, alt, roll, pitch, yaw = filtered
    if time_s.size < 2:
        raise ValueError("Not enough valid rows with time + lat/lon to animate.")
    return time_s, lat, lon, alt, roll, pitch, yaw

# ---- WGS84 conversions (geodetic -> ECEF -> ENU) ---- #
def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)

    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    cos_lon = np.cos(lon); sin_lon = np.sin(lon)

    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    X = (N + h_m) * cos_lat * cos_lon
    Y = (N + h_m) * cos_lat * sin_lon
    Z = (N * (1 - e2) + h_m) * sin_lat
    return X, Y, Z

def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, h0_m):
    X0, Y0, Z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    dx, dy, dz = x - X0, y - Y0, z - Z0

    lat0 = np.deg2rad(lat0_deg); lon0 = np.deg2rad(lon0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)

    t = np.array([[-slon,        clon,       0],
                  [-sl*clon,    -sl*slon,    cl],
                  [ cl*clon,     cl*slon,    sl]], dtype=float)
    e = t[0,0]*dx + t[0,1]*dy + t[0,2]*dz
    n = t[1,0]*dx + t[1,1]*dy + t[1,2]*dz
    u = t[2,0]*dx + t[2,1]*dy + t[2,2]*dz
    return e, n, u

def geodetic_to_enu(lat_deg, lon_deg, h_m):
    i0 = np.where(np.isfinite(lat_deg) & np.isfinite(lon_deg))[0][0]
    lat0, lon0, h0 = lat_deg[i0], lon_deg[i0], float(np.nan_to_num(h_m[i0], nan=0.0))
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, np.nan_to_num(h_m, nan=0.0))
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, h0)
    return e, n, u, (lat0, lon0, h0)

# ---- attitude handling ---- #
def rpy_deg_to_body_axes(roll_deg, pitch_deg, yaw_deg):
    r = np.deg2rad(roll_deg); p = np.deg2rad(pitch_deg); y = np.deg2rad(yaw_deg)
    if YAW_IS_HEADING_CW_FROM_NORTH:
        y = np.deg2rad(90.0 - yaw_deg)  # heading(CW from N) -> ENU yaw(CCW from E)

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])
    return Rz @ Ry @ Rx  # world_from_body

# ---- resampling ---- #
def resample_uniform(time_s, arrays_dict, fps=20):
    t0, t1 = time_s[0], time_s[-1]
    dt = 1.0 / fps
    t_uni = np.arange(t0, t1, dt)

    out = {}
    for k, arr in arrays_dict.items():
        if k in ("roll", "pitch", "yaw"):
            out[k] = np.rad2deg(np.interp(t_uni, time_s, np.unwrap(np.deg2rad(arr))))
        else:
            out[k] = np.interp(t_uni, time_s, arr)
    return t_uni, out

# ---- drone geometry ---- #
def make_drone_geometry(size=0.4):
    L = size
    arms = {
        "arm1": np.array([[-L, 0, 0], [ L, 0, 0]], dtype=float),
        "arm2": np.array([[0, -L, 0], [0,  L, 0]], dtype=float),
    }
    upvec = np.array([[0,0,0],[0,0,L*1.2]], dtype=float)
    return arms, upvec

def transform_segment(seg_body, R, pos):
    return (R @ seg_body.T).T + pos

# ---- animation ---- #
def animate_flight(csv_path=INPUT_CSV, fps=FPS, save_mp4=SAVE_MP4, mp4_name=MP4_NAME, start_offset_s=START_OFFSET_S):
    # Load & prepare data
    t, lat, lon, alt, roll, pitch, yaw = load_flight(csv_path)
    e, n, u, origin = geodetic_to_enu(lat, lon, alt)
    arrays = {"e": e, "n": n, "u": u, "roll": roll, "pitch": pitch, "yaw": yaw}
    t_uni, arr = resample_uniform(t, arrays, fps=fps)

    # ----- start-at-offset trimming -----
    t_start = t_uni[0] + float(start_offset_s)
    i0 = int(np.searchsorted(t_uni, t_start, side="left"))
    if i0 >= len(t_uni) - 1:
        print(f"[note] Flight shorter than start offset ({start_offset_s}s). Starting near the end.")
        i0 = max(0, len(t_uni) - 2)
    # slice
    t_uni = t_uni[i0:]
    for k in arr.keys():
        arr[k] = arr[k][i0:]

    # Limits tightly hugging the flight path
    def _lims(data):
        dmin = float(np.min(data))
        dmax = float(np.max(data))
        if not np.isfinite(dmin) or not np.isfinite(dmax):
            dmin, dmax = -DRONE_SIZE_M, DRONE_SIZE_M
        if np.isclose(dmin, dmax):
            dmin -= DRONE_SIZE_M
            dmax += DRONE_SIZE_M
        return dmin, dmax

    xmin, xmax = _lims(arr["e"])
    ymin, ymax = _lims(arr["n"])
    zmin, zmax = _lims(arr["u"])

    span_e = xmax - xmin
    span_n = ymax - ymin
    span_u = zmax - zmin
    max_span = max(span_e, span_n, span_u)
    drone_size = DRONE_SIZE_M
    if AUTO_SCALE_DRONE and np.isfinite(max_span) and max_span > 0:
        drone_size = max(DRONE_SIZE_MIN, max_span * DRONE_SIZE_SCALE)

    # Geometry
    arms, upvec = make_drone_geometry(drone_size)

    # Figure
    plt.rcParams["toolbar"] = "toolmanager"
    fig = plt.figure("Drone 3D Animation", figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Up (m)")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax-xmin, ymax-ymin, max(1e-6, zmax-zmin)))

    path_line, = ax.plot([], [], [], lw=1)
    arm1_line, = ax.plot([], [], [], lw=3)
    arm2_line, = ax.plot([], [], [], lw=3)
    up_line,   = ax.plot([], [], [], lw=2)
    pt,        = ax.plot([], [], [], marker="o")
    title = ax.set_title("")

    path_xyz = np.column_stack([arr["e"], arr["n"], arr["u"]])

    def update(i):
        pos = np.array([arr["e"][i], arr["n"][i], arr["u"][i]])
        R = rpy_deg_to_body_axes(arr["roll"][i], arr["pitch"][i], arr["yaw"][i])

        a1 = transform_segment(arms["arm1"], R, pos)
        a2 = transform_segment(arms["arm2"], R, pos)
        up = transform_segment(upvec,        R, pos)

        arm1_line.set_data(a1[:,0], a1[:,1]); arm1_line.set_3d_properties(a1[:,2])
        arm2_line.set_data(a2[:,0], a2[:,1]); arm2_line.set_3d_properties(a2[:,2])
        up_line.set_data(up[:,0],   up[:,1]); up_line.set_3d_properties(up[:,2])
        pt.set_data([pos[0]], [pos[1]]); pt.set_3d_properties([pos[2]])

        path_line.set_data(path_xyz[:i+1,0], path_xyz[:i+1,1])
        path_line.set_3d_properties(path_xyz[:i+1,2])

        title.set_text(f"t = {t_uni[i]-t_uni[0]:.2f} s (offset {start_offset_s:.0f}s)")
        return arm1_line, arm2_line, up_line, pt, path_line, title

    frames = len(t_uni)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0/fps, blit=False, repeat=True)

    if save_mp4:
        out_name = mp4_name or str(Path(csv_path).with_name(Path(csv_path).stem + "_anim.mp4"))
        try:
            anim.save(out_name, writer="ffmpeg", fps=fps, dpi=150, )
            print(f"Saved animation to {out_name}")
        except Exception as e:
            print("Could not save MP4 (need ffmpeg installed). Showing live instead.", e)
            plt.show()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Animate a GNSS-derived flight CSV (from add_unix/gnss_* scripts).")
    parser.add_argument("input_csv", nargs="?", default=INPUT_CSV,
                        help="CSV file (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=FPS, help="Playback frames per second.")
    parser.add_argument("--offset", type=float, default=START_OFFSET_S, help="Start offset into the flight (seconds).")
    parser.add_argument("--no-mp4", action="store_true", help="Disable MP4 export even if enabled in config.")
    parser.add_argument("--mp4-name", type=str, default=None, help="Override MP4 output path/name.")
    args = parser.parse_args()

    animate_flight(
        csv_path=args.input_csv,
        fps=args.fps,
        save_mp4=False if args.no_mp4 else SAVE_MP4,
        mp4_name=args.mp4_name or MP4_NAME,
        start_offset_s=args.offset,
    )

if __name__ == "__main__":
    main()
