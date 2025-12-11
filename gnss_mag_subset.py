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
      2) token fallback: every token in cand.lower().split(':') must be contained in col.lower()
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
    """
    Find the first column whose lowercased name contains all tokens in ANY of the token_lists (checked in order).
    Example: token_lists = [["gps","latitude"], ["latitude"]]
    """
    low = [c.lower() for c in cols]
    for tokens in token_lists:
        for i, name in enumerate(low):
            if all(tok in name for tok in tokens):
                return cols[i]
    return None

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Build a magnetometer subset CSV with GNSS-derived Unix timestamps.")
    parser.add_argument("input_csv", help="CSV exported from DatCon (e.g., 2025-12-09_17-30-39_FLY075.csv)")
    args = parser.parse_args()

    infile = args.input_csv
    outfile = _append_suffix(infile, "_gnss_mag")

    # Read as strings to preserve untouched formatting
    df = pd.read_csv(infile, dtype=str, keep_default_na=False)

    # ---- 1) TIME HANDLING (same logic you approved) ----
    # Required: GPS:Date and GPS:Time
    date_col = _find_name(df.columns, ["GPS:Date","gps:date","GPS:date"])
    time_col = _find_name(df.columns, ["GPS:Time","gps:time","GPS:time"])
    if not date_col or not time_col:
        raise ValueError("Required columns GPS:Date and/or GPS:Time not found.")

    # Optional tick column
    tick_col = _find_name(df.columns, ["Clock:Tick#","Clock:Tick","clock:tick#","clock:tick","Tick#","tick#","tick"])

    # Combine date + time → POSIX (UTC), integer seconds
    dt_str = (df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()).replace({"": np.nan})
    dt = pd.to_datetime(dt_str, utc=True, errors="coerce")
    unix_int = np.floor(dt.view("int64") / 1e9).astype("float")  # keep NaN for bad rows
    valid_unix = ~np.isnan(unix_int)

    # Fractional seconds from ticks
    frac = np.full(len(df), np.nan, dtype=float)
    if tick_col:
        tick = pd.to_numeric(df[tick_col], errors="coerce").to_numpy()

        # Group by each integer second (stable order)
        unix_series = pd.Series(unix_int)
        uSec = unix_series.dropna().astype(np.int64).drop_duplicates(keep="first")

        # Build maps of first/last index for each second
        first_idx = {}
        last_idx  = {}
        for sec in uSec:
            rows = np.where(unix_int == sec)[0]
            if rows.size > 0:
                first_idx[sec] = rows[0]
                last_idx[sec]  = rows[-1]

        # Compute per-second denominators (ticks per second)
        denoms = {}   # sec -> ticks per second
        for sec in uSec:
            i_first = first_idx.get(sec, None)
            i_last  = last_idx.get(sec, None)
            if i_first is None or i_last is None:
                continue
            # Prefer: first_of_(t+1) - first_of_t
            next_first = first_idx.get(sec+1, None)
            if next_first is not None and np.isfinite(tick[next_first]) and np.isfinite(tick[i_first]):
                d = tick[next_first] - tick[i_first]
            else:
                # Fallback: last_of_t - first_of_t
                if np.isfinite(tick[i_last]) and np.isfinite(tick[i_first]):
                    d = tick[i_last] - tick[i_first]
                else:
                    d = np.nan
            denoms[sec] = d if (np.isfinite(d) and d > 0) else np.nan

        # Global median for fallback
        denom_vals = np.array([v for v in denoms.values() if np.isfinite(v) and v > 0], dtype=float)
        denom_med = np.median(denom_vals) if denom_vals.size else np.nan

        # Compute fractional seconds for each second
        for sec in uSec:
            rows = np.where(unix_int == sec)[0]
            if rows.size == 0:
                continue
            i_first = first_idx.get(sec, None)
            if i_first is None or not np.isfinite(tick[i_first]):
                continue
            d = denoms.get(sec, np.nan)
            if not (np.isfinite(d) and d > 0):
                d = denom_med
            if not (np.isfinite(d) and d > 0):
                continue
            frac[rows] = (tick[rows] - tick[i_first]) / d

    # Clamp and round; default 0 if no tick
    frac = np.clip(frac, 0.0, 0.999999)
    frac6 = np.round(frac, 6)

    unix_str = pd.Series(unix_int).where(valid_unix, other=np.nan)
    unix_str = unix_str.dropna().astype("Int64").astype(str).reindex(range(len(df))).fillna("")
    frac_str = pd.Series(frac6).map(lambda v: f"{float(v):.6f}" if np.isfinite(v) else "0.000000")

    # ---- 2) PICK COLUMNS ----
    cols = list(df.columns)

    # Latitude / Longitude (prefer GPS:* first)
    lat_col = _find_first_containing(cols, [["gps","latitude"], ["latitude"], ["lat"]])
    lon_col = _find_first_containing(cols, [["gps","longitude"], ["longitude"], ["lon"]])

    # Absolute altitude: **force your requested column first**, then fallbacks
    alt_abs_col = (
        _find_name(cols, ["IMU_ATTI(0):alti:D[meters]"])  # explicit
        or _find_first_containing(cols, [["gps","alt"]])
        or _find_first_containing(cols, [["absoluteheight"]])
        or _find_first_containing(cols, [["alti"]])
        or _find_first_containing(cols, [["altitude"]])
        or _find_first_containing(cols, [["height"]])
    )

    # Relative height: look for common variants (DatCon/DJI)
    rel_h_col = (
        _find_first_containing(cols, [["relativeheight"]])
        or _find_first_containing(cols, [["relative","height"]])
        or _find_first_containing(cols, [["height","relative"]])
        or _find_first_containing(cols, [["height","takeoff"]])      # height above takeoff/home
        or _find_first_containing(cols, [["height","home"]])
        or _find_first_containing(cols, [["agl"]])                    # above ground level
        or _find_first_containing(cols, [["relative","alt"]])
        or _find_first_containing(cols, [["rel","height"]])
    )

    # Magnetometer axes (look for 'mag' + axis)
    magx_col = _find_first_containing(cols, [["mag","x"], ["magx"]])
    magy_col = _find_first_containing(cols, [["mag","y"], ["magy"]])
    magz_col = _find_first_containing(cols, [["mag","z"], ["magz"]])

    # GPS num sats (common names)
    sats_col = (
        _find_first_containing(cols, [["gps","numsats"]])
        or _find_first_containing(cols, [["numsats"]])
        or _find_first_containing(cols, [["satellites"]])
        or _find_first_containing(cols, [["num","sat"]])
    )

    # ---- 3) BUILD OUTPUT (tick first, then the rest) ----
    out_cols = {}

    # 1) tick first
    out_cols["tick"] = df[tick_col] if tick_col else pd.Series([""]*len(df))

    # 2) time fields
    out_cols["unix"] = unix_str
    out_cols["fractional_seconds"] = frac_str
    out_cols["gps_date"] = df[date_col]
    out_cols["gps_time"] = df[time_col]
    out_cols["gps_num_sats"] = df[sats_col] if sats_col else pd.Series([""]*len(df))

    # 3) position
    out_cols["latitude"]  = df[lat_col] if lat_col else pd.Series([""]*len(df))
    out_cols["longitude"] = df[lon_col] if lon_col else pd.Series([""]*len(df))

    # 4) altitudes (absolute + relative)
    out_cols["altitude"]        = df[alt_abs_col] if alt_abs_col else pd.Series([""]*len(df))
    out_cols["relative_height"] = df[rel_h_col]  if rel_h_col  else pd.Series([""]*len(df))

    # 5) magnetometer
    out_cols["mag_x"] = df[magx_col] if magx_col else pd.Series([""]*len(df))
    out_cols["mag_y"] = df[magy_col] if magy_col else pd.Series([""]*len(df))
    out_cols["mag_z"] = df[magz_col] if magz_col else pd.Series([""]*len(df))

    # Fixed output order with tick first
    order = [
        "tick",
        "unix", "fractional_seconds",
        "gps_date", "gps_time", "gps_num_sats",
        "latitude", "longitude",
        "altitude", "relative_height",
        "mag_x", "mag_y", "mag_z",
    ]

    out = pd.DataFrame({k: out_cols[k] for k in order})

    # ---- 4) WRITE FILE ----
    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile} with columns: {', '.join(out.columns)}")

if __name__ == "__main__":
    main()
