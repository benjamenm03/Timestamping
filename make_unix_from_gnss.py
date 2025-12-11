import pandas as pd
import numpy as np
from pathlib import Path

def _append_suffix(path_str, suffix):
    p = Path(path_str)
    return str(p.with_name(p.stem + suffix + p.suffix)) if p.suffix else str(p.with_name(p.name + suffix))

def _find_name(cols, candidates):
    # exact (case-insensitive)
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    # token fallback (e.g., "gps:date")
    low = [c.lower() for c in cols]
    for cand in candidates:
        toks = [t for t in cand.lower().split(":") if t]
        for i, name in enumerate(low):
            if all(tok in name for tok in toks):
                return cols[i]
    return None

def main():
    infile  = "datdefined.csv"
    outfile = _append_suffix(infile, "_with_unix")  # datdefined_with_unix.csv

    # Read as strings to preserve untouched columns
    df = pd.read_csv(infile, dtype=str, keep_default_na=False)

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

        # Compute per-second denominators
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

    # Replace Clock:offsetTime if present; else append
    cols = list(df.columns)
    if "Clock:offsetTime" in cols:
        idx = cols.index("Clock:offsetTime")
        new_cols = cols[:idx] + ["unix", "fractional_seconds"] + cols[idx+1:]
        out = pd.concat([df.iloc[:, :idx],
                         pd.DataFrame({"unix": unix_str, "fractional_seconds": frac_str}),
                         df.iloc[:, idx+1:]], axis=1)[new_cols]
    else:
        out = pd.concat([df, pd.DataFrame({"unix": unix_str, "fractional_seconds": frac_str})], axis=1)

    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile} (unix from GPS:Date/Time; fractional_seconds from ticks).")

if __name__ == "__main__":
    main()
