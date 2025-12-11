# Timestamping Workflow

This repository holds the CSV post-processing tools I use after converting DJI `.DAT` logs with **DatCon 4.3.0**. The overall flow is:

1. Pull the `.DAT` file from the drone.
2. Run the DatCon 4.3.0 executable (from the separate DatCon4.3.0 repo) to create a CSV export. The file name looks like `2025-12-09_17-30-39_FLY075.csv`. Launching DatCon from that repo: `java -jar DatCon.4.3.0.jar`.
3. Drop that CSV into this repository and process it with the Python scripts described below.

The scripts rely on Python 3 with `pandas` and `numpy` installed. On macOS you can set that up with:

```bash
python3 -m pip install --user pandas numpy matplotlib
```

## Scripts and usage

All scripts take the DatCon-generated CSV path as a required argument and emit new files by appending a suffix to the same base name.

### `add_unix_from_gnss.py`
- Purpose: Adds `unix` and `fractional_seconds` columns derived from the `GPS:Date`, `GPS:Time`, and optional `Clock:Tick#` fields while preserving every original column.
- Run on macOS Terminal:
  ```bash
  python3 add_unix_from_gnss.py 2025-12-09_17-30-39_FLY075.csv
  ```
  Output: `2025-12-09_17-30-39_FLY075_with_unix.csv`

### `gnss_attitude_subset.py`
- Purpose: Builds a tidy CSV containing GNSS time/position metadata plus roll, pitch, and yaw (degrees) for attitude analysis.
- Run:
  ```bash
  python3 gnss_attitude_subset.py 2025-12-09_17-30-39_FLY075.csv
  ```
  Output: `2025-12-09_17-30-39_FLY075_gnss_att.csv`

### `gnss_mag_subset.py`
- Purpose: Extracts GNSS time/position metadata along with magnetometer axes for magnetic-field investigations.
- Run:
  ```bash
  python3 gnss_mag_subset.py 2025-12-09_17-30-39_FLY075.csv
  ```
  Output: `2025-12-09_17-30-39_FLY075_gnss_mag.csv`

### `simulation.py`
- Purpose: Animates a flight path using the GNSS + attitude subset (`*_gnss_att.csv`). Update the `INPUT_CSV` constant to point at the file produced by `gnss_attitude_subset.py`, then run:
  ```bash
  python3 simulation.py
  ```
  Set `SAVE_MP4` inside the script if you want to capture the animation.

## Typical macOS workflow

```bash
cd /path/to/Timestamping
python3 gnss_attitude_subset.py 2025-12-09_17-30-39_FLY075.csv
python3 gnss_mag_subset.py 2025-12-09_17-30-39_FLY075.csv
python3 add_unix_from_gnss.py 2025-12-09_17-30-39_FLY075.csv
```

Each command produces a new CSV with GNSS-aligned Unix timestamps that can be committed or analyzed as needed. Replace the sample file name with the actual DatCon export you copied from the DatCon 4.3.0 run.
