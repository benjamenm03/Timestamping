# Timestamping Workflow

This repository holds the CSV post-processing tools I use after converting DJI `.DAT` logs with **DatCon 4.3.0**. The overall flow is:

1. Pull the `.DAT` file from the drone.
2. Run the DatCon 4.3.0 executable (from the separate DatCon4.3.0 repo) to create a CSV export. The file name looks like `2025-12-09_17-30-39_FLY075.csv`. Launching DatCon from that repo: `java -jar DatCon.4.3.0.jar`.
3. Drop that CSV into this repository and process it with the Python scripts described below.

The scripts rely on Python 3 with `pandas` and `numpy` installed. On macOS you can set that up with:

```bash
python3 -m pip install --user pandas numpy matplotlib
```
If macOS reports that this Python is “externally managed” (PEP 668), create a virtual environment instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib
```
Run the scripts with `.venv/bin/python3` (or keep the shell activated) so they use the same interpreter that has the dependencies.

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
- Purpose: Animates a flight path using GNSS time/position and attitude data. Pass the CSV to animate (ideally the output of `gnss_attitude_subset.py`, which guarantees roll/pitch/yaw) or rely on the default `INPUT_CSV` constant.
  ```bash
  python3 simulation.py 2025-12-09_17-30-39_FLY075_gnss_att.csv
  ```
  If you omit the argument it uses the filename hard-coded near the top of the script. Add `--no-mp4` to skip exporting a movie or `--offset <seconds>` to change the start point. Using the `_with_unix` or `_gnss_mag` files also works, but they may lack clean attitude columns so the drone will level out automatically. Each run also saves a `<flight>_overview.png` containing three subplots: altitude MSL vs time, yaw vs time, and the lat/lon ground track.

## Typical macOS workflow

```bash
cd /path/to/Timestamping
python3 gnss_attitude_subset.py 2025-12-09_17-30-39_FLY075.csv
python3 gnss_mag_subset.py 2025-12-09_17-30-39_FLY075.csv
python3 add_unix_from_gnss.py 2025-12-09_17-30-39_FLY075.csv
```

Each command produces a new CSV with GNSS-aligned Unix timestamps that can be committed or analyzed as needed. Replace the sample file name with the actual DatCon export you copied from the DatCon 4.3.0 run.

## Running the simulation

1. Ensure the dependencies (`pandas`, `numpy`, `matplotlib`) are installed as noted above.
2. Generate the attitude subset for the flight you want to animate:
   ```bash
   python3 gnss_attitude_subset.py 2025-12-09_17-30-39_FLY075.csv
   ```
3. Launch the animation (replace the filename with your own):
   ```bash
   python3 simulation.py 2025-12-09_17-30-39_FLY075_gnss_att.csv
   ```
   Add `--offset 0` if you want it to start immediately, use `--no-mp4` to avoid saving a movie, or adjust `--fps` for faster/slower playback.
   If you set up a virtual environment, run the command as `.venv/bin/python3 simulation.py ...` (or activate the environment first) so it uses the interpreter with the installed packages.
