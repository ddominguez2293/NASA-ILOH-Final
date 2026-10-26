#!/usr/bin/env python3

# run_pipeline.py
# Runs the pre-model pipeline (STEP 1-6) in order for the HUCs in config.yaml.
#
# Usage   : python run_pipeline.py                          # everything, skipping steps already up to date
#           python run_pipeline.py --config my_hucs.yaml    # a different HUC set / time window
#           python run_pipeline.py --from-step catchments   # restart part-way through
#           python run_pipeline.py --only meteo --force     # rebuild one step
# Outputs : everything under data/<aoi_name>/, plus the two model inputs copied to data/daily_drivers/.
#           Then train with: python SRC/DDNN_Model.py ...
#
# Each step decides on its own whether it is up to date (stamp in data/<aoi>/.stamps/), so a rerun
# after changing the HUCs or dates rebuilds exactly the steps whose inputs changed.
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "SRC"))
from pipeline_utils import load_config  # noqa: E402

## 2. Arguments
NAMES = ["boundaries", "streamflow", "temp", "catchments", "meteo", "q_processing"]
parser = argparse.ArgumentParser(description="Run the pre-model pipeline")
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--from-step", choices=NAMES, help="start at this step")
parser.add_argument("--only", choices=NAMES, nargs="+", help="run only these steps")
parser.add_argument("--force", action="store_true", help="rebuild the selected steps even if up to date")
args = parser.parse_args()

## 3. Steps in order
#### "meteo" dispatches to Meteo_gridmet.py or Meteo.py depending on config.yaml's met_source
#### (gridmet by default); both write identical output columns, so nothing downstream changes.
met_source = load_config(args.config).get("met_source", "gridmet")
if met_source not in ("gridmet", "era5"):
    raise ValueError(f"config.yaml met_source must be 'gridmet' or 'era5', got {met_source!r}")
meteo_script = "Meteo_gridmet.py" if met_source == "gridmet" else "Meteo.py"
print(f"Meteorology source: {met_source} ({meteo_script})")

STEPS = [
    ("boundaries", "Boundaries.py"),
    ("streamflow", "Streamflow.py"),
    ("temp", "Temp.py"),
    ("catchments", "Catchments.py"),
    ("meteo", meteo_script),
    ("q_processing", "Q_Processing_python.py"),
]

selected = STEPS
if args.from_step:
    selected = STEPS[NAMES.index(args.from_step):]
if args.only:
    selected = [s for s in STEPS if s[0] in args.only]

## 4. Run
### Each step runs as its own process from the repo root; stop at the first failure
total = time.time()
for name, script in selected:
    cmd = [sys.executable, str(REPO_ROOT / "SRC" / script)]
    if args.config:
        cmd += ["--config", str(Path(args.config).resolve())]
    if args.force:
        cmd.append("--force")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\nPIPELINE STOPPED: {name} failed (exit {result.returncode}). "
              f"Fix it and rerun with --from-step {name}.")
        sys.exit(result.returncode)
    print(f"--- {name} finished in {time.time() - t0:,.0f} s")

print(f"\nPipeline finished in {(time.time() - total) / 60:,.1f} min. "
      f"Model inputs are in data/daily_drivers/.")
