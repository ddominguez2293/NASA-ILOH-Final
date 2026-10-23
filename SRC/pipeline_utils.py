# pipeline_utils.py
# Shared config, paths and helpers for the pre-model pipeline (replaces Functions_Paths.R).
# Imported at the top of every SRC/ step with: from pipeline_utils import load_config, get_paths, ...
#
# Inputs  : config.yaml (repo root)
# Outputs : none directly; creates the data/<aoi_name>/ folder tree on first use
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml


## 2. Repository root and config
### Resolve the repo root from this file's location (SRC/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

REQUIRED_KEYS = [
    "aoi_name", "hucs", "start_date", "end_date",
    "streamorder_min", "snap_crs", "snap_max_distance_m", "era5",
    "analysis_cut_date", "trainval_streamorder_min", "model_split_date",
]


### Load and validate config.yaml
#### HUCs are kept as strings so leading zeros ("0712") survive
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing keys: {missing}")

    cfg["hucs"] = validate_hucs(cfg["hucs"])

    for k in ["start_date", "end_date", "analysis_cut_date", "model_split_date"]:
        cfg[k] = _as_date(cfg[k])
    if cfg["start_date"] > cfg["end_date"]:
        raise ValueError("start_date must be <= end_date")

    if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(cfg["aoi_name"])):
        raise ValueError("aoi_name may only contain letters, digits, '_' and '-'")

    cfg["config_path"] = str(path)
    return cfg


def _as_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


## 3. HUC helpers
### Validate HUC codes
#### WBD codes are 2, 4, 6, 8, 10 or 12 digits; only 2 / 4 / 8 are supported here
def validate_hucs(hucs) -> list[str]:
    if isinstance(hucs, (str, int)):
        hucs = [hucs]
    out = []
    for h in hucs:
        h = str(h).strip()
        if not re.fullmatch(r"\d+", h) or len(h) not in (2, 4, 8):
            raise ValueError(f"HUC '{h}' must be a 2-, 4- or 8-digit string (quote it in config.yaml)")
        out.append(h)
    if not out:
        raise ValueError("config.yaml 'hucs' is empty")
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate HUCs in config.yaml: {out}")
    return out


## 4. Paths
### Every output for one AOI lives under data/<aoi_name>/
#### model_input_dir is the fixed folder DDNN_Model.py reads from; the runner copies final tables there
@dataclass(frozen=True)
class Paths:
    data_dir: Path
    aoi_dir: Path
    shape_dir: Path
    nhd_dir: Path
    obs_dir: Path
    meteo_dir: Path
    drivers_dir: Path
    model_input_dir: Path

    ### Named files passed between steps
    @property
    def aoi_gpkg(self) -> Path:
        return self.shape_dir / "aoi.gpkg"

    @property
    def hu4_list(self) -> Path:
        return self.shape_dir / "hu4s.json"

    @property
    def q_obs(self) -> Path:
        return self.obs_dir / "q_obs.parquet"

    @property
    def q_t(self) -> Path:
        return self.obs_dir / "q_t.parquet"

    @property
    def nhd_cache_dir(self) -> Path:
        return self.data_dir / "_nhdplushr"  # raw HU4 zips, shared by every AOI

    @property
    def catchments_all(self) -> Path:
        return self.nhd_dir / "catchments_all.gpkg"

    @property
    def network(self) -> Path:
        return self.nhd_dir / "network.parquet"

    @property
    def flowlines(self) -> Path:
        return self.nhd_dir / "flowlines.gpkg"

    @property
    def catchments(self) -> Path:
        return self.nhd_dir / "catchments.gpkg"

    @property
    def sites_with_comid(self) -> Path:
        return self.obs_dir / "sites_with_comid.csv"

    @property
    def q_t_comid(self) -> Path:
        return self.obs_dir / "q_t_comid.parquet"

    @property
    def drivers_daily(self) -> Path:
        return self.drivers_dir / "drivers_daily"  # one <year>.parquet per year (Meteo.py)

    @property
    def era5_raw_dir(self) -> Path:
        return self.meteo_dir / "era5_raw"


def get_paths(cfg: dict, create: bool = True) -> Paths:
    data_dir = REPO_ROOT / "data"
    aoi_dir = data_dir / cfg["aoi_name"]
    p = Paths(
        data_dir=data_dir,
        aoi_dir=aoi_dir,
        shape_dir=aoi_dir / "shapes",
        nhd_dir=aoi_dir / "nhdplus",
        obs_dir=aoi_dir / "observations",
        meteo_dir=aoi_dir / "meteo",
        drivers_dir=aoi_dir / "daily_drivers",
        model_input_dir=data_dir / "daily_drivers",
    )
    if create:
        for d in [p.shape_dir, p.nhd_dir, p.obs_dir, p.meteo_dir, p.drivers_dir, p.model_input_dir]:
            d.mkdir(parents=True, exist_ok=True)
    return p


## 5. Stamps (skip steps whose inputs have not changed)
### Fingerprint the config keys a step depends on
#### A step is up to date only if its stamp matches AND every listed output exists,
#### so changing the HUCs or dates forces a rebuild instead of reusing stale files
def config_fingerprint(cfg: dict, keys: list[str]) -> str:
    subset = {k: cfg[k] for k in keys}
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _stamp_path(paths: Paths, step: str) -> Path:
    return paths.aoi_dir / ".stamps" / f"{step}.json"


def is_up_to_date(paths: Paths, step: str, fingerprint: str, outputs: list[Path]) -> bool:
    sp = _stamp_path(paths, step)
    if not sp.exists() or not all(Path(o).exists() for o in outputs):
        return False
    return json.loads(sp.read_text()).get("fingerprint") == fingerprint


def write_stamp(paths: Paths, step: str, fingerprint: str, outputs: list[Path]) -> None:
    sp = _stamp_path(paths, step)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({
        "step": step,
        "fingerprint": fingerprint,
        "written": datetime.now().isoformat(timespec="seconds"),
        "outputs": [str(o) for o in outputs],
    }, indent=2))


## 6. Small helpers
### Check that a step's inputs exist before doing any work
def require_files(files: dict[str, Path]) -> None:
    missing = [f"{label}: {path}" for label, path in files.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


### Print a step banner
def banner(step: str, cfg: dict) -> None:
    print(f"\n=== {step} | AOI={cfg['aoi_name']} | HUCs={','.join(cfg['hucs'])} "
          f"| {cfg['start_date']} to {cfg['end_date']} ===")


### Read the dissolved AOI written by Boundaries.py
def read_aoi(paths: Paths):
    import geopandas as gpd
    return gpd.read_file(paths.aoi_gpkg, layer="aoi", engine="pyogrio").to_crs(4326)


## 7. Upstream accumulation (Catchments.py, Meteo.py)
### Routing links from node topology: reach j drains to every reach whose fromnode == j's tonode
#### At a divergence the upstream total is split equally between the downstream branches, so
#### - a braided side channel inherits the upstream basin (basin MEANS are unchanged by the split)
#### - branches add back up where they rejoin, so no catchment is ever counted twice
#### (NHDPlus DnHydroSeq-only routing gives minor channels no upstream basin at all)
def routing_links(net):
    import pandas as pd

    links = (
        net[["comid", "tonode"]].rename(columns={"comid": "up_comid"})
        .merge(net[["comid", "fromnode"]].rename(columns={"comid": "dn_comid"}),
               left_on="tonode", right_on="fromnode")
        [["up_comid", "dn_comid"]]
    )
    links = links[links["up_comid"] != links["dn_comid"]]
    links["frac"] = 1.0 / links.groupby("up_comid")["dn_comid"].transform("size")
    return links.reset_index(drop=True)


### Main-path links: every reach drains only to its NHDPlus main-path downstream reach (DnHydroSeq)
#### Never loses area at a dead-end braid (a side channel that HR marks as terminal), but gives
#### minor channels no upstream basin. Catchments.py picks, per reach, whichever routing
#### accumulates closer to NHDPlus TotDASqKm (column basin_routing in network.parquet)
def main_path_links(net):
    links = net.loc[net["tocomid"] > 0, ["comid", "tocomid"]]
    links = links.rename(columns={"comid": "up_comid", "tocomid": "dn_comid"})
    return links.assign(frac=1.0).reset_index(drop=True)


ROUTINGS = {"split": routing_links, "main": main_path_links}


### Solve S = X + D S, where D[i, j] = frac of reach j's total passed to reach i
#### Usage: solve = upstream_solver(comid, hydroseq, links); S = solve(X)  (X: n or n x k)
def upstream_solver(comid, hydroseq, links):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.sparse.linalg import splu

    comid = np.asarray(comid, dtype=np.int64)
    n = len(comid)
    idx = pd.Index(comid)
    up = idx.get_indexer(links["up_comid"].to_numpy())
    dn = idx.get_indexer(links["dn_comid"].to_numpy())
    w = links["frac"].to_numpy(dtype=np.float64)

    ### Order upstream -> downstream so the system is triangular
    #### HydroSeq decreases downstream; fall back to a topological sort if any link disagrees
    order = np.argsort(-np.asarray(hydroseq, dtype=np.float64), kind="stable")
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    if np.any(pos[up] >= pos[dn]):
        order = _topological_order(n, up, dn)
        pos[order] = np.arange(n)

    A = sparse.identity(n, format="csc") - sparse.csc_matrix((w, (pos[dn], pos[up])), shape=(n, n))
    lu = splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0)

    def solve(X):
        X = np.asarray(X, dtype=np.float64)
        Xp = X[order]
        Sp = lu.solve(Xp)
        S = np.empty_like(Sp)
        S[order] = Sp
        return S

    return solve


### Kahn topological sort (upstream first); raises if the network has a loop
def _topological_order(n, up, dn):
    import numpy as np
    from collections import deque

    indeg = np.bincount(dn, minlength=n)
    children = [[] for _ in range(n)]
    for u, d in zip(up, dn):
        children[u].append(d)
    q = deque(np.flatnonzero(indeg == 0).tolist())
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for d in children[u]:
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    if len(order) != n:
        raise RuntimeError(f"Routing network has a loop ({n - len(order)} reaches unresolved)")
    return np.asarray(order, dtype=np.int64)


## 8. USGS daily values (Streamflow.py, Temp.py)
### Daily-mean time series for one parameter inside the AOI
#### Uses the USGS Water Data API (dataretrieval.waterdata); the legacy NWIS services are being retired
#### Sites are kept if they fall inside the AOI buffered by buffer_m, so outlet gauges on the boundary survive
#### Returns (values, sites): values = one row per (time_series_id, date); sites = one row per site with lon / lat
def usgs_daily_mean(parameter_code: str, aoi, start: date, end: date,
                    buffer_m: float = 1000, batch_size: int = 50):
    import geopandas as gpd
    import pandas as pd
    import dataretrieval.waterdata as wd

    ### Find daily-mean series overlapping the time window
    west, south, east, north = aoi.total_bounds
    meta, _ = wd.get_time_series_metadata(
        bbox=[west, south, east, north],
        parameter_code=parameter_code,
        computation_identifier="Mean",
        computation_period_identifier="Daily",
        skip_geometry=False,
    )
    if len(meta) == 0:
        raise RuntimeError(f"No daily-mean {parameter_code} series found in the AOI bbox")

    meta = gpd.GeoDataFrame(meta, geometry="geometry", crs=4326)
    meta = meta[(meta["end"].dt.date >= start) & (meta["begin"].dt.date <= end)]

    ### Clip to the buffered AOI polygon (meters, EPSG:5070)
    aoi_buf = aoi.to_crs(5070).buffer(buffer_m).to_crs(4326).union_all()
    meta = meta[meta.within(aoi_buf)]
    print(f"  {parameter_code}: {meta['monitoring_location_id'].nunique()} sites / "
          f"{len(meta)} daily-mean series inside the AOI and time window")

    ### Download in batches of time series
    ts_ids = meta["time_series_id"].tolist()
    window = f"{start.isoformat()}/{end.isoformat()}"
    frames = []
    for i in range(0, len(ts_ids), batch_size):
        batch = ts_ids[i:i + batch_size]
        d, _ = wd.get_daily(
            time_series_id=batch, time=window, skip_geometry=True,
            properties=["time_series_id", "monitoring_location_id", "time", "value",
                        "unit_of_measure", "approval_status"],
        )
        frames.append(d)
        print(f"    batch {i // batch_size + 1}/{-(-len(ts_ids) // batch_size)}: {len(d):,} rows")

    values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    values["site_no"] = values["monitoring_location_id"].str.replace(r"^USGS-", "", regex=True)
    values["Date"] = pd.to_datetime(values["time"]).dt.date
    values = values.drop(columns=["time"])

    ### One row per site with coordinates (used by Catchments.py for snapping)
    sites = (
        meta.drop_duplicates("monitoring_location_id")
        .assign(site_no=lambda g: g["monitoring_location_id"].str.replace(r"^USGS-", "", regex=True),
                lon=lambda g: g.geometry.x, lat=lambda g: g.geometry.y)
        [["site_no", "monitoring_location_id", "lon", "lat"]]
        .reset_index(drop=True)
    )
    return values, sites
