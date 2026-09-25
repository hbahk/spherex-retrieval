"""Local L2 archive index: one row per SPHEREx L2 file, built without opening a FITS file.

A local archive laid out like the SPHEREx production tree has two parts that
this module joins:

* ``loads/<load>/obscore/*.tbl`` — the IPAC ObsCore tables written at ingest,
  one row per product with its footprint (``s_region``), times and band;
* ``repo/level2/<week>/<procver>/<det>/level2_*.fits`` — the current L2 files,
  usually reached through per-week symbolic links into ``loads/``.

The ObsCore level-2 rows are joined to a listing of the level-2 tree by file
name. The join does three jobs at once: it keeps only files that exist (the
tables still list superseded processing versions), it drops files that several
loads list, and it finds the files on disk that no table lists (loads whose
``obscore/`` directory is empty). Those are indexed from their own headers,
which give the same numbers as the tables (:func:`header_metadata`). Last, an
exposure and detector keeps one file, its latest processing version
(:func:`procver_key`): reprocessed files sit next to the ones they replace.

The index is a parquet file (needs ``pyarrow``) with one row per file:

==================  =======  ================================================
obs_creator_did     str      e.g. ``2025W17_4B_0001_1D1`` (unique)
obs_id              str      e.g. ``2025W17_4B_0001_1``
week                str      e.g. ``2025W17_4B``
expo                int16    exposure number
dither              int8
detector            int8     1..6
procver             str      e.g. ``l2b-v19-2025-241``
path                str      relative to the archive root
size                int64    bytes (the MEF layout follows from it)
s_ra, s_dec         float64  frame centre, deg
c1_ra ... c4_dec    float64  the four ``s_region`` corners, deg
t_min, t_max        float64  MJD
em_min, em_max      float64  m
load                str      load holding the file (whose ObsCore table listed it)
meta                str      ``obscore`` or ``header``: where the metadata came from
==================  =======  ================================================

Rows are in path order (directory order: the order that reads the archive
sequentially).

Command line: ``spherex-index build <archive_root> -o index.parquet`` and
``spherex-index info index.parquet``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

#: ``level2_<week>_<expo>_<dither>D<det>_spx_<procver>.fits``
LEVEL2_NAME_RE = re.compile(
    r"^level2_(?P<week>\d{4}W\d{2}_\d[A-Z])_(?P<expo>\d{4})_(?P<dither>\d)"
    r"D(?P<det>\d)_spx_(?P<procver>.+)\.fits$"
)

_PROCVER_RE = re.compile(r"v(\d+)-(\d{4})-(\d{3})$")


def procver_key(procver: str) -> int:
    """Order processing versions: ``l2b-v20-2025-269`` > ``l2b-v20-2025-267``.

    Pipeline version, then processing year and day; ``l2b_retry-v26-2026-197``
    ranks by the same numbers. Unparseable versions rank lowest.
    """
    m = _PROCVER_RE.search(str(procver))
    if not m:
        return -1
    return int(m.group(1)) * 10_000_000 + int(m.group(2)) * 1_000 + int(m.group(3))


#: ObsCore columns the index needs.
OBSCORE_COLUMNS = ("calib_level", "obs_id", "obs_creator_did", "s_ra", "s_dec",
                   "t_min", "t_max", "em_min", "em_max", "s_region", "access_url")

INDEX_METADATA_KEY = b"spherex_index"

_CORNER_COLUMNS = tuple(f"c{i}_{ax}" for i in range(1, 5) for ax in ("ra", "dec"))


# --------------------------------------------------------------------------- #
# IPAC ObsCore tables
# --------------------------------------------------------------------------- #

def read_ipac_columns(path: str | Path,
                      columns: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Read columns of a fixed-width IPAC table as stripped ``bytes`` arrays.

    Column boundaries come from the ``|`` positions of the first header row,
    so values holding spaces (``s_region``) stay whole. Data rows of one length
    are sliced as a single 2-D byte array; ragged rows fall back to per-line
    slicing. Values are returned undecoded; see :func:`_to_float`.
    """
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    i = 0
    header_rows = []
    while i < len(lines) and lines[i][:1] in (b"\\", b"|"):
        if lines[i][:1] == b"|":
            header_rows.append(lines[i].rstrip(b"\r"))
        i += 1
    if not header_rows:
        raise ValueError(f"{path}: no IPAC column header")
    names_row = header_rows[0]
    bars = [k for k, c in enumerate(names_row) if c == ord("|")]
    spans = {names_row[a + 1:b].strip().decode(): (a, b) for a, b in zip(bars[:-1], bars[1:])}
    wanted = tuple(spans) if columns is None else columns
    missing = [c for c in wanted if c not in spans]
    if missing:
        raise KeyError(f"{path}: columns not in table: {missing}")

    rows = [ln.rstrip(b"\r") for ln in lines[i:]]
    rows = [ln for ln in rows if ln.strip()]
    out: dict[str, np.ndarray] = {}
    if not rows:
        return {c: np.array([], dtype="S1") for c in wanted}
    width = len(rows[0])
    if all(len(r) == width for r in rows):
        block = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), width)
        for c in wanted:
            a, b = spans[c]
            b = min(b, width)
            cell = np.ascontiguousarray(block[:, a:b]).view(f"S{b - a}").ravel()
            out[c] = np.char.strip(cell)
    else:
        for c in wanted:
            a, b = spans[c]
            out[c] = np.array([r[a:b].strip() for r in rows])
    return out


def _to_float(values: np.ndarray) -> np.ndarray:
    """``bytes`` cells to float64; IPAC ``null`` and blanks become NaN."""
    values = np.asarray(values)
    bad = (values == b"null") | (values == b"")
    clean = np.where(bad, b"nan", values)
    return clean.astype(np.float64)


def parse_s_region(values: np.ndarray) -> np.ndarray:
    """``POLYGON ICRS ra1 dec1 ... ra4 dec4`` cells to an ``(n, 4, 2)`` array.

    The production tables pad the polygon with runs of spaces, so the value is
    split on whitespace. Polygons with other than four vertices raise.
    """
    corners = np.full((len(values), 4, 2), np.nan)
    for k, v in enumerate(values):
        tok = (v.decode() if isinstance(v, bytes) else str(v)).split()
        if len(tok) != 10 or tok[0].upper() != "POLYGON":
            raise ValueError(f"not a 4-vertex POLYGON: {v!r}")
        corners[k] = np.asarray(tok[2:], dtype=np.float64).reshape(4, 2)
    return corners


def read_obscore_level2(path: str | Path) -> dict[str, np.ndarray]:
    """Level-2 rows of one ObsCore table: file ``name`` plus parsed columns."""
    cols = read_ipac_columns(path, OBSCORE_COLUMNS)
    keep = cols["calib_level"] == b"2"
    cols = {k: v[keep] for k, v in cols.items()}
    urls = cols["access_url"]
    names = np.array([u.rsplit(b"/", 1)[-1].decode() for u in urls], dtype=object)
    out = {
        "name": names,
        "obs_id": np.array([v.decode() for v in cols["obs_id"]], dtype=object),
        "obs_creator_did": np.array([v.decode() for v in cols["obs_creator_did"]], dtype=object),
        "corners": parse_s_region(cols["s_region"]),
    }
    for c in ("s_ra", "s_dec", "t_min", "t_max", "em_min", "em_max"):
        out[c] = _to_float(cols[c])
    return out


# --------------------------------------------------------------------------- #
# The level-2 tree on disk
# --------------------------------------------------------------------------- #

def list_level2_tree(level2_dir: str | Path, *, workers: int = 16) -> dict[str, tuple[str, int]]:
    """Map each ``level2_*.fits`` name under ``level2_dir`` to ``(relpath, size)``.

    ``relpath`` is ``<week>/<procver>/<det>/<name>`` relative to ``level2_dir``.
    Week directories may be symbolic links; they are followed. Directory scans
    run on ``workers`` threads (the cost is file-system metadata, not CPU).
    """
    level2_dir = Path(level2_dir)
    det_dirs = []
    for week in sorted(os.listdir(level2_dir)):
        wdir = level2_dir / week
        if not wdir.is_dir():
            continue
        for pv in sorted(os.listdir(wdir)):
            pdir = wdir / pv
            if not pdir.is_dir():
                continue
            for det in sorted(os.listdir(pdir)):
                if (pdir / det).is_dir():
                    det_dirs.append(f"{week}/{pv}/{det}")

    def scan(rel: str) -> list[tuple[str, str, int]]:
        found = []
        with os.scandir(level2_dir / rel) as it:
            for e in it:
                if e.name.startswith("level2_") and e.name.endswith(".fits"):
                    found.append((e.name, f"{rel}/{e.name}", e.stat().st_size))
        return found

    listing: dict[str, tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for found in ex.map(scan, det_dirs):
            for name, rel, size in found:
                listing[name] = (rel, size)
    return listing


# --------------------------------------------------------------------------- #
# Build / read
# --------------------------------------------------------------------------- #

#: SPHEREx detector bands (m), as the ObsCore tables give them.
DETECTOR_BAND_M = {1: (7.455e-07, 1.1167e-06), 2: (1.1033e-06, 1.6499e-06),
                   3: (1.6301e-06, 2.4346e-06), 4: (2.4027e-06, 3.8473e-06),
                   5: (3.8113e-06, 4.43e-06), 6: (4.4115e-06, 5.0096e-06)}


def header_metadata(path: str | Path) -> dict:
    """ObsCore-equivalent metadata of one L2 file, from its PRIMARY and IMAGE headers.

    Reproduces the production ObsCore tables exactly (checked on indexed QR2
    files): ``s_ra``/``s_dec`` are the SIP world position of 0-based pixel
    ``((NAXIS1-1)/2, (NAXIS2-1)/2)``, the corners those of the outer pixel
    edges, ``t_min``/``t_max`` are ``MJD-BEG``/``MJD-END``, and the band is
    the detector's.
    """
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning

    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
        primary = hdul[0].header
        image = hdul["IMAGE"].header
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(image)
    nx, ny = int(image["NAXIS1"]), int(image["NAXIS2"])
    centre = wcs.all_pix2world([[(nx - 1) / 2.0, (ny - 1) / 2.0]], 0)[0]
    corners = wcs.all_pix2world([[-0.5, -0.5], [nx - 0.5, -0.5],
                                 [nx - 0.5, ny - 0.5], [-0.5, ny - 0.5]], 0)
    m = LEVEL2_NAME_RE.match(Path(path).name)
    det = int(m["det"]) if m else int(image.get("DETECTOR", -1))

    def mjd(key):
        val = primary.get(key, image.get(key))
        return float(val) if val is not None else np.nan

    em = DETECTOR_BAND_M.get(det, (np.nan, np.nan))
    return {"s_ra": float(centre[0]), "s_dec": float(centre[1]), "corners": np.asarray(corners),
            "t_min": mjd("MJD-BEG"), "t_max": mjd("MJD-END"), "em_min": em[0], "em_max": em[1]}


def _header_rows_worker(paths: list[str]) -> list[tuple[str, dict | None, str]]:
    out = []
    for path in paths:
        try:
            out.append((path, header_metadata(path), ""))
        except Exception as exc:  # noqa: BLE001 - reported, the file stays unindexed
            out.append((path, None, f"{type(exc).__name__}: {exc}"))
    return out


def scan_headers(paths: list[str], *, workers: int = 16, chunk: int = 256):
    """:func:`header_metadata` over many files in ``workers`` processes.

    Returns ``(results, failures)``: ``{path: metadata}`` and ``{path: error}``.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    batches = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
    results, failures = {}, {}
    if not batches:
        return results, failures
    if workers <= 1:
        done = map(_header_rows_worker, batches)
        for batch in done:
            for path, meta, err in batch:
                (results.__setitem__(path, meta) if meta is not None
                 else failures.__setitem__(path, err))
        return results, failures
    # spawn, not fork: the caller may already run threads (the directory scan)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        for batch in ex.map(_header_rows_worker, batches):
            for path, meta, err in batch:
                (results.__setitem__(path, meta) if meta is not None
                 else failures.__setitem__(path, err))
    return results, failures


def _load_of(week_dir: Path) -> str:
    """The load a week directory lives in (``.../loads/<load>/data/level2/<week>``), or ''."""
    parts = week_dir.resolve().parts
    return parts[parts.index("loads") + 1] if "loads" in parts[:-1] else ""


def build_index(archive_root: str | Path, *, obscore_glob: str = "loads/*/obscore/*.tbl",
                level2_dir: str = "repo/level2", workers: int = 16, release: str | None = None,
                headers: bool = True, log=None) -> tuple["pyarrow.Table", dict]:  # noqa: F821
    """Index every L2 file under ``archive_root/level2_dir``.

    Metadata come from the ObsCore tables where a file has a level-2 row,
    else (``headers=True``) from the file's own headers
    (:func:`header_metadata`, the same numbers the tables hold); the ``meta``
    column says which. Returns ``(table, report)``; ``report`` counts what the
    ObsCore join dropped: ``stale_rows`` (listed but not on disk, e.g.
    superseded processing versions), ``duplicate_rows`` (a file listed by more
    than one table; the first table in sorted order wins), ``header_rows``,
    ``superseded`` (files of an exposure and detector that a later processing
    version replaced, both on disk: only the latest is indexed) and
    ``unindexed`` (files left without metadata, with ``header_failures``).
    ``release`` (e.g. ``"qr2"``) is recorded for the ``local`` query backend,
    which labels the frames ``spherex_<release>`` (it decides the PSF kind and
    the calibration release downstream).
    """
    pa = _require_pyarrow()
    root = Path(archive_root)
    say = log or (lambda msg: None)
    rel_level2 = Path(level2_dir).as_posix().strip("/")

    t0 = time.time()
    listing = list_level2_tree(root / level2_dir, workers=workers)
    say(f"listed {len(listing):,} level-2 files under {level2_dir} in {time.time() - t0:.0f} s")

    tables = sorted(root.glob(obscore_glob))
    recs: dict[str, list] = {k: [] for k in ("name", "obs_id", "obs_creator_did", "s_ra", "s_dec",
                                             "corners", "t_min", "t_max", "em_min", "em_max",
                                             "load", "meta")}
    n_rows, n_stale, n_dup = 0, 0, 0
    seen: set[str] = set()
    t1 = time.time()
    for k, tbl in enumerate(tables):
        rows = read_obscore_level2(tbl)
        n_rows += len(rows["name"])
        on_disk = np.array([n in listing for n in rows["name"]], dtype=bool)
        n_stale += int((~on_disk).sum())
        fresh = np.array([n not in seen for n in rows["name"]], dtype=bool)
        # a table can list the same file twice as well as repeat another table
        _, first = np.unique(rows["name"].astype(str), return_index=True)
        once = np.zeros(len(rows["name"]), dtype=bool)
        once[first] = True
        keep = on_disk & fresh & once
        n_dup += int((on_disk & ~(fresh & once)).sum())
        seen.update(rows["name"][keep])
        load = tbl.parent.parent.name
        for c in ("name", "obs_id", "obs_creator_did", "s_ra", "s_dec", "corners",
                  "t_min", "t_max", "em_min", "em_max"):
            recs[c].append(rows[c][keep])
        recs["load"].append(np.full(int(keep.sum()), load, dtype=object))
        recs["meta"].append(np.full(int(keep.sum()), "obscore", dtype=object))
        say(f"[{k + 1}/{len(tables)}] {load}: {len(rows['name']):,} level-2 rows, "
            f"{int(keep.sum()):,} kept")
    say(f"read {len(tables)} ObsCore tables in {time.time() - t1:.0f} s")

    missing = sorted(n for n in listing if n not in seen)
    failures: dict[str, str] = {}
    n_header = 0
    if headers and missing:
        t2 = time.time()
        paths = [str(root / rel_level2 / listing[n][0]) for n in missing]
        found, failures = scan_headers(paths, workers=workers)
        names_h = [n for n, p in zip(missing, paths) if p in found]
        metas = [found[str(root / rel_level2 / listing[n][0])] for n in names_h]
        n_header = len(names_h)
        loads_by_week: dict[str, str] = {}
        for n in names_h:
            week = listing[n][0].split("/", 1)[0]
            if week not in loads_by_week:
                loads_by_week[week] = _load_of(root / rel_level2 / week)
        if names_h:
            ids = [LEVEL2_NAME_RE.match(n) for n in names_h]
            recs["name"].append(np.array(names_h, dtype=object))
            recs["obs_creator_did"].append(np.array(
                [f"{m['week']}_{m['expo']}_{m['dither']}D{m['det']}" for m in ids], dtype=object))
            recs["obs_id"].append(np.array(
                [f"{m['week']}_{m['expo']}_{m['dither']}" for m in ids], dtype=object))
            for c in ("s_ra", "s_dec", "t_min", "t_max", "em_min", "em_max"):
                recs[c].append(np.array([md[c] for md in metas], dtype=np.float64))
            recs["corners"].append(np.stack([md["corners"] for md in metas]))
            recs["load"].append(np.array([loads_by_week[listing[n][0].split("/", 1)[0]]
                                          for n in names_h], dtype=object))
            recs["meta"].append(np.full(n_header, "header", dtype=object))
            seen.update(names_h)
        say(f"read the headers of {len(paths):,} files without ObsCore rows in "
            f"{time.time() - t2:.0f} s ({n_header:,} indexed, {len(failures):,} failed)")

    def cat(key, empty):
        return np.concatenate(recs[key]) if recs[key] else empty

    names = cat("name", np.array([], dtype=object))
    corners = cat("corners", np.zeros((0, 4, 2)))
    parsed = [LEVEL2_NAME_RE.match(n) for n in names]
    bad = [n for n, m in zip(names, parsed) if m is None]
    if bad:
        raise ValueError(f"{len(bad)} level-2 names do not parse, e.g. {bad[:3]}")
    empty_f = np.array([], dtype=np.float64)
    columns = {
        "obs_creator_did": pa.array(cat("obs_creator_did", np.array([], dtype=object)).astype(str)),
        "obs_id": pa.array(cat("obs_id", np.array([], dtype=object)).astype(str)),
        "week": pa.array([m["week"] for m in parsed], type=pa.string()),
        "expo": pa.array([int(m["expo"]) for m in parsed], type=pa.int16()),
        "dither": pa.array([int(m["dither"]) for m in parsed], type=pa.int8()),
        "detector": pa.array([int(m["det"]) for m in parsed], type=pa.int8()),
        "procver": pa.array([m["procver"] for m in parsed], type=pa.string()),
        "path": pa.array([f"{rel_level2}/{listing[n][0]}" for n in names], type=pa.string()),
        "size": pa.array([listing[n][1] for n in names], type=pa.int64()),
        "s_ra": pa.array(cat("s_ra", empty_f), type=pa.float64()),
        "s_dec": pa.array(cat("s_dec", empty_f), type=pa.float64()),
    }
    for j, name in enumerate(_CORNER_COLUMNS):
        columns[name] = pa.array(corners[:, j // 2, j % 2], type=pa.float64())
    for c in ("t_min", "t_max", "em_min", "em_max"):
        columns[c] = pa.array(cat(c, empty_f), type=pa.float64())
    columns["load"] = pa.array(cat("load", np.array([], dtype=object)).astype(str), type=pa.string())
    columns["meta"] = pa.array(cat("meta", np.array([], dtype=object)).astype(str), type=pa.string())
    table = pa.table(columns)
    # one file per exposure and detector: a reprocessing supersedes the file it
    # replaces even when both are still on disk (and only one has ObsCore rows)
    did = np.asarray(table.column("obs_creator_did").to_pylist(), dtype=object)
    pv_key = np.array([procver_key(v) for v in table.column("procver").to_pylist()], dtype=np.int64)
    order = np.lexsort((pv_key, did.astype(str)))
    last = np.ones(len(order), dtype=bool)
    last[:-1] = did[order][1:] != did[order][:-1]
    superseded = sorted(table.column("path").take(order[~last]).to_pylist())
    table = table.take(np.sort(order[last]))
    # directory order: the order that reads the archive sequentially
    table = table.take(np.argsort(np.asarray(table.column("path").to_pylist(), dtype=object),
                                  kind="stable"))

    unindexed = sorted(f"{rel_level2}/{rel}" for n, (rel, _) in listing.items() if n not in seen)
    report = {
        "archive_root": str(root.resolve()),
        "release": release,
        "level2_dir": rel_level2,
        "obscore_glob": obscore_glob,
        "obscore_tables": len(tables),
        "files_listed": len(listing),
        "obscore_level2_rows": n_rows,
        "indexed": table.num_rows,
        "stale_rows": n_stale,
        "duplicate_rows": n_dup,
        "header_rows": n_header,
        "superseded": superseded,
        "header_failures": failures,
        "unindexed": unindexed,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return table, report


def write_index(table, path: str | Path, report: dict | None = None) -> Path:
    """Write the index parquet; ``report`` (minus the unindexed list) goes in its metadata."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    meta = dict(table.schema.metadata or {})
    if report is not None:
        meta[INDEX_METADATA_KEY] = json.dumps(_summary(report)).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")
    return path


def _summary(report: dict) -> dict:
    """The report without its long lists: counts plus a few examples."""
    out = {k: v for k, v in report.items() if k not in ("unindexed", "header_failures", "superseded")}
    out["unindexed_count"] = len(report.get("unindexed", []))
    out["superseded_count"] = len(report.get("superseded", []))
    out["superseded_examples"] = list(report.get("superseded", []))[:10]
    failures = report.get("header_failures", {})
    out["header_failure_count"] = len(failures)
    out["header_failure_examples"] = dict(list(failures.items())[:10])
    return out


def read_index(path: str | Path, columns: list[str] | None = None):
    """Read an index parquet as a ``pyarrow.Table``."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns)


def index_metadata(table_or_path) -> dict:
    """The build summary stored by :func:`write_index` (empty if absent)."""
    if isinstance(table_or_path, (str, Path)):
        _require_pyarrow()
        import pyarrow.parquet as pq

        meta = pq.read_schema(table_or_path).metadata or {}
    else:
        meta = table_or_path.schema.metadata or {}
    raw = meta.get(INDEX_METADATA_KEY)
    return json.loads(raw) if raw else {}


def index_corners(table) -> np.ndarray:
    """The ``s_region`` corners of an index table as an ``(n, 4, 2)`` array (deg)."""
    cols = [np.asarray(table.column(c).to_numpy(), dtype=np.float64) for c in _CORNER_COLUMNS]
    return np.stack(cols, axis=1).reshape(-1, 4, 2)


def _require_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("the local-archive index needs pyarrow: "
                          "pip install 'spherex-retrieval[local]'") from exc
    return pa


# --------------------------------------------------------------------------- #
# Overlap: which frames cover which targets
# --------------------------------------------------------------------------- #

#: SPHEREx L2 pixel scale used to turn ``margin_pix`` into an angle.
PIXEL_SCALE_ARCSEC = 6.15


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def _tangent_basis(ra_deg, dec_deg) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at (ra, dec)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    east = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=-1)
    north = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], axis=-1)
    return east, north


def _frame_planes(table) -> dict[str, np.ndarray]:
    """Per frame: centre, tangent basis and the footprint corners in the gnomonic plane.

    The corners are re-ordered counter-clockwise around their centroid, which
    makes the inside test independent of the ``s_region`` winding (it is not
    consistent across the archive) and of a crossing vertex order.
    """
    ra0 = np.asarray(table.column("s_ra").to_numpy(), dtype=np.float64)
    dec0 = np.asarray(table.column("s_dec").to_numpy(), dtype=np.float64)
    centre = _unit_vectors(ra0, dec0)
    east, north = _tangent_basis(ra0, dec0)
    corners = index_corners(table)
    v = _unit_vectors(corners[..., 0], corners[..., 1])            # (M, 4, 3)
    w = np.einsum("mkj,mj->mk", v, centre)
    xi = np.einsum("mkj,mj->mk", v, east) / w
    eta = np.einsum("mkj,mj->mk", v, north) / w
    ang = np.arctan2(eta - eta.mean(axis=1, keepdims=True), xi - xi.mean(axis=1, keepdims=True))
    order = np.argsort(ang, axis=1)
    xi = np.take_along_axis(xi, order, axis=1)
    eta = np.take_along_axis(eta, order, axis=1)
    return {"centre": centre, "east": east, "north": north, "xi": xi, "eta": eta}


def _inside_with_margin(p: np.ndarray, planes: dict, frames: np.ndarray, margin_rad: float) -> np.ndarray:
    """Is each target ``p[i]`` within ``margin_rad`` of (or inside) frame ``frames[i]``?"""
    c = planes["centre"][frames]
    w = np.einsum("ij,ij->i", p, c)
    front = w > 0
    w = np.where(front, w, 1.0)
    px = np.einsum("ij,ij->i", p, planes["east"][frames]) / w
    py = np.einsum("ij,ij->i", p, planes["north"][frames]) / w
    x0, y0 = planes["xi"][frames], planes["eta"][frames]          # (n, 4), counter-clockwise
    x1, y1 = np.roll(x0, -1, axis=1), np.roll(y0, -1, axis=1)
    ex, ey = x1 - x0, y1 - y0
    # signed distance of p from each edge line, positive on the inner side
    d = (ex * (py[:, None] - y0) - ey * (px[:, None] - x0)) / np.hypot(ex, ey)
    return front & np.all(d >= -np.tan(margin_rad), axis=1)


def find_overlapping_many(ra, dec, size_arcsec: float, index, *, margin_pix: float = 10.0,
                          cone_radius_deg: float = 2.6, chunk: int = 4096):
    """Pair every target with the index rows (frames) whose footprint can hold its box.

    Two generous stages (a missed frame loses data for good; a spare one costs
    one WCS check when the frame is opened):

    1. cone: frame centres within ``cone_radius_deg`` of the target (KD-tree on
       unit vectors; a frame's half-diagonal is about 2.45 deg);
    2. polygon: the target inside the ``s_region`` quadrilateral grown by half
       the box diagonal plus ``margin_pix`` pixels, tested in the gnomonic
       plane about the frame centre, where the great-circle edges are straight.

    The exact test is the frame's own WCS (with SIP) at extraction time.

    Parameters
    ----------
    ra, dec : array_like
        Target positions, deg (ICRS).
    size_arcsec : float
        Side of the square box to be cut around each target.
    index : pyarrow.Table
        From :func:`read_index` (needs ``s_ra``, ``s_dec`` and the corners).

    Returns
    -------
    numpy structured array with fields ``target`` and ``frame`` (row numbers in
    ``ra``/``dec`` and in ``index``), sorted by frame, then target: the order
    that reads each file once.
    """
    from scipy.spatial import cKDTree

    ra = np.atleast_1d(np.asarray(ra, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec, dtype=np.float64))
    planes = _frame_planes(index)
    tree = cKDTree(planes["centre"])
    chord = 2.0 * np.sin(np.radians(cone_radius_deg) / 2.0)
    margin_rad = np.radians((size_arcsec / np.sqrt(2.0) + margin_pix * PIXEL_SCALE_ARCSEC) / 3600.0)
    targets = _unit_vectors(ra, dec)
    out_t, out_f = [], []
    for start in range(0, len(ra), chunk):
        stop = min(start + chunk, len(ra))
        hits = tree.query_ball_point(targets[start:stop], r=chord)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            continue
        t_idx = np.repeat(np.arange(start, stop), lens)
        f_idx = np.fromiter((f for h in hits for f in h), dtype=np.int64, count=int(lens.sum()))
        ok = _inside_with_margin(targets[t_idx], planes, f_idx, margin_rad)
        out_t.append(t_idx[ok])
        out_f.append(f_idx[ok])
    pairs = np.zeros(sum(len(t) for t in out_t), dtype=[("target", np.int64), ("frame", np.int64)])
    if out_t:
        pairs["target"] = np.concatenate(out_t)
        pairs["frame"] = np.concatenate(out_f)
    return np.sort(pairs, order=("frame", "target"))


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spherex-index",
                                description="Index a local SPHEREx L2 archive from its ObsCore tables.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Build index.parquet for an archive root.")
    b.add_argument("archive_root", type=Path)
    b.add_argument("-o", "--output", type=Path, required=True)
    b.add_argument("--obscore-glob", default="loads/*/obscore/*.tbl")
    b.add_argument("--level2-dir", default="repo/level2")
    b.add_argument("--workers", type=int, default=16, help="Threads for the directory scan.")
    b.add_argument("--no-headers", action="store_true",
                   help="Do not read the headers of files that no ObsCore table lists.")
    b.add_argument("--release", default=None,
                   help="Data release of the archive (e.g. qr2); recorded for the local query backend.")
    b.add_argument("--unindexed-out", type=Path, default=None,
                   help="Write the paths of files left without metadata, one per line.")
    b.add_argument("--superseded-out", type=Path, default=None,
                   help="Write the paths of files a later processing version replaced.")
    i = sub.add_parser("info", help="Summarise an index parquet.")
    i.add_argument("index", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        table, report = build_index(args.archive_root, obscore_glob=args.obscore_glob,
                                    level2_dir=args.level2_dir, workers=args.workers,
                                    release=args.release, headers=not args.no_headers,
                                    log=lambda m: print(m, flush=True))
        write_index(table, args.output, report)
        if args.unindexed_out is not None:
            Path(args.unindexed_out).write_text("".join(f"{p}\n" for p in report["unindexed"]))
        if args.superseded_out is not None:
            Path(args.superseded_out).write_text("".join(f"{p}\n" for p in report["superseded"]))
        print(json.dumps(_summary(report), indent=2))
        return 0
    if args.command == "info":
        table = read_index(args.index, columns=["detector", "week", "procver", "size"])
        meta = index_metadata(args.index)
        print(json.dumps(meta, indent=2))
        print(f"rows: {table.num_rows:,}")
        full = read_index(args.index, columns=["meta"])
        src = np.asarray(full.column("meta").to_pylist(), dtype=object)
        print("metadata source:", {str(v): int((src == v).sum()) for v in np.unique(src)})
        det = np.asarray(table.column("detector").to_numpy())
        print("per detector:", {int(d): int((det == d).sum()) for d in np.unique(det)})
        size = np.asarray(table.column("size").to_numpy())
        vals, counts = np.unique(size, return_counts=True)
        print("file sizes:", {int(v): int(c) for v, c in zip(vals, counts)})
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
