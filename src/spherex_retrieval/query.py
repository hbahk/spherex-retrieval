"""Discovery queries: which SPHEREx L2 MEFs cover a given sky position?

Three backends are exposed; the default ``astroquery`` backend uses IRSA's
SIA2 service, the ``pyvo`` backend issues an ADQL TAP query, and the
``local`` backend searches the index of a local archive
(:mod:`spherex_retrieval.index`), whose ``access_url`` is a local path.

The backends return an :class:`~astropy.table.Table` with a common set
of columns:

    obs_id              : str    — SPHEREx Observation ID
    detector            : int    — 1..6
    bandpass            : str    — e.g. 'SPHEREx-D2' (the IVOA energy_bandpassname)
    access_url          : str    — HTTPS URL of the on-prem L2 MEF
    cloud_uri           : str    — S3 URI (or empty string)
    time_bounds_lower   : float  — start time (MJD)
    collection          : str    — e.g. spherex_qr2 / spherex_qr2_deep / spherex_qr2_cal

A ``bandpass`` filter (e.g. ``'SPHEREx-D2'``) can be passed to either
backend to restrict the results to a single SPHEREx detector.
"""

from __future__ import annotations

import re
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table


def _empty_canonical_table() -> Table:
    """Empty table with the canonical schema and dtypes."""
    return Table(
        {
            "access_url": np.array([], dtype=str),
            "cloud_uri": np.array([], dtype=str),
            "obs_id": np.array([], dtype=str),
            "bandpass": np.array([], dtype=str),
            "detector": np.array([], dtype=np.int32),
            "time_bounds_lower": np.array([], dtype=np.float64),
            "collection": np.array([], dtype=str),
        }
    )

CollectionName = Literal["spherex_qr2", "spherex_qr2_deep", "spherex_qr2_cal",
                         "spherex_qr3", "spherex_qr3_deep", "spherex_qr3_cal"]
#: Default search order: every quick release, oldest first. QR2 files carry the
#: optical PSF cube, QR3 (pipeline R7) files the effective PSF (``EPSF``); the
#: bundle records which (``PSFKIND``), so mixing releases in one retrieval is
#: fine for the retrieval — the photometry layer groups by kind.
SUPPORTED_COLLECTIONS = ("spherex_qr2", "spherex_qr2_deep", "spherex_qr3", "spherex_qr3_deep")
#: Releases that IRSA's SIA2/CAOM service does not list (2026-09-18: QR3 images
#: are in the ``spherex.plane``/``spherex.artifact`` TAP tables but not in
#: CAOM); their discovery falls back to the TAP backend automatically.
SIA2_MISSING_RELEASES = ("qr3",)

_RELEASE_RE = re.compile(r"^spherex_([a-z0-9]+?)(?:_deep|_cal)?$")
_RELEASE_URI_RE = re.compile(r"/spherex/([a-z0-9]+)/")


def release_of_collection(collection: str) -> str:
    """``'spherex_qr3_deep' -> 'qr3'``: the data-release directory of a collection."""
    m = _RELEASE_RE.match(str(collection))
    if not m:
        raise ValueError(f"not a SPHEREx collection name: {collection!r}")
    return m.group(1)


def release_of_url(url: str) -> str | None:
    """``'.../ibe/data/spherex/qr3/level2/...' -> 'qr3'`` (``None`` if absent)."""
    m = _RELEASE_URI_RE.search(str(url))
    return m.group(1) if m else None


def observation_id_from_filename(name: str) -> str:
    """``level2_2026W32_1A_0001_1D1_spx_l2b-v27-2026-223.fits -> 2026W32_1A_0001_1``."""
    m = re.search(r"level2_(\d{4}W\d{2}_\d[A-Z]_\d{4}_\d)D\d_", str(name).rsplit("/", 1)[-1])
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# Astroquery / SIA2 backend (default)
# --------------------------------------------------------------------------- #

def query_sia2(
    coord: SkyCoord,
    size: u.Quantity,
    *,
    collection: CollectionName = "spherex_qr2",
    bandpass: str | None = None,
    timeout: float = 120.0,
) -> Table:
    """Search for L2 MEFs covering ``coord`` via IRSA's SIA2 (astroquery).

    This is the default backend.  It returns a table with the canonical
    columns described in the module docstring.
    """
    from astroquery.ipac.irsa import Irsa

    Irsa.TIMEOUT = timeout
    radius = (size / 2.0).to(u.deg)
    raw = Irsa.query_sia(pos=(coord, radius), collection=collection)
    if bandpass is not None and "energy_bandpassname" in raw.colnames:
        raw = raw[raw["energy_bandpassname"] == bandpass]
    return _normalize_sia2_table(raw, collection=collection)


def _normalize_sia2_table(raw: Table, *, collection: str) -> Table:
    """Map the raw SIA2 columns to the canonical schema.

    All columns are typed explicitly so that empty results still merge
    correctly with non-empty results in :func:`find_overlapping`.
    """
    n = len(raw)
    if n == 0:
        return _empty_canonical_table()

    if "obs_id" in raw.colnames:
        obs_ids = [str(s) for s in raw["obs_id"]]
    elif "dataproduct_subtype" in raw.colnames:
        obs_ids = [str(s) for s in raw["dataproduct_subtype"]]
    else:
        obs_ids = [""] * n

    if "energy_bandpassname" in raw.colnames:
        bandpasses = [str(s) for s in raw["energy_bandpassname"]]
    else:
        bandpasses = [""] * n

    if "t_min" in raw.colnames:
        time_lower = np.asarray(raw["t_min"], dtype=np.float64)
    elif "time_bounds_lower" in raw.colnames:
        time_lower = np.asarray(raw["time_bounds_lower"], dtype=np.float64)
    else:
        time_lower = np.full(n, np.nan, dtype=np.float64)

    return Table(
        {
            "access_url": np.asarray([str(s) for s in raw["access_url"]], dtype=str),
            "cloud_uri": np.asarray(
                [_extract_cloud_uri(row) for row in raw], dtype=str
            ),
            "obs_id": np.asarray(obs_ids, dtype=str),
            "bandpass": np.asarray(bandpasses, dtype=str),
            "detector": np.asarray(
                [_detector_from_bandpass(s) for s in bandpasses], dtype=np.int32
            ),
            "time_bounds_lower": time_lower,
            "collection": np.asarray([collection] * n, dtype=str),
        }
    )


def _extract_cloud_uri(row) -> str:
    if "cloud_access" not in row.colnames:
        return ""
    val = row["cloud_access"]
    if val is None:
        return ""
    text = str(val)
    # cloud_access is a JSON-ish blob; pull the s3 uri if present.
    import json
    try:
        info = json.loads(text)
    except Exception:
        return ""
    aws = info.get("aws", {}) if isinstance(info, dict) else {}
    bucket = aws.get("bucket_name") or aws.get("bucket")
    key = aws.get("key")
    if bucket and key:
        return f"s3://{bucket}/{key}"
    return ""


def _detector_from_bandpass(bandpass: str) -> int:
    """Extract the SPHEREx detector index 1..6 from a bandpass string.

    The IVOA ``energy_bandpassname`` column for SPHEREx uses the form
    ``'SPHEREx-D{n}'`` where ``n`` is 1..6.
    """
    if not bandpass:
        return -1
    s = str(bandpass)
    for i in range(1, 7):
        if f"D{i}" in s:
            return i
    return -1


# --------------------------------------------------------------------------- #
# Pyvo / TAP backend (alternate, matches the existing notebook)
# --------------------------------------------------------------------------- #

TAP_ENDPOINT = "https://irsa.ipac.caltech.edu/TAP"


def query_tap(
    coord: SkyCoord,
    size: u.Quantity,  # noqa: ARG001 (kept for parity with sia2 signature)
    *,
    collection: CollectionName = "spherex_qr2",
    bandpass: str | None = None,
    timeout: float = 120.0,
) -> Table:
    """Alternate backend: ADQL against IRSA's ``spherex.plane`` / ``spherex.artifact``.

    The point-in-footprint test uses ``p.poly``; the release is selected by
    the artifact path (``.../spherex/<release>/level2/...``), which is how
    the ``qr3`` images are reachable while SIA2 does not list them, and the
    collection by ``spherex.observation.collection`` — the path alone does not
    tell a wide image from a deep one (``spherex_qr3`` and ``spherex_qr3_deep``
    share ``/spherex/qr3/``), so without it each collection of a release would
    return every image of the release. Returns
    raw L2 MEF URLs (not cutouts) so the cutout layer can choose the IRSA
    cutout service or S3 byte ranges; the S3 URI is derived from the IBE
    path, since the TAP artifact table carries no cloud column. The SPHEREx
    observation id is parsed from the file name (``p.obsid`` is a UUID).

    Set ``bandpass`` (e.g. ``'SPHEREx-D2'``) to filter by detector at
    the query level. Uses ``pyvo`` when importable, else the sync endpoint
    over HTTP.
    """
    release = release_of_collection(collection)
    ra = coord.icrs.ra.to_value(u.deg)
    dec = coord.icrs.dec.to_value(u.deg)
    extra_filter = (
        f"AND p.energy_bandpassname = '{bandpass}'" if bandpass else ""
    )
    adql = f"""
    SELECT
        a.uri AS access_path,
        p.time_bounds_lower,
        p.obsid,
        p.energy_bandpassname,
        p.provenance_version
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    JOIN spherex.observation o ON o.obsid = p.obsid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
        AND o.collection = '{collection}'
        AND a.uri LIKE '%/spherex/{release}/level2/%'
        {extra_filter}
    ORDER BY p.time_bounds_lower
    """
    raw = _run_adql(adql, timeout=timeout)
    n = len(raw)
    if n == 0:
        return _empty_canonical_table()

    paths = [str(p).lstrip("/") for p in raw["access_path"]]
    bandpasses = [str(s) for s in raw["energy_bandpassname"]]
    return Table(
        {
            "access_url": np.asarray(
                [f"https://irsa.ipac.caltech.edu/{p}" for p in paths], dtype=str),
            "cloud_uri": np.asarray(
                [f"s3://nasa-irsa-spherex/{p.split('ibe/data/spherex/', 1)[1]}"
                 if "ibe/data/spherex/" in p else "" for p in paths], dtype=str),
            "obs_id": np.asarray([observation_id_from_filename(p) for p in paths], dtype=str),
            "bandpass": np.asarray(bandpasses, dtype=str),
            "detector": np.asarray(
                [_detector_from_bandpass(s) for s in bandpasses], dtype=np.int32
            ),
            "time_bounds_lower": np.asarray(
                raw["time_bounds_lower"], dtype=np.float64
            ),
            "collection": np.asarray([collection] * n, dtype=str),
        }
    )


def _run_adql(adql: str, *, timeout: float = 120.0) -> Table:
    """Run a synchronous ADQL query; ``pyvo`` if available, else plain HTTP."""
    try:
        import pyvo
    except ImportError:
        pyvo = None
    if pyvo is not None:
        service = pyvo.dal.TAPService(TAP_ENDPOINT)
        return service.search(adql).to_table()
    import io

    import requests
    resp = requests.get(f"{TAP_ENDPOINT}/sync", params={"QUERY": adql, "FORMAT": "csv"},
                        timeout=timeout)
    resp.raise_for_status()
    return Table.read(io.StringIO(resp.text), format="ascii.csv")


# --------------------------------------------------------------------------- #
# Local archive backend
# --------------------------------------------------------------------------- #

def query_local(
    coord: SkyCoord,
    size: u.Quantity,
    *,
    index,
    archive_root=None,
    release: str | None = None,
    bandpass: str | None = None,
    margin_pix: float = 10.0,
) -> Table:
    """Frames of a local archive index whose footprint can hold the ``size`` box.

    ``index`` is an index parquet path or its ``pyarrow.Table``
    (:func:`spherex_retrieval.index.read_index`). ``archive_root`` and
    ``release`` default to the values recorded when the index was built. The
    footprint test is the generous one of
    :func:`~spherex_retrieval.index.find_overlapping_many`; a frame the box
    misses fails at the cutout step, as an IRSA cutout request would.
    """
    from pathlib import Path

    from . import index as sidx

    meta = {}
    if isinstance(index, (str, Path)):
        meta = sidx.index_metadata(index)
        index = sidx.read_index(index)
    else:
        meta = sidx.index_metadata(index)
    root = Path(archive_root or meta.get("archive_root") or "")
    release = release or meta.get("release")
    if not release:
        raise ValueError("the index records no release; pass release= (e.g. 'qr2')")
    ra = coord.icrs.ra.to_value(u.deg)
    dec = coord.icrs.dec.to_value(u.deg)
    pairs = sidx.find_overlapping_many([ra], [dec], size.to_value(u.arcsec), index,
                                       margin_pix=margin_pix)
    rows = index.take(pairs["frame"]) if len(pairs) else index.slice(0, 0)
    det = np.asarray(rows.column("detector").to_numpy(), dtype=np.int32)
    if bandpass is not None:
        keep = det == _detector_from_bandpass(bandpass)
        rows, det = rows.filter(keep), det[keep]
    n = rows.num_rows
    if n == 0:
        return _empty_canonical_table()
    return Table(
        {
            "access_url": np.asarray([str(root / p) for p in rows.column("path").to_pylist()],
                                     dtype=str),
            "cloud_uri": np.asarray([""] * n, dtype=str),
            "obs_id": np.asarray(rows.column("obs_id").to_pylist(), dtype=str),
            "bandpass": np.asarray([f"SPHEREx-D{d}" for d in det], dtype=str),
            "detector": det,
            "time_bounds_lower": np.asarray(rows.column("t_min").to_numpy(), dtype=np.float64),
            "collection": np.asarray([f"spherex_{release}"] * n, dtype=str),
        }
    )


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #

def find_overlapping(
    coord: SkyCoord,
    size: u.Quantity,
    *,
    backend: Literal["astroquery", "pyvo", "local"] = "astroquery",
    collections: tuple[CollectionName, ...] = SUPPORTED_COLLECTIONS,
    bandpass: str | None = None,
    timeout: float = 120.0,
    index=None,
    archive_root=None,
) -> Table:
    """Find all L2 MEFs covering ``coord`` across the requested collections.

    Each L2 file appears once: a file listed by more than one collection keeps
    the row of the first collection in ``collections`` (fitting the same
    exposure twice would put two points per exposure in the spectrum).

    ``backend="local"`` searches a local archive ``index`` instead (see
    :func:`query_local`); ``collections`` does not apply there — the index is
    one archive of one release.
    """
    if backend == "local":
        if index is None:
            raise ValueError("backend='local' needs index= (an index parquet or table)")
        return _drop_duplicate_files(query_local(coord, size, index=index,
                                                 archive_root=archive_root, bandpass=bandpass))
    tables = []
    for col in collections:
        if backend == "astroquery":
            if release_of_collection(col) in SIA2_MISSING_RELEASES:
                # not in CAOM/SIA2 yet: the TAP tables have the footprints
                t = query_tap(coord, size, collection=col, bandpass=bandpass, timeout=timeout)
            else:
                t = query_sia2(coord, size, collection=col, bandpass=bandpass, timeout=timeout)
        elif backend == "pyvo":
            t = query_tap(coord, size, collection=col, bandpass=bandpass, timeout=timeout)
        else:
            raise ValueError(f"unknown query backend: {backend!r}")
        if len(t) > 0:
            tables.append(t)
    if not tables:
        return _empty_canonical_table()
    from astropy.table import vstack
    combined = _drop_duplicate_files(vstack(tables))
    combined.sort("time_bounds_lower")
    return combined


def _drop_duplicate_files(table: Table) -> Table:
    """Keep the first row of each L2 file, keyed by file name.

    The name (``level2_<week>_<expo>_<dither>D<det>_spx_<procver>.fits``) is
    the product identity: the same name is the same exposure, detector and
    processing, whichever collection or host lists it.
    """
    names = [str(url).split("?", 1)[0].rsplit("/", 1)[-1] for url in table["access_url"]]
    _, first = np.unique(names, return_index=True)
    if len(first) == len(table):
        return table
    return table[np.sort(first)]
