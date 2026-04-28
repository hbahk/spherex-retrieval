"""Discovery queries: which SPHEREx L2 MEFs cover a given sky position?

Two backends are exposed; the default ``astroquery`` backend uses IRSA's
SIA2 service, and the ``pyvo`` backend issues an ADQL TAP query.

The two backends return an :class:`~astropy.table.Table` with a common set
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

from typing import Literal

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

CollectionName = Literal["spherex_qr2", "spherex_qr2_deep", "spherex_qr2_cal"]
SUPPORTED_COLLECTIONS = ("spherex_qr2", "spherex_qr2_deep")


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
    """Map the raw SIA2 columns to the canonical schema."""
    out = Table()
    out["access_url"] = [str(s) for s in raw["access_url"]]
    out["cloud_uri"] = [_extract_cloud_uri(row) for row in raw]
    out["obs_id"] = [
        str(s) for s in raw.get("obs_id", raw.get("dataproduct_subtype", [""] * len(raw)))
    ]
    if "energy_bandpassname" in raw.colnames:
        bandpasses = [str(s) for s in raw["energy_bandpassname"]]
    else:
        bandpasses = [""] * len(raw)
    out["bandpass"] = bandpasses
    out["detector"] = [_detector_from_bandpass(s) for s in bandpasses]
    if "t_min" in raw.colnames:
        out["time_bounds_lower"] = raw["t_min"]
    elif "time_bounds_lower" in raw.colnames:
        out["time_bounds_lower"] = raw["time_bounds_lower"]
    else:
        out["time_bounds_lower"] = [float("nan")] * len(raw)
    out["collection"] = [collection] * len(raw)
    return out


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
    collection: CollectionName = "spherex_qr2",  # noqa: ARG001
    bandpass: str | None = None,
    timeout: float = 120.0,  # noqa: ARG001
) -> Table:
    """Alternate backend using pyvo + ADQL.

    The ADQL schema served by IRSA exposes ``spherex.artifact`` /
    ``spherex.plane`` tables.  We return a raw access URL (the L2 MEF,
    not a cutout) so the cutout layer can decide between IRSA cutout-
    service or S3 byte-range paths.

    Set ``bandpass`` (e.g. ``'SPHEREx-D2'``) to filter by detector at
    the query level.
    """
    import pyvo

    ra = coord.icrs.ra.to_value(u.deg)
    dec = coord.icrs.dec.to_value(u.deg)

    service = pyvo.dal.TAPService(TAP_ENDPOINT)
    extra_filter = (
        f"AND p.energy_bandpassname = '{bandpass}'" if bandpass else ""
    )
    adql = f"""
    SELECT
        a.uri AS access_path,
        p.time_bounds_lower,
        p.obs_id,
        p.energy_bandpassname
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
        {extra_filter}
    ORDER BY p.time_bounds_lower
    """
    raw = service.search(adql).to_table()

    out = Table()
    out["access_url"] = [
        f"https://irsa.ipac.caltech.edu/{p.lstrip('/')}" for p in raw["access_path"]
    ]
    out["cloud_uri"] = [""] * len(raw)
    out["obs_id"] = [str(s) for s in raw["obs_id"]]
    bandpasses = [str(s) for s in raw["energy_bandpassname"]]
    out["bandpass"] = bandpasses
    out["detector"] = [_detector_from_bandpass(s) for s in bandpasses]
    out["time_bounds_lower"] = raw["time_bounds_lower"]
    out["collection"] = [collection] * len(raw)
    return out


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #

def find_overlapping(
    coord: SkyCoord,
    size: u.Quantity,
    *,
    backend: Literal["astroquery", "pyvo"] = "astroquery",
    collections: tuple[CollectionName, ...] = SUPPORTED_COLLECTIONS,
    bandpass: str | None = None,
    timeout: float = 120.0,
) -> Table:
    """Find all L2 MEFs covering ``coord`` across the requested collections."""
    tables = []
    for col in collections:
        if backend == "astroquery":
            tables.append(
                query_sia2(coord, size, collection=col, bandpass=bandpass, timeout=timeout)
            )
        elif backend == "pyvo":
            tables.append(
                query_tap(coord, size, collection=col, bandpass=bandpass, timeout=timeout)
            )
        else:
            raise ValueError(f"unknown query backend: {backend!r}")
    if not tables:
        return Table(names=("access_url", "cloud_uri", "obs_id", "bandpass",
                            "detector", "time_bounds_lower", "collection"))
    from astropy.table import vstack
    combined = vstack(tables)
    combined.sort("time_bounds_lower")
    return combined
