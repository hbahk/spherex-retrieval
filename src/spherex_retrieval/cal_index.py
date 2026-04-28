"""Discovery of SPHEREx calibration products.

Two strategies, in order:

1. **SIA2** (``Irsa.query_sia(pos=coord, collection="spherex_qr2_cal")``).
   Returns the right cal file when a position is supplied.  Cal files are
   detector-wide so any sky position covered by the detector matches.
2. **Browsable-directory listing** — fetch the IRSA ``ibe`` HTML index for
   the cal product family, regex out tokens of the form
   ``cal-<family>-v<N>-YYYY-DDD``, and pick the lexicographically largest
   (= latest version, then latest processing date).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

CalFamily = Literal["spectral_wcs", "solid_angle_pixel_map"]

_FAMILY_PREFIX = {
    "spectral_wcs": "cal-wcs",
    "solid_angle_pixel_map": "cal-sapm",
}


def cal_filename(family: CalFamily, detector: int, token: str) -> str:
    """e.g. ``spectral_wcs_D1_spx_cal-wcs-v4-2025-254.fits``."""
    return f"{family}_D{detector}_spx_{token}.fits"


def cal_http_url(family: CalFamily, detector: int, token: str, *, data_release: str = "qr2") -> str:
    fname = cal_filename(family, detector, token)
    return (
        f"https://irsa.ipac.caltech.edu/ibe/data/spherex/{data_release}/"
        f"{family}/{token}/{detector}/{fname}"
    )


def cal_s3_uri(family: CalFamily, detector: int, token: str, *, data_release: str = "qr2") -> str:
    fname = cal_filename(family, detector, token)
    return (
        f"s3://nasa-irsa-spherex/{data_release}/{family}/{token}/{detector}/{fname}"
    )


# --------------------------------------------------------------------------- #
# Strategy 1: SIA2
# --------------------------------------------------------------------------- #

def find_via_sia(
    family: CalFamily,
    detector: int,
    *,
    coord: SkyCoord | None = None,
    radius: u.Quantity = 0.01 * u.deg,
    data_release: str = "qr2",
) -> tuple[str, str] | None:
    """Return ``(http_url, s3_uri)`` from SIA2, or ``None`` on miss."""
    try:
        from astroquery.ipac.irsa import Irsa
    except Exception:
        return None

    try:
        if coord is not None:
            raw = Irsa.query_sia(pos=(coord, radius), collection=f"spherex_{data_release}_cal")
        else:
            raw = Irsa.query_sia(collection=f"spherex_{data_release}_cal")
    except Exception:
        return None

    prefix = f"{family}_D{detector}_"
    best_token = None
    best_row = None
    for row in raw:
        fname = Path(str(row["access_url"])).name
        if not fname.startswith(prefix):
            continue
        token = _token_from_filename(family, fname)
        if token is None:
            continue
        if best_token is None or token > best_token:  # lex sort = latest version+date
            best_token = token
            best_row = row
    if best_row is None:
        return None

    http_url = str(best_row["access_url"])
    s3 = ""
    if "cloud_access" in best_row.colnames:
        from .query import _extract_cloud_uri  # local import to avoid cycle
        s3 = _extract_cloud_uri(best_row)
    if not s3:
        s3 = cal_s3_uri(family, detector, best_token, data_release=data_release)
    return http_url, s3


def _token_from_filename(family: CalFamily, fname: str) -> str | None:
    prefix = _FAMILY_PREFIX[family]
    m = re.search(rf"({re.escape(prefix)}-v\d+-\d{{4}}-\d{{3}})", fname)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Strategy 2: HTML directory listing on irsa.ipac.caltech.edu/ibe
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def latest_cal_token_via_listing(
    family: CalFamily,
    *,
    data_release: str = "qr2",
    timeout: float = 60.0,
) -> str | None:
    """Hit the IRSA ``ibe`` listing API and return the latest cal token.

    The endpoint ``/ibe/dir/list/<path>`` returns NDJSON
    (``{"name": "...", "last_modified": "...", "size": "..."}`` per line)
    where each entry is a child of ``<path>``.  We pull the names matching
    the cal-product token pattern and pick the lex-largest.
    """
    listing_url = (
        f"https://irsa.ipac.caltech.edu/ibe/dir/list/spherex/{data_release}/{family}"
    )
    prefix = _FAMILY_PREFIX[family]
    try:
        resp = requests.get(listing_url, timeout=timeout)
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.text.strip():
        return None
    import json as _json
    pattern = re.compile(rf"^{re.escape(prefix)}-v\d+-\d{{4}}-\d{{3}}$")
    tokens: list[str] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        name = entry.get("name", "")
        if pattern.match(name):
            tokens.append(name)
    if not tokens:
        return None
    return sorted(set(tokens))[-1]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def discover_cal_product(
    family: CalFamily,
    detector: int,
    *,
    coord: SkyCoord | None = None,
    cal_token: str | None = None,
    data_release: str = "qr2",
) -> tuple[str, str]:
    """Resolve a calibration product to ``(http_url, s3_uri)``.

    Resolution order:
      1. ``cal_token`` argument (caller pinned a specific version).
      2. SIA2 (``spherex_qr2_cal`` collection, positional).
      3. HTML directory listing of the IRSA ``ibe`` browsable index.

    Raises
    ------
    RuntimeError
        If none of the strategies yield a valid cal token.
    """
    if cal_token:
        return (
            cal_http_url(family, detector, cal_token, data_release=data_release),
            cal_s3_uri(family, detector, cal_token, data_release=data_release),
        )

    sia = find_via_sia(family, detector, coord=coord, data_release=data_release)
    if sia is not None:
        return sia

    token = latest_cal_token_via_listing(family, data_release=data_release)
    if token is None:
        raise RuntimeError(
            f"could not discover {family} cal product for D{detector} "
            f"(SIA2 returned nothing and the directory listing was unreachable)"
        )
    return (
        cal_http_url(family, detector, token, data_release=data_release),
        cal_s3_uri(family, detector, token, data_release=data_release),
    )
