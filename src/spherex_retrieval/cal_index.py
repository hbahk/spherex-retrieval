"""Discovery of SPHEREx calibration products.

Two strategies, in order:

1. **SIA2** (``Irsa.query_sia(pos=coord, collection="spherex_qr2_cal")``).
   Returns the right cal file when a position is supplied.  Cal files are
   detector-wide so any sky position covered by the detector matches.
2. **Browsable-directory listing** — fetch the IRSA ``ibe`` HTML index for
   the cal product family, regex out tokens of the form
   ``cal-<family>-v<N>-YYYY-DDD``, and pick the lexicographically largest
   (= latest version, then latest processing date).

A release can instead be served from a **local calibration tree** laid out
like IRSA's (``<root>/<family>/<token>/<det>/<family>_D<det>_spx_<token>.fits``),
set with :func:`set_local_cal_roots` or ``SPHEREX_CAL_ROOTS="qr2=/a,qr3=/b"``.
A configured release never touches the network: the token rule is the
listing's (the latest token whose directory holds the detector), and a
missing product is an error rather than a silent fallback to IRSA.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

CalFamily = Literal["spectral_wcs", "solid_angle_pixel_map", "average_psf", "epsf",
                    "l3_flux_corrections"]

#: Token prefixes per family; a family may have been renamed between releases
#: (the spectral WCS is ``cal-wcs-v4-...`` under ``qr2`` and ``cal-swcs-v5-...``
#: under ``qr3``), so each entry lists every prefix seen.
_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "spectral_wcs": ("cal-wcs", "cal-swcs"),
    "solid_angle_pixel_map": ("cal-sapm",),
    "average_psf": ("cal-psf",),
    "epsf": ("cal-epsf",),                     # R7 effective PSF library (qr3+)
    "l3_flux_corrections": ("cal-flxc",),      # QR2 -> R7 per-pixel gain factors
}


def _token_pattern(family: CalFamily) -> re.Pattern[str]:
    alts = "|".join(re.escape(p) for p in _FAMILY_PREFIXES[family])
    return re.compile(rf"((?:{alts})-v\d+-\d{{4}}-\d{{3}})")


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
# Local calibration trees
# --------------------------------------------------------------------------- #

_LOCAL_CAL_ROOTS: dict[str, Path] = {}


def set_local_cal_roots(roots: dict[str, str | Path] | None) -> None:
    """Serve the given releases' cal products from local trees (``None`` clears).

    ``roots`` maps a data release (``"qr2"``, ``"qr3"``, ...) to a directory
    laid out like IRSA's ``spherex/<release>/``. Process-wide; takes precedence
    over ``SPHEREX_CAL_ROOTS``.
    """
    _LOCAL_CAL_ROOTS.clear()
    for release, root in (roots or {}).items():
        _LOCAL_CAL_ROOTS[str(release)] = Path(root)
    _DISCOVERED.clear()


def local_cal_roots() -> dict[str, Path]:
    """The configured local cal roots: ``SPHEREX_CAL_ROOTS`` overlaid by
    :func:`set_local_cal_roots`."""
    roots: dict[str, Path] = {}
    for item in os.environ.get("SPHEREX_CAL_ROOTS", "").split(","):
        if "=" in item:
            release, root = item.split("=", 1)
            roots[release.strip()] = Path(root.strip())
    roots.update(_LOCAL_CAL_ROOTS)
    return roots


def local_cal_product(family: CalFamily, detector: int, *, data_release: str,
                      cal_token: str | None = None) -> tuple[str, str] | None:
    """``(path, token)`` of a cal product in the local tree of ``data_release``.

    ``None`` when no local root is configured for that release. With a root
    configured, the product must be there: ``FileNotFoundError`` otherwise.
    """
    root = local_cal_roots().get(data_release)
    if root is None:
        return None
    fam = root / family
    if cal_token:
        tokens = [cal_token]
    else:
        pattern = _token_pattern(family)
        names = os.listdir(fam) if fam.is_dir() else []
        tokens = sorted((n for n in names if pattern.fullmatch(n)), reverse=True)
    for token in tokens:
        path = fam / token / str(int(detector)) / cal_filename(family, detector, token)
        if path.is_file():
            return str(path), token
    raise FileNotFoundError(
        f"no {family} product for D{detector} under the local {data_release} cal root {root}"
        + (f" (token {cal_token})" if cal_token else ""))


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
    m = _token_pattern(family).search(fname)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Strategy 2: HTML directory listing on irsa.ipac.caltech.edu/ibe
# --------------------------------------------------------------------------- #

def _list_names(path: str, *, timeout: float = 60.0) -> list[str] | None:
    """Child names of an IRSA ``ibe`` directory (NDJSON listing), or ``None``
    when the listing is unreachable."""
    listing_url = f"https://irsa.ipac.caltech.edu/ibe/dir/list/spherex/{path}"
    try:
        resp = requests.get(listing_url, timeout=timeout)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    import json as _json
    names: list[str] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        name = entry.get("name", "")
        if name:
            names.append(name)
    return names


@lru_cache(maxsize=32)
def latest_cal_token_via_listing(
    family: CalFamily,
    *,
    data_release: str = "qr2",
    timeout: float = 60.0,
    detector: int | None = None,
) -> str | None:
    """Hit the IRSA ``ibe`` listing API and return the latest cal token.

    The endpoint ``/ibe/dir/list/<path>`` returns NDJSON
    (``{"name": "...", "last_modified": "...", "size": "..."}`` per line)
    where each entry is a child of ``<path>``.  We pull the names matching
    the cal-product token pattern and pick the lex-largest.

    With ``detector`` given, only tokens whose directory holds that
    detector count: a version may be re-issued for one detector only (the
    ``qr3`` ``epsf`` family has ``cal-epsf-v2-2026-191`` for D3 alone, the
    finer 11x41 lattice, while D1, D2, D4-6 stay on ``v1``), and the
    lex-largest token overall would then be a 404 for the others.
    """
    names = _list_names(f"{data_release}/{family}", timeout=timeout)
    if not names:
        return None
    pattern = _token_pattern(family)
    tokens = sorted({n for n in names if pattern.fullmatch(n)})
    if not tokens:
        return None
    if detector is None:
        return tokens[-1]
    for token in reversed(tokens):
        children = _list_names(f"{data_release}/{family}/{token}", timeout=timeout)
        if children is not None and str(int(detector)) in children:
            return token
    return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

_DISCOVERED: dict[tuple, tuple[str, str]] = {}


def discover_cal_product(
    family: CalFamily,
    detector: int,
    *,
    coord: SkyCoord | None = None,
    cal_token: str | None = None,
    data_release: str = "qr2",
) -> tuple[str, str]:
    """Memoised :func:`_discover_cal_product`.

    Cal products are detector-wide, so the answer does not depend on
    ``coord``; one SIA2 round trip (0.6-3 s) per (family, detector) per
    process instead of one per cutout.  Failures are not cached.
    """
    key = (family, detector, cal_token, data_release,
           str(local_cal_roots().get(data_release, "")))
    hit = _DISCOVERED.get(key)
    if hit is None:
        hit = _DISCOVERED[key] = _discover_cal_product(
            family, detector, coord=coord, cal_token=cal_token, data_release=data_release
        )
    return hit


def _discover_cal_product(
    family: CalFamily,
    detector: int,
    *,
    coord: SkyCoord | None = None,
    cal_token: str | None = None,
    data_release: str = "qr2",
) -> tuple[str, str]:
    """Resolve a calibration product to ``(http_url, s3_uri)``.

    Resolution order:
      0. the local cal tree of ``data_release``, if one is configured
         (``s3_uri`` is then empty);
      1. ``cal_token`` argument (caller pinned a specific version).
      2. SIA2 (``spherex_qr2_cal`` collection, positional).
      3. HTML directory listing of the IRSA ``ibe`` browsable index.

    Raises
    ------
    RuntimeError
        If none of the strategies yield a valid cal token.
    """
    local = local_cal_product(family, detector, data_release=data_release, cal_token=cal_token)
    if local is not None:
        return local[0], ""
    if cal_token:
        return (
            cal_http_url(family, detector, cal_token, data_release=data_release),
            cal_s3_uri(family, detector, cal_token, data_release=data_release),
        )

    sia = find_via_sia(family, detector, coord=coord, data_release=data_release)
    if sia is not None:
        return sia

    token = latest_cal_token_via_listing(family, data_release=data_release,
                                         detector=detector)
    if token is None:
        raise RuntimeError(
            f"could not discover {family} cal product for D{detector} "
            f"(SIA2 returned nothing and the directory listing was unreachable)"
        )
    return (
        cal_http_url(family, detector, token, data_release=data_release),
        cal_s3_uri(family, detector, token, data_release=data_release),
    )
