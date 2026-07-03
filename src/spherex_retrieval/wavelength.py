"""Per-pixel wavelength maps from the standalone Spectral WCS calibration product.

The L2 MEF carries a WCS-WAVE lookup table that is explicitly flagged in
the SPHEREx Explanatory Supplement as visualization-only (~1 nm).
For science (and especially forced photometry of multiple sources at
different positions in the same cutout), IRSA recommends the
``spectral_wcs_D[Det]_spx_cal-wcs-...`` product, which holds full-pixel
``CWAVE`` (central wavelength, microns) and ``CBAND`` (bandwidth, microns)
arrays at 2040 x 2040.

This module:

* Locates the matching calibration product per (detector, version,
  processing date) by querying SIA2 with ``COLLECTION=spherex_qr2_cal``,
  with a small in-process cache so we don't redo the lookup per cutout.
* Crops CWAVE and CBAND to the same pixel box as the science cutout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord

from .io import open_fits


@dataclass
class WavelengthMaps:
    cwave: np.ndarray   # central wavelength, microns
    cband: np.ndarray   # bandwidth, microns
    source_url: str


# --------------------------------------------------------------------------- #
# Cal-product discovery
# --------------------------------------------------------------------------- #

_CAL_FILENAME_RE = re.compile(
    r"^level2_(?P<obsid>[^/]+)D(?P<det>\d)_spx_l2b-v(?P<ver>\d+)-(?P<date>\d{4}-\d{3})\.fits"
)


def parse_l2_filename(name: str) -> dict | None:
    """Pull (obsid, detector, version, processing_date) from an L2 filename."""
    m = _CAL_FILENAME_RE.search(Path(name).name)
    if not m:
        return None
    d = m.groupdict()
    d["det"] = int(d["det"])
    d["ver"] = int(d["ver"])
    return d


def find_cal_product(
    detector: int,
    processing_date: str = "",  # noqa: ARG001 — kept for backwards compatibility
    *,
    backend: str = "astroquery",  # noqa: ARG001 — backend selection happens in cal_index
    data_release: str = "qr2",
    coord: SkyCoord | None = None,
    cal_token: str | None = None,
) -> tuple[str, str]:
    """Return ``(http_url, s3_uri)`` for the matching Spectral WCS cal file.

    Resolution chain (see :mod:`spherex_retrieval.cal_index`):

    1. ``cal_token`` if pinned by the caller.
    2. SIA2 (``COLLECTION=spherex_qr2_cal``) using ``coord`` (cal files are
       detector-wide, so any covered position works).
    3. HTML directory listing of the IRSA browsable index — picks the
       latest ``cal-wcs-vN-YYYY-DDD`` token.

    The L2 ``processing_date`` is **not** used as a filter here: cal
    products are released independently from the L2 pipeline and rarely
    share a processing date with the spectral images they apply to.
    """
    from .cal_index import discover_cal_product

    return discover_cal_product(
        "spectral_wcs",
        detector,
        coord=coord,
        cal_token=cal_token,
        data_release=data_release,
    )


# --------------------------------------------------------------------------- #
# Cropping the CWAVE/CBAND maps
# --------------------------------------------------------------------------- #

def crop_wavelength_maps(
    cal_target: str,
    *,
    pixel_origin: tuple[int, int],   # (xlo, ylo) 0-based detector pixels
    cutout_shape: tuple[int, int],   # (ny, nx)
    cache_dir=None,
    fsspec_kwargs: dict | None = None,
) -> WavelengthMaps:
    """Open the cal product and crop CWAVE/CBAND to the cutout bbox."""
    xlo, ylo = pixel_origin
    ny, nx = cutout_shape
    # A negative origin would make .section[ylo:ylo+ny] wrap around to the
    # mirrored detector rows (silent within-detector wavelength reversal);
    # fail loudly instead.  pixel_origin must be a true 0-based detector origin.
    if xlo < 0 or ylo < 0:
        raise ValueError(
            f"pixel_origin must be non-negative detector pixels, got {pixel_origin!r}"
        )

    with open_fits(cal_target, mode="auto", cache_dir=cache_dir,
                   fsspec_kwargs=fsspec_kwargs) as hdul:
        cwave_hdu = hdul["CWAVE"] if "CWAVE" in hdul else hdul[1]
        cband_hdu = hdul["CBAND"] if "CBAND" in hdul else hdul[2]
        # Use .section so we only fetch the relevant pixel box when streaming.
        cwave = np.asarray(cwave_hdu.section[ylo:ylo + ny, xlo:xlo + nx], dtype=np.float32)
        cband = np.asarray(cband_hdu.section[ylo:ylo + ny, xlo:xlo + nx], dtype=np.float32)

    return WavelengthMaps(cwave=cwave, cband=cband, source_url=cal_target)


def wavelength_at(
    maps: WavelengthMaps,
    *,
    x_cut: float,
    y_cut: float,
) -> tuple[float, float]:
    """Bilinear-interpolate (lambda, dlambda) at a 0-based cutout pixel position."""
    cwave = _bilinear(maps.cwave, x_cut, y_cut)
    cband = _bilinear(maps.cband, x_cut, y_cut)
    return cwave, cband


def _bilinear(arr: np.ndarray, x: float, y: float) -> float:
    ny, nx = arr.shape
    x = np.clip(x, 0, nx - 1)
    y = np.clip(y, 0, ny - 1)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    fx, fy = x - x0, y - y0
    a = arr[y0, x0] * (1 - fx) * (1 - fy)
    b = arr[y0, x1] * fx * (1 - fy)
    c = arr[y1, x0] * (1 - fx) * fy
    d = arr[y1, x1] * fx * fy
    return float(a + b + c + d)
