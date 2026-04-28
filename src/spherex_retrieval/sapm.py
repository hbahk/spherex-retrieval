"""Solid Angle Pixel Map (SAPM) calibration product.

The SAPM is a per-detector 2040 x 2040 image of pixel solid angles in
``arcsec^2``.  Multiplying the calibrated SPHEREx surface brightness
(``MJy/sr``, after conversion to ``uJy/arcsec^2``) by SAPM converts the
image to flux density (``uJy``), which is the natural unit for forced
photometry.  See [Solid Angle Pixel Map] in the IRSA SPHEREx user guide
and the worked example in the SPHEREx Source Discovery Tool demo.

Conversion convention used downstream::

    img_uJy = (img_MJy_per_sr * u.MJy / u.sr).to(u.uJy / u.arcsec**2) * (sapm * u.arcsec**2)

This module locates the matching SAPM product per detector and crops it
to the cutout pixel box, mirroring :mod:`spherex_retrieval.wavelength`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import open_fits


@dataclass
class SolidAnglePixelMap:
    data: np.ndarray   # arcsec^2 per pixel
    bunit: str         # always "arcsec2" in QR-2
    source_url: str


# --------------------------------------------------------------------------- #
# SAPM cal-product discovery
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=128)
def find_sapm_product(
    detector: int,
    *,
    backend: str = "astroquery",
    data_release: str = "qr2",
    cal_token: str | None = None,
) -> tuple[str, str]:
    """Return ``(http_url, s3_uri)`` for the matching SAPM file.

    SAPM products are released independently of the per-pointing L2 MEFs
    so we don't filter by processing date — the latest SAPM the SIA2
    catalog returns for the requested detector is the right one.

    Parameters
    ----------
    detector : int
        SPHEREx detector index 1..6.
    cal_token : str, optional
        Override token of the form ``cal-sapm-v{ver}-{YYYY-DDD}``.  When
        passed, the directory-fallback branch uses it directly instead of
        the SIA-derived value.  Useful when reproducing legacy results
        against a fixed cal version (the SDT demo hardcodes
        ``cal-sapm-v2-2025-164``).
    """
    if backend == "astroquery" and cal_token is None:
        try:
            from astroquery.ipac.irsa import Irsa

            raw = Irsa.query_sia(pos=None, collection=f"spherex_{data_release}_cal")
            best = None
            for row in raw:
                fname = Path(str(row["access_url"])).name
                if not fname.startswith("solid_angle_pixel_map_"):
                    continue
                if f"D{detector}_" not in fname:
                    continue
                # Pick lexicographically largest filename: SAPM filenames
                # carry the processing date, so this picks the most recent.
                if best is None or fname > Path(best[0]).name:
                    http_url = str(row["access_url"])
                    s3 = ""
                    if "cloud_access" in row.colnames:
                        from .query import _extract_cloud_uri  # avoid import cycle
                        s3 = _extract_cloud_uri(row)
                    best = (http_url, s3)
            if best is not None:
                return best
        except Exception:
            pass

    # Fallback: construct the canonical browsable-directory + S3 URI.
    token = cal_token or "cal-sapm-vLATEST-LATEST"
    fname = f"solid_angle_pixel_map_D{detector}_spx_{token}.fits"
    base_path = f"{data_release}/solid_angle_pixel_map/{token}/{detector}/{fname}"
    http_url = f"https://irsa.ipac.caltech.edu/ibe/data/spherex/{base_path}"
    s3_uri = f"s3://nasa-irsa-spherex/{base_path}"
    return http_url, s3_uri


# --------------------------------------------------------------------------- #
# Cropping the SAPM to the cutout
# --------------------------------------------------------------------------- #

def crop_sapm(
    cal_target: str,
    *,
    pixel_origin: tuple[int, int],
    cutout_shape: tuple[int, int],
    cache_dir=None,
    fsspec_kwargs: dict | None = None,
) -> SolidAnglePixelMap:
    """Open the SAPM cal product and crop to the cutout pixel bbox."""
    xlo, ylo = pixel_origin
    ny, nx = cutout_shape
    with open_fits(cal_target, mode="auto", cache_dir=cache_dir,
                   fsspec_kwargs=fsspec_kwargs) as hdul:
        sapm_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        data = np.asarray(
            sapm_hdu.section[ylo:ylo + ny, xlo:xlo + nx], dtype=np.float32
        )
        bunit = str(sapm_hdu.header.get("BUNIT", "arcsec2"))
    return SolidAnglePixelMap(data=data, bunit=bunit, source_url=cal_target)
