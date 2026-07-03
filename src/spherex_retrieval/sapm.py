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

def find_sapm_product(
    detector: int,
    *,
    backend: str = "astroquery",  # noqa: ARG001 — dispatch lives in cal_index
    data_release: str = "qr2",
    cal_token: str | None = None,
    coord: "SkyCoord | None" = None,
) -> tuple[str, str]:
    """Return ``(http_url, s3_uri)`` for the matching SAPM file.

    Resolution chain (see :mod:`spherex_retrieval.cal_index`):

    1. ``cal_token`` if pinned by the caller (e.g. the SDT demo's
       ``"cal-sapm-v2-2025-164"``).
    2. SIA2 ``COLLECTION=spherex_qr2_cal`` using ``coord``.
    3. HTML directory listing of the IRSA browsable index — picks the
       latest ``cal-sapm-vN-YYYY-DDD`` token.
    """
    from .cal_index import discover_cal_product

    return discover_cal_product(
        "solid_angle_pixel_map",
        detector,
        coord=coord,
        cal_token=cal_token,
        data_release=data_release,
    )


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
    # A negative origin would wrap .section[ylo:ylo+ny] to the mirrored
    # detector rows; fail loudly (see crop_wavelength_maps for the rationale).
    if xlo < 0 or ylo < 0:
        raise ValueError(
            f"pixel_origin must be non-negative detector pixels, got {pixel_origin!r}"
        )
    with open_fits(cal_target, mode="auto", cache_dir=cache_dir,
                   fsspec_kwargs=fsspec_kwargs) as hdul:
        sapm_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        data = np.asarray(
            sapm_hdu.section[ylo:ylo + ny, xlo:xlo + nx], dtype=np.float32
        )
        bunit = str(sapm_hdu.header.get("BUNIT", "arcsec2"))
    return SolidAnglePixelMap(data=data, bunit=bunit, source_url=cal_target)
