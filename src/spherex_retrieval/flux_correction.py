"""R7 flux corrections for pre-R7 (QR2) images.

The absolute gain calibration was re-derived for the Year-1 reprocessing (R7:
QR3 and DR1). So that QR2 images, calibrated with the previous gains, can be
combined with R7 ones, the SSDC publishes ``l3_flux_corrections_D<n>``
(family ``l3_flux_corrections``, tokens ``cal-flxc-v1-2026-191`` under
``qr3/``): one 2040 x 2040 image per detector of "multiplicative flux
correction factors for pre-QR3 data" (its own header comment), in the L2
image frame (``DETCOORD = 'sky'``), no unit. Measured on the v1 product: D1
median 0.970 with row medians 0.875-0.983 (the factor varies along the
dispersion direction), D5 median 1.007 (0.987-1.029); a few 1e-4 of the
pixels are wild (0.011 ... 3653), the dead/hot pixels.

:func:`crop_flux_correction` fetches and crops the map like the SAPM and
wavelength products; :func:`apply_flux_correction` multiplies a cutout's
IMAGE by the factor and its VARIANCE by the factor squared. The ZODI plane is
a model in physical units and is left alone: correcting the image is what
makes ``IMAGE - ZODI`` consistent again. Wild factors are replaced by the
median of their detector row (the pixels they belong to are flagged anyway),
so a bad pixel cannot poison its neighbours through the background fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import open_fits

#: Factors outside this range are treated as bad pixels and replaced by the
#: row median (the product's own dead/hot pixels; see the module docstring).
FACTOR_RANGE = (0.5, 2.0)


@dataclass
class FluxCorrection:
    data: np.ndarray      # per-pixel multiplicative factor, cropped to the cutout
    source_url: str
    token: str            # cal token, e.g. cal-flxc-v1-2026-191
    source_file: str      # the 'Calibration source file' HISTORY of the product, if any
    n_replaced: int       # wild factors replaced by their row median


def find_flux_correction_product(
    detector: int,
    *,
    data_release: str = "qr3",
    cal_token: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(http_url, s3_uri, token)`` of the detector's correction map."""
    from .cal_index import (
        cal_http_url,
        cal_s3_uri,
        latest_cal_token_via_listing,
        local_cal_product,
    )

    local = local_cal_product("l3_flux_corrections", detector, data_release=data_release,
                              cal_token=cal_token)
    if local is not None:
        return local[0], "", local[1]
    token = cal_token or latest_cal_token_via_listing(
        "l3_flux_corrections", data_release=data_release, detector=detector)
    if token is None:
        raise RuntimeError(
            f"could not list l3_flux_corrections cal products for D{detector} under "
            f"{data_release} on IRSA")
    return (cal_http_url("l3_flux_corrections", detector, token, data_release=data_release),
            cal_s3_uri("l3_flux_corrections", detector, token, data_release=data_release),
            token)


def crop_flux_correction(
    cal_target: str,
    *,
    pixel_origin: tuple[int, int],
    cutout_shape: tuple[int, int],
    token: str = "",
    cache_dir=None,
    fsspec_kwargs: dict | None = None,
) -> FluxCorrection:
    """Open the correction product and crop it to the cutout pixel bbox."""
    xlo, ylo = pixel_origin
    ny, nx = cutout_shape
    if xlo < 0 or ylo < 0:
        raise ValueError(
            f"pixel_origin must be non-negative detector pixels, got {pixel_origin!r}")
    with open_fits(cal_target, mode="auto", cache_dir=cache_dir,
                   fsspec_kwargs=fsspec_kwargs) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        # the row medians come from the full rows, not the crop: a narrow
        # cutout would otherwise take its median from a handful of pixels
        full_rows = np.asarray(hdu.section[ylo:ylo + ny, :], dtype=np.float64)
        source_file = ""
        for card in hdu.header.get("HISTORY", []):
            if "Calibration source file" in str(card):
                source_file = str(card).split(":", 1)[1].strip()
    lo, hi = FACTOR_RANGE
    bad_full = ~np.isfinite(full_rows) | (full_rows < lo) | (full_rows > hi)
    row_med = np.nanmedian(np.where(bad_full, np.nan, full_rows), axis=1)
    row_med = np.where(np.isfinite(row_med), row_med, 1.0)
    data = full_rows[:, xlo:xlo + nx].copy()
    bad = bad_full[:, xlo:xlo + nx]
    data[bad] = np.broadcast_to(row_med[:, None], data.shape)[bad]
    return FluxCorrection(data=data.astype(np.float32), source_url=cal_target, token=token,
                          source_file=source_file, n_replaced=int(bad.sum()))


def apply_flux_correction(cutout, corr: FluxCorrection) -> None:
    """Multiply the cutout's IMAGE by the factor and its VARIANCE by its square, in place."""
    f = np.asarray(corr.data, dtype=np.float64)
    if f.shape != cutout.image.shape:
        raise ValueError(f"flux-correction shape {f.shape} != image shape {cutout.image.shape}")
    # keep the arrays' dtype (and byte order) so the bundle is the same apart
    # from the values
    cutout.image = (np.asarray(cutout.image, dtype=np.float64) * f).astype(cutout.image.dtype)
    cutout.variance = (np.asarray(cutout.variance, dtype=np.float64) * f ** 2).astype(cutout.variance.dtype)
