"""Cutout retrieval from a SPHEREx Spectral Image MEF.

Two retrieval backends:

* ``irsa``   — append the IRSA cutout-service query string to the on-prem
  access URL.  The server returns a complete 6-extension MEF whose
  IMAGE/FLAGS/VARIANCE/ZODI HDUs are pre-cropped, the WCS-WAVE table has
  been re-mapped onto the cropped grid, and the PSF cube is passed through
  unchanged.  This is the simplest and the default mode.
* ``fsspec`` — open the full L2 MEF (HTTP byte-range or S3) and crop on the
  client using ``ImageHDU.section`` + :class:`~astropy.nddata.Cutout2D`.
  Useful when running in the same AWS region as the data, or when the IRSA
  service is unavailable.

The output is a :class:`CutoutPayload` dataclass that carries the cropped
arrays, the spatial WCS, and the relevant headers.  Wavelength maps are
retrieved separately by :mod:`spherex_retrieval.wavelength`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from .io import open_fits

CutoutBackend = Literal["irsa", "fsspec"]


@dataclass
class CutoutPayload:
    """Container for the spatial side of a SPHEREx cutout."""

    image: np.ndarray
    flags: np.ndarray
    variance: np.ndarray
    zodi: np.ndarray
    psf_cube: np.ndarray              # full 121-plane cube as delivered
    psf_header: fits.Header
    image_header: fits.Header         # used for spatial + spectral WCS
    primary_header: fits.Header       # carries VERSION, OBSID, etc. — needed for the PSF erratum check
    spatial_wcs: WCS
    detector: int
    pixel_origin: tuple[int, int]     # (xlo, ylo) of the cutout in original detector pixels (0-based)
    psf_oversamp: int = 10            # OVERSAMP keyword from the PSF header (default per QR-2)


# --------------------------------------------------------------------------- #
# IRSA cutout service (default)
# --------------------------------------------------------------------------- #

def build_irsa_cutout_url(access_url: str, coord: SkyCoord, size: u.Quantity) -> str:
    """Append IRSA cutout-service parameters to an L2 MEF access URL."""
    ra = coord.icrs.ra.to_value(u.deg)
    dec = coord.icrs.dec.to_value(u.deg)
    size_deg = size.to_value(u.deg)
    sep = "&" if "?" in access_url else "?"
    # IRSA's ibe cutout endpoint requires an explicit unit suffix on
    # ``size`` (accepted units include 'deg', 'arcsec', 'pixel', ...).
    return f"{access_url}{sep}center={ra},{dec}&size={size_deg}deg"


def fetch_irsa_cutout(
    access_url: str,
    coord: SkyCoord,
    size: u.Quantity,
    *,
    cache_dir=None,
) -> CutoutPayload:
    cutout_url = build_irsa_cutout_url(access_url, coord, size)
    with open_fits(cutout_url, mode="http", cache_dir=cache_dir) as hdul:
        return _payload_from_irsa_hdul(hdul)


def _payload_from_irsa_hdul(hdul: fits.HDUList) -> CutoutPayload:
    primary = hdul[0].header.copy()
    image_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
    flags_hdu = hdul["FLAGS"] if "FLAGS" in hdul else hdul[2]
    var_hdu = hdul["VARIANCE"] if "VARIANCE" in hdul else hdul[3]
    zodi_hdu = hdul["ZODI"] if "ZODI" in hdul else hdul[4]
    psf_hdu = hdul["PSF"] if "PSF" in hdul else hdul[5]

    header = image_hdu.header
    crpix1a = int(round(header.get("CRPIX1A", 1)))
    crpix2a = int(round(header.get("CRPIX2A", 1)))
    # CRPIX*A are 1-based pixel positions of cutout (0,0) in the original
    # detector frame.  Convert to a 0-based origin.
    pixel_origin = (crpix1a - 1, crpix2a - 1)

    return CutoutPayload(
        image=np.array(image_hdu.data, copy=True),
        flags=np.array(flags_hdu.data, copy=True),
        variance=np.array(var_hdu.data, copy=True),
        zodi=np.array(zodi_hdu.data, copy=True),
        psf_cube=np.array(psf_hdu.data, copy=True),
        psf_header=psf_hdu.header.copy(),
        image_header=header.copy(),
        primary_header=primary,
        spatial_wcs=WCS(header).celestial,
        detector=int(header.get("DETECTOR", -1)),
        pixel_origin=pixel_origin,
        psf_oversamp=int(psf_hdu.header.get("OVERSAMP", 10)),
    )


# --------------------------------------------------------------------------- #
# fsspec / byte-range backend
# --------------------------------------------------------------------------- #

def fetch_fsspec_cutout(
    target: str,
    coord: SkyCoord,
    size: u.Quantity,
    *,
    fsspec_kwargs: dict | None = None,
) -> CutoutPayload:
    """Crop a SPHEREx L2 MEF on the client using ``.section`` + Cutout2D."""
    with open_fits(target, mode="auto", fsspec_kwargs=fsspec_kwargs) as hdul:
        primary = hdul[0].header.copy()
        image_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        flags_hdu = hdul["FLAGS"] if "FLAGS" in hdul else hdul[2]
        var_hdu = hdul["VARIANCE"] if "VARIANCE" in hdul else hdul[3]
        zodi_hdu = hdul["ZODI"] if "ZODI" in hdul else hdul[4]
        psf_hdu = hdul["PSF"] if "PSF" in hdul else hdul[5]

        wcs_full = WCS(image_hdu.header).celestial
        size_pix = _size_to_pixels(size, wcs_full)

        cut_image = Cutout2D(image_hdu.section, position=coord, size=size_pix,
                             wcs=wcs_full, copy=True, mode="trim")
        sl = cut_image.slices_original  # (y_slice, x_slice)

        flags = np.asarray(flags_hdu.section[sl[0], sl[1]])
        var = np.asarray(var_hdu.section[sl[0], sl[1]])
        zodi = np.asarray(zodi_hdu.section[sl[0], sl[1]])
        psf_cube = np.asarray(psf_hdu.data, copy=True)  # PSF is small, fetch in full

        cropped_header = image_hdu.header.copy()
        cropped_header.update(cut_image.wcs.to_header())
        cropped_header["NAXIS1"] = cut_image.data.shape[1]
        cropped_header["NAXIS2"] = cut_image.data.shape[0]
        # Encode the cutout origin so downstream code can map back to detector pixels.
        cropped_header["CRPIX1A"] = sl[1].start + 1
        cropped_header["CRPIX2A"] = sl[0].start + 1
        pixel_origin = (sl[1].start, sl[0].start)

        return CutoutPayload(
            image=np.asarray(cut_image.data),
            flags=flags,
            variance=var,
            zodi=zodi,
            psf_cube=psf_cube,
            psf_header=psf_hdu.header.copy(),
            image_header=cropped_header,
            primary_header=primary,
            spatial_wcs=cut_image.wcs,
            detector=int(image_hdu.header.get("DETECTOR", -1)),
            pixel_origin=pixel_origin,
            psf_oversamp=int(psf_hdu.header.get("OVERSAMP", 10)),
        )


def _size_to_pixels(size: u.Quantity, wcs: WCS) -> tuple[int, int]:
    """Convert an angular size to a square pixel size for Cutout2D."""
    pscale = np.abs(wcs.proj_plane_pixel_scales()[0]).to(u.arcsec)
    n = int(np.ceil((size.to(u.arcsec) / pscale).value))
    return (n, n)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def fetch_cutout(
    *,
    access_url: str,
    cloud_uri: str,
    coord: SkyCoord,
    size: u.Quantity,
    backend: CutoutBackend = "irsa",
    cache_dir=None,
    fsspec_kwargs: dict | None = None,
) -> CutoutPayload:
    if backend == "irsa":
        return fetch_irsa_cutout(access_url, coord, size, cache_dir=cache_dir)
    if backend == "fsspec":
        target = cloud_uri or access_url
        return fetch_fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs)
    raise ValueError(f"unknown cutout backend: {backend!r}")
