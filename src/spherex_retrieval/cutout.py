"""Cutout retrieval from a SPHEREx Spectral Image MEF.

Two retrieval backends:

* ``irsa``   — append the IRSA cutout-service query string to the on-prem
  access URL.  The server returns a complete 6-extension MEF whose
  IMAGE/FLAGS/VARIANCE/ZODI HDUs are pre-cropped, the WCS-WAVE table has
  been re-mapped onto the cropped grid, and the PSF cube is passed through
  unchanged.  This is the simplest and the default mode.  The uncropped PSF
  cube is ~97 % of a small cutout's bytes; given a
  :class:`~spherex_retrieval.psf_shared.SharedPsfRegistry` the download
  stops right after the PSF header and the cube comes from the per-detector
  ``average_psf`` cal product instead.
* ``fsspec`` — open the full L2 MEF (HTTP byte-range or S3) and crop on the
  client using ``ImageHDU.section`` + :class:`~astropy.nddata.Cutout2D`.
  Useful when running in the same AWS region as the data, or when the IRSA
  service is unavailable.

The output is a :class:`CutoutPayload` dataclass that carries the cropped
arrays, the spatial WCS, and the relevant headers.  Wavelength maps are
retrieved separately by :mod:`spherex_retrieval.wavelength`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from .io import http_fetch_until, open_fits, url_to_cache_path
from .psf_shared import SharedPsfRegistry

CutoutBackend = Literal["irsa", "fsspec"]


@dataclass
class CutoutPayload:
    """Container for the spatial side of a SPHEREx cutout."""

    image: np.ndarray
    flags: np.ndarray
    variance: np.ndarray
    zodi: np.ndarray
    psf_cube: np.ndarray              # full 121-plane cube (read-only when shared from the cal product)
    psf_header: fits.Header
    image_header: fits.Header         # used for spatial + spectral WCS
    primary_header: fits.Header       # carries VERSION, OBSID, etc. — needed for the PSF erratum check
    spatial_wcs: WCS
    detector: int
    pixel_origin: tuple[int, int]     # (xlo, ylo) of the cutout in original detector pixels (0-based)
    psf_oversamp: int = 10            # OVERSAMP keyword from the PSF header (default per QR-2)
    psf_source: str = "l2"            # "l2", or "cal:<average_psf file>" when the cube was shared


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
    psf_registry: SharedPsfRegistry | None = None,
    detector: int | None = None,
) -> CutoutPayload:
    cutout_url = build_irsa_cutout_url(access_url, coord, size)

    def _full() -> CutoutPayload:
        with open_fits(cutout_url, mode="http", cache_dir=cache_dir) as hdul:
            return _payload_from_irsa_hdul(hdul)

    if detector is None:
        detector = _detector_from_url(access_url)
    # A complete copy already on disk costs nothing to reuse.
    if (psf_registry is None or detector is None
            or url_to_cache_path(cutout_url, cache_dir=cache_dir).exists()):
        return _full()

    payload, source = psf_registry.fetch(
        detector,
        fetch_full=_full,
        fetch_light=lambda cube: _fetch_irsa_cutout_without_psf(cutout_url, cube),
        cube_of=lambda p: p.psf_cube,
    )
    payload.psf_source = source
    return payload


def _detector_from_url(access_url: str) -> int | None:
    from .wavelength import parse_l2_filename

    parsed = parse_l2_filename(access_url.split("?", 1)[0])
    return parsed["det"] if parsed else None


def _fetch_irsa_cutout_without_psf(cutout_url: str, psf_cube: np.ndarray) -> CutoutPayload:
    """Download up to the end of the PSF header, then hang up."""
    scanner = _PsfHeaderScanner()
    body, stopped = http_fetch_until(cutout_url, scanner)
    if not stopped:
        # No PSF header was recognised; the whole MEF is in hand, use it as is.
        with fits.open(io.BytesIO(body)) as hdul:
            return _payload_from_irsa_hdul(hdul)
    psf_start, psf_end = scanner.psf_span
    psf_header = fits.Header.fromstring(body[psf_start:psf_end].decode("ascii"))
    with fits.open(io.BytesIO(body[:psf_start])) as hdul:
        return _payload_from_irsa_hdul(hdul, psf_cube=psf_cube, psf_header=psf_header)


_FITS_BLOCK = 2880
_END_CARD = b"END" + b" " * 77


class _PsfHeaderScanner:
    """Walk the HDU headers of a growing FITS byte buffer up to the PSF one.

    Called with the buffer after every received chunk; returns the offset at
    which the PSF header ends (= where its data would start) once that much
    has arrived, else ``None``.  ``psf_span`` then holds the header's
    ``(start, end)``.
    """

    def __init__(self, extname: str = "PSF", fallback_index: int = 5):
        self.extname = extname
        self.fallback_index = fallback_index
        self.psf_span: tuple[int, int] | None = None
        self._hdu_start = 0      # offset of the header being read
        self._block = 0          # next header block to look for END in
        self._index = 0

    def __call__(self, buf: bytearray) -> int | None:
        while True:
            end = self._header_end(buf)
            if end is None:
                return None
            header = fits.Header.fromstring(bytes(buf[self._hdu_start:end]).decode("ascii"))
            name = str(header.get("EXTNAME", "")).strip().upper()
            if name == self.extname or (not name and self._index == self.fallback_index):
                self.psf_span = (self._hdu_start, end)
                return end
            self._hdu_start = self._block = end + _padded_data_size(header)
            self._index += 1

    def _header_end(self, buf: bytearray) -> int | None:
        while self._block + _FITS_BLOCK <= len(buf):
            block = buf[self._block:self._block + _FITS_BLOCK]
            self._block += _FITS_BLOCK
            if any(block[i:i + 80] == _END_CARD for i in range(0, _FITS_BLOCK, 80)):
                return self._block
        return None


def _padded_data_size(header: fits.Header) -> int:
    naxis = int(header.get("NAXIS", 0))
    if naxis == 0:
        return 0
    n = 1
    for i in range(1, naxis + 1):
        n *= int(header[f"NAXIS{i}"])
    nbytes = (abs(int(header["BITPIX"])) // 8) * int(header.get("GCOUNT", 1)) * (
        int(header.get("PCOUNT", 0)) + n)
    return -(-nbytes // _FITS_BLOCK) * _FITS_BLOCK


def _payload_from_irsa_hdul(
    hdul: fits.HDUList,
    *,
    psf_cube: np.ndarray | None = None,
    psf_header: fits.Header | None = None,
) -> CutoutPayload:
    """Build the payload; ``psf_cube``/``psf_header`` stand in for a PSF HDU
    that was deliberately not downloaded."""
    primary = hdul[0].header.copy()
    image_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
    flags_hdu = hdul["FLAGS"] if "FLAGS" in hdul else hdul[2]
    var_hdu = hdul["VARIANCE"] if "VARIANCE" in hdul else hdul[3]
    zodi_hdu = hdul["ZODI"] if "ZODI" in hdul else hdul[4]
    if psf_cube is None:
        psf_hdu = hdul["PSF"] if "PSF" in hdul else hdul[5]
        psf_cube = np.array(psf_hdu.data, copy=True)
        psf_header = psf_hdu.header.copy()

    header = image_hdu.header
    crpix1a = int(round(header.get("CRPIX1A", 1)))
    crpix2a = int(round(header.get("CRPIX2A", 1)))
    # CRPIX*A give the cutout's alternate 'A' (detector-pixel) WCS, whose
    # reference point sits at detector pixel (1,1).  The 0-based detector
    # origin of cutout pixel (0,0) is therefore -(CRPIX*A - 1), i.e. the
    # negation of the naive (CRPIX*A - 1): equivalently the (ix,iy) -> detector
    # mapping is (-(CRPIX1A-1)+ix, -(CRPIX2A-1)+iy), matching cutout_to_orig()
    # and verified to the decimal against the L2 spatial WCS.  For off-origin
    # cutouts CRPIX*A is <= 1 (often strongly negative, e.g. -1839), so the old
    # (CRPIX*A - 1) made the origin negative and the cal-product crop
    # section[ylo:ylo+ny] negative-wrapped to the vertically MIRRORED detector
    # rows -> a within-detector CWAVE reversal (and mis-picked PSF zones / SAPM).
    pixel_origin = (1 - crpix1a, 1 - crpix2a)

    return CutoutPayload(
        image=np.array(image_hdu.data, copy=True),
        flags=np.array(flags_hdu.data, copy=True),
        variance=np.array(var_hdu.data, copy=True),
        zodi=np.array(zodi_hdu.data, copy=True),
        psf_cube=psf_cube,
        psf_header=psf_header,
        image_header=header.copy(),
        primary_header=primary,
        spatial_wcs=WCS(header).celestial,
        detector=int(header.get("DETECTOR", -1)),
        pixel_origin=pixel_origin,
        psf_oversamp=int(psf_header.get("OVERSAMP", 10)),
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
    psf_registry: SharedPsfRegistry | None = None,
    detector: int | None = None,
) -> CutoutPayload:
    """Crop a SPHEREx L2 MEF on the client using ``.section`` + Cutout2D."""
    if detector is None:
        detector = _detector_from_url(target)
    if psf_registry is None or detector is None:
        return _fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs)

    payload, source = psf_registry.fetch(
        detector,
        fetch_full=lambda: _fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs),
        fetch_light=lambda cube: _fsspec_cutout(
            target, coord, size, fsspec_kwargs=fsspec_kwargs, psf_cube=cube),
        cube_of=lambda p: p.psf_cube,
    )
    payload.psf_source = source
    return payload


def _fsspec_cutout(
    target: str,
    coord: SkyCoord,
    size: u.Quantity,
    *,
    fsspec_kwargs: dict | None = None,
    psf_cube: np.ndarray | None = None,
) -> CutoutPayload:
    """``psf_cube`` given: skip the 4.9 MB PSF read and attach that cube."""
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
        if psf_cube is None:
            psf_cube = np.asarray(psf_hdu.data, copy=True)

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
    psf_registry: SharedPsfRegistry | None = None,
    detector: int | None = None,
) -> CutoutPayload:
    """``psf_registry`` given: take the PSF cube from the per-detector cal
    product instead of downloading it with the cutout (see
    :mod:`spherex_retrieval.psf_shared`)."""
    if backend == "irsa":
        return fetch_irsa_cutout(access_url, coord, size, cache_dir=cache_dir,
                                 psf_registry=psf_registry, detector=detector)
    if backend == "fsspec":
        target = cloud_uri or access_url
        return fetch_fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs,
                                   psf_registry=psf_registry, detector=detector)
    raise ValueError(f"unknown cutout backend: {backend!r}")
