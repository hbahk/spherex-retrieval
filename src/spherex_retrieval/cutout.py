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
  client with ``ImageHDU.section`` and the cutout service's own window rule
  (:func:`irsa_window`), so both backends cut the same pixels. Useful when
  running in the same AWS region as the data, or when the IRSA service is
  unavailable.
* ``local`` — the ``fsspec`` crop applied to a file on a local file system
  (``access_url`` is its path; see the ``local`` query backend).

The output is a :class:`CutoutPayload` dataclass that carries the cropped
arrays, the spatial WCS, and the relevant headers.  Wavelength maps are
retrieved separately by :mod:`spherex_retrieval.wavelength`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .io import http_fetch_until, open_fits, url_to_cache_path
from .psf_shared import SharedPsfRegistry

CutoutBackend = Literal["irsa", "fsspec", "local"]


class NoOverlapError(ValueError):
    """The requested box does not overlap the frame (a spare discovery candidate)."""


@dataclass
class CutoutPayload:
    """Container for the spatial side of a SPHEREx cutout."""

    image: np.ndarray
    flags: np.ndarray
    variance: np.ndarray
    zodi: np.ndarray
    psf_cube: np.ndarray              # (n_zones, S, S): the QR2 121-plane optical cube, or the
                                      # R7 ePSF arrays (n_zones, 33, 33); read-only when shared
    psf_header: fits.Header
    image_header: fits.Header         # used for spatial + spectral WCS
    primary_header: fits.Header       # carries VERSION, OBSID, etc. — needed for the PSF erratum check
    spatial_wcs: WCS
    detector: int
    pixel_origin: tuple[int, int]     # (xlo, ylo) of the cutout in original detector pixels (0-based)
    psf_oversamp: int = 10            # OVERSAMP (QR2) / OVSMPX (R7) from the PSF header
    psf_source: str = "l2"            # "l2", "cal:<average_psf file>" or "epsf:<epsf file>" when shared
    psf_kind: str = "optical"         # "optical" (QR2 cube) or "effective" (R7 EPSF table)
    psf_table: np.ndarray | None = None   # the R7 EPSF bintable rows (zone metadata), else None


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
    if psf_registry is None or detector is None:
        return _full()
    # A complete copy already on disk costs nothing to reuse. It carries the
    # file's OWN PSF product, which is what a verifying registry ("cal") would
    # attach anyway; an attaching registry ("epsf-cal" on a QR2 image) still
    # has to swap in its library, or a warm HTTP cache would silently hand
    # back the optical cube.
    if url_to_cache_path(cutout_url, cache_dir=cache_dir).exists():
        payload = _full()
        if psf_registry.verify:
            return payload
        product = psf_registry.product(detector)
        if product is not None:
            for key, val in _psf_fields_from_product(product, payload.psf_header).items():
                setattr(payload, key, val)
            payload.psf_source = psf_registry.source_tag(detector)
        return payload

    payload, source = psf_registry.fetch(
        detector,
        fetch_full=_full,
        fetch_light=lambda product: _fetch_irsa_cutout_without_psf(cutout_url, product),
        cube_of=lambda p: p.psf_cube,
        provenance_of=_psf_provenance,
    )
    payload.psf_source = source
    return payload


def _psf_provenance(payload: CutoutPayload) -> str | None:
    from .psf_shared import epsf_source_file
    return epsf_source_file(payload.psf_header)


def _detector_from_url(access_url: str) -> int | None:
    from .wavelength import parse_l2_filename

    parsed = parse_l2_filename(access_url.split("?", 1)[0])
    return parsed["det"] if parsed else None


def _fetch_irsa_cutout_without_psf(cutout_url: str, psf_product) -> CutoutPayload:
    """Download up to the end of the PSF header, then hang up.

    ``psf_product`` is the shared per-detector product: the QR2 optical cube
    (an array) or an R7 :class:`~spherex_retrieval.psf_shared.EpsfLibrary`.
    """
    scanner = _PsfHeaderScanner()
    body, stopped = http_fetch_until(cutout_url, scanner)
    if not stopped:
        # No PSF header was recognised; the whole MEF is in hand, use it as is.
        with fits.open(io.BytesIO(body)) as hdul:
            return _payload_from_irsa_hdul(hdul)
    psf_start, psf_end = scanner.psf_span
    psf_header = fits.Header.fromstring(body[psf_start:psf_end].decode("ascii"))
    with fits.open(io.BytesIO(body[:psf_start])) as hdul:
        return _payload_from_irsa_hdul(hdul, psf_product=psf_product, psf_header=psf_header)


_FITS_BLOCK = 2880
_END_CARD = b"END" + b" " * 77
#: PSF extension names: ``PSF`` (QR2 optical cube), ``EPSF`` (R7 effective table).
PSF_EXTNAMES = ("PSF", "EPSF")


class _PsfHeaderScanner:
    """Walk the HDU headers of a growing FITS byte buffer up to the PSF one.

    Called with the buffer after every received chunk; returns the offset at
    which the PSF header ends (= where its data would start) once that much
    has arrived, else ``None``.  ``psf_span`` then holds the header's
    ``(start, end)``.
    """

    def __init__(self, extname: str | tuple[str, ...] = PSF_EXTNAMES, fallback_index: int = 5):
        self.extnames = (extname,) if isinstance(extname, str) else tuple(extname)
        self.fallback_index = fallback_index
        self.psf_span: tuple[int, int] | None = None
        self.extname: str | None = None   # the name actually found
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
            if name in self.extnames or (not name and self._index == self.fallback_index):
                self.psf_span = (self._hdu_start, end)
                self.extname = name or self.extnames[0]
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


def _psf_fields_from_hdu(psf_hdu) -> dict:
    """PSF payload fields from a ``PSF`` (QR2 image cube) or ``EPSF`` (R7
    bintable) HDU."""
    from .psf_shared import epsf_library_from_hdu

    name = str(psf_hdu.header.get("EXTNAME", "")).strip().upper()
    if name == "EPSF" or psf_hdu.header.get("XTENSION") == "BINTABLE":
        lib = epsf_library_from_hdu(psf_hdu)
        return _psf_fields_from_product(lib, psf_hdu.header.copy())
    return dict(psf_cube=np.array(psf_hdu.data, copy=True), psf_header=psf_hdu.header.copy(),
                psf_kind="optical", psf_table=None,
                psf_oversamp=int(psf_hdu.header.get("OVERSAMP", 10)))


def _psf_fields_from_product(product, psf_header: fits.Header) -> dict:
    """PSF payload fields from a shared product (cube or EpsfLibrary) and the
    L2 PSF header that came with the cutout."""
    from .psf_shared import EpsfLibrary

    if isinstance(product, EpsfLibrary):
        from .psf_shared import epsf_source_file
        if epsf_source_file(psf_header) is None:
            # the L2 file did not carry this product (psf_source="epsf-cal"):
            # the library's own header is the PSF header of record
            psf_header = product.header.copy()
        ox, oy = psf_header.get("OVSMPX", product.header.get("OVSMPX", 5)), \
            psf_header.get("OVSMPY", product.header.get("OVSMPY", 5))
        if int(ox) != int(oy):
            raise ValueError(f"anisotropic ePSF oversampling {ox}x{oy} is not supported")
        return dict(psf_cube=product.cube, psf_header=psf_header, psf_kind="effective",
                    psf_table=product.table, psf_oversamp=int(ox))
    return dict(psf_cube=product, psf_header=psf_header, psf_kind="optical", psf_table=None,
                psf_oversamp=int(psf_header.get("OVERSAMP", 10)))


def _payload_from_irsa_hdul(
    hdul: fits.HDUList,
    *,
    psf_cube: np.ndarray | None = None,
    psf_product=None,
    psf_header: fits.Header | None = None,
) -> CutoutPayload:
    """Build the payload; ``psf_product`` (or the older ``psf_cube``) and
    ``psf_header`` stand in for a PSF HDU that was deliberately not
    downloaded."""
    primary = hdul[0].header.copy()
    image_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
    flags_hdu = hdul["FLAGS"] if "FLAGS" in hdul else hdul[2]
    var_hdu = hdul["VARIANCE"] if "VARIANCE" in hdul else hdul[3]
    zodi_hdu = hdul["ZODI"] if "ZODI" in hdul else hdul[4]
    if psf_product is None:
        psf_product = psf_cube
    if psf_product is None:
        psf_hdu = _find_psf_hdu(hdul)
        psf_fields = _psf_fields_from_hdu(psf_hdu)
    else:
        psf_fields = _psf_fields_from_product(psf_product, psf_header)

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
        image_header=header.copy(),
        primary_header=primary,
        spatial_wcs=WCS(header).celestial,
        detector=int(header.get("DETECTOR", -1)),
        pixel_origin=pixel_origin,
        **psf_fields,
    )


def _find_psf_hdu(hdul: fits.HDUList):
    for name in PSF_EXTNAMES:
        if name in hdul:
            return hdul[name]
    return hdul[5]


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
    """Crop a SPHEREx L2 MEF on the client (``.section`` reads, IRSA's window)."""
    if detector is None:
        detector = _detector_from_url(target)
    if psf_registry is None or detector is None:
        return _fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs)

    payload, source = psf_registry.fetch(
        detector,
        fetch_full=lambda: _fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs),
        fetch_light=lambda product: _fsspec_cutout(
            target, coord, size, fsspec_kwargs=fsspec_kwargs, psf_product=product),
        cube_of=lambda p: p.psf_cube,
        provenance_of=_psf_provenance,
    )
    payload.psf_source = source
    return payload


def irsa_window(x: float, y: float, n: tuple[int, int],
                shape: tuple[int, int]) -> tuple[slice, slice] | None:
    """The detector window IRSA's cutout service returns, as ``(rows, cols)`` slices.

    ``(x, y)`` is the target's 0-based pixel position (SIP applied), ``n`` the
    box size in pixels per axis ``(nx, ny)``, ``shape`` the frame ``(ny, nx)``.
    The first pixel along an axis is ``floor(c + 1 - n/2)``: for odd ``n`` the
    box is centred on the pixel holding the target, for even ``n`` on the
    pixel corner nearest to it. The box is trimmed to the frame; ``None`` when
    nothing is left. Measured against the service on QR2 frames for
    ``n`` = 10, 12, 15, 16, 17 and 20 (both parities, both halves of a pixel).
    """
    out = []
    for c, m, full in ((y, n[1], shape[0]), (x, n[0], shape[1])):
        lo = int(np.floor(c + 1.0 - m / 2.0))
        a, b = max(lo, 0), min(lo + m, full)
        if b <= a:
            return None
        out.append(slice(a, b))
    return out[0], out[1]


def box_pixels(size: u.Quantity, wcs: WCS) -> tuple[int, int]:
    """``(nx, ny)`` of a ``size`` box: ``round(size / pixel scale)`` per axis.

    The scales are the linear-WCS ones (``proj_plane_pixel_scales``); like
    IRSA's service, an anisotropic scale can give ``nx != ny``.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    scales = np.abs(proj_plane_pixel_scales(wcs)) * 3600.0          # arcsec per pixel
    arcsec = size.to_value(u.arcsec)
    return tuple(max(1, int(np.round(arcsec / s))) for s in scales[:2])


_CRPIX_RE = re.compile(r"^CRPIX([12])([A-Z]?)$")


def _shift_reference_pixels(header: fits.Header, x0: int, y0: int) -> fits.Header:
    """A copy of ``header`` for the window starting at 0-based ``(x0, y0)``.

    Every WCS reference pixel moves with the window, as in the cutout
    service's headers: ``CRPIX1/2`` (celestial, with SIP) and the alternates
    ``CRPIX1A/2A`` (detector pixels: the cutout value ``1 - x0`` reads back as
    the 0-based detector origin) and ``CRPIX1W/2W`` (wavelength). A missing
    ``CRPIX*A`` counts as the full frame's ``1``.
    """
    out = header.copy()
    for axis in ("1", "2"):
        out.setdefault(f"CRPIX{axis}A", 1.0)
    for key in list(out.keys()):
        m = _CRPIX_RE.match(key)
        if m:
            out[key] = float(out[key]) - (x0 if m.group(1) == "1" else y0)
    return out


def _fsspec_cutout(
    target: str,
    coord: SkyCoord,
    size: u.Quantity,
    *,
    fsspec_kwargs: dict | None = None,
    psf_cube: np.ndarray | None = None,
    psf_product=None,
) -> CutoutPayload:
    """Crop the full MEF with IRSA's window rule (:func:`irsa_window`).

    ``psf_product`` (or ``psf_cube``) given: skip the PSF data read and attach
    that product.
    """
    if psf_product is None:
        psf_product = psf_cube
    with open_fits(target, mode="auto", fsspec_kwargs=fsspec_kwargs) as hdul:
        primary = hdul[0].header.copy()
        image_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        flags_hdu = hdul["FLAGS"] if "FLAGS" in hdul else hdul[2]
        var_hdu = hdul["VARIANCE"] if "VARIANCE" in hdul else hdul[3]
        zodi_hdu = hdul["ZODI"] if "ZODI" in hdul else hdul[4]
        psf_hdu = _find_psf_hdu(hdul)

        header = image_hdu.header
        wcs_full = WCS(header).celestial
        x, y = wcs_full.world_to_pixel(coord)          # SIP applied
        full_shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        window = irsa_window(float(x), float(y), box_pixels(size, wcs_full), full_shape)
        if window is None:
            raise NoOverlapError(f"the {size} box around {coord.to_string('decimal')} "
                                 f"does not overlap the frame")
        rows, cols = window

        image = np.asarray(image_hdu.section[rows, cols])
        flags = np.asarray(flags_hdu.section[rows, cols])
        var = np.asarray(var_hdu.section[rows, cols])
        zodi = np.asarray(zodi_hdu.section[rows, cols])
        if psf_product is None:
            psf_fields = _psf_fields_from_hdu(psf_hdu)
        else:
            psf_fields = _psf_fields_from_product(psf_product, psf_hdu.header.copy())

        cropped_header = _shift_reference_pixels(header, cols.start, rows.start)
        cropped_header["NAXIS1"] = image.shape[1]
        cropped_header["NAXIS2"] = image.shape[0]
        pixel_origin = (cols.start, rows.start)

        return CutoutPayload(
            image=image,
            flags=flags,
            variance=var,
            zodi=zodi,
            image_header=cropped_header,
            primary_header=primary,
            spatial_wcs=WCS(cropped_header).celestial,
            detector=int(header.get("DETECTOR", -1)),
            pixel_origin=pixel_origin,
            **psf_fields,
        )


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
    if backend in ("fsspec", "local"):
        target = access_url if backend == "local" else (cloud_uri or access_url)
        return fetch_fsspec_cutout(target, coord, size, fsspec_kwargs=fsspec_kwargs,
                                   psf_registry=psf_registry, detector=detector)
    raise ValueError(f"unknown cutout backend: {backend!r}")
