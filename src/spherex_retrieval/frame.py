"""Every target's cutout from one pass over a local L2 file.

A frame (one L2 MEF) that holds many targets is read once. Its headers are
walked with raw ``os.pread`` calls — the MEF layout varies (five file sizes on
the QR2 archive, the headers differing in length), so the offsets come from
the headers and never from a table of sizes. Then either the four pixel planes
are read whole in one request (``"planes"``) or, for a frame with few targets,
only the detector rows each target needs (``"bands"``).

Each target's window, header and WCS follow the cutout service's rules, as the
``local`` cutout backend does (:func:`spherex_retrieval.cutout.irsa_window`),
and the bundle is finished by the same code as :func:`spherex_retrieval.retrieve`
(:func:`spherex_retrieval.core.complete_bundle`: flux correction, PSF header
erratum, zone subset, wavelength and SAPM crops). A bundle from here is
therefore bit-identical to the one the per-target path writes. The PSF product
comes from the shared per-detector registry; the PSF plane itself is read only
when the registry asks for a full cutout (its verification sample) or for
``psf_source="l2"``.

:func:`retrieve_catalog` drives a whole catalogue: it pairs targets with frames
from the archive index, reads the frames file-major on a thread pool and yields
``(target index, Bundle)``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .bundle import Bundle, RetrievalStatus, cutout_filename, write_bundle, write_summary
from .cutout import (
    CutoutPayload,
    _find_psf_hdu,
    _padded_data_size,
    _psf_fields_from_hdu,
    _psf_fields_from_product,
    _psf_provenance,
    _shift_reference_pixels,
    box_pixels,
    irsa_window,
)
from .psf import ZONE_MARGIN_DEFAULT
from .psf_shared import EPSF_RELEASE_DEFAULT, PSF_VERIFY_EVERY_DEFAULT, PsfSource

AccessStrategy = Literal["auto", "planes", "bands"]

_BLOCK = 2880
_END_CARD = b"END" + b" " * 77
#: Bytes asked for when a header's length is unknown; the longest L2 header
#: (the QR2 PSF zone table) is 43,200 bytes.
HEADER_PROBE = 65536
PIXEL_PLANES = ("IMAGE", "FLAGS", "VARIANCE", "ZODI")
_PSF_EXTNAMES = ("PSF", "EPSF")
#: Targets per frame below which ``access_strategy="auto"`` reads row bands
#: instead of whole planes (the crossover measured on olaf, 2026-09-02: about 9).
BANDS_BELOW = 9
_DTYPES = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


class FrameLayoutError(ValueError):
    """The file is not the expected L2 HDU sequence; read it with astropy instead."""


@dataclass
class _Hdu:
    header: fits.Header
    data_offset: int
    shape: tuple[int, int]
    dtype: str

    @property
    def row_bytes(self) -> int:
        return self.shape[1] * np.dtype(self.dtype).itemsize

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.row_bytes


def _header_end(buf: bytes) -> int | None:
    """Length of the header at the start of ``buf`` (whole blocks), or ``None`` if its END is not in ``buf``."""
    pos = 0
    while pos + _BLOCK <= len(buf):
        block = buf[pos:pos + _BLOCK]
        pos += _BLOCK
        i = block.find(_END_CARD)
        while i != -1 and i % 80:
            i = block.find(_END_CARD, i + 1)
        if i != -1:
            return pos
    return None


class _Reader:
    """``pread`` with one resident span, so headers inside a big read cost no request."""

    def __init__(self, fd: int):
        self.fd = fd
        self.requests = 0
        self.nbytes = 0
        self.span_offset = 0
        self.span = b""

    def _pread(self, n: int, offset: int) -> bytes:
        data = os.pread(self.fd, n, offset)
        self.requests += 1
        self.nbytes += len(data)
        return data

    def read(self, offset: int, n: int) -> bytes:
        rel = offset - self.span_offset
        if 0 <= rel and rel + n <= len(self.span):
            return bytes(memoryview(self.span)[rel:rel + n])
        return self._pread(n, offset)

    def load(self, offset: int, n: int) -> None:
        """Make ``[offset, offset + n)`` resident (one request)."""
        self.span = self._pread(n, offset)
        self.span_offset = offset

    def header(self, offset: int) -> tuple[fits.Header, int]:
        """The header at ``offset`` and the offset of its data."""
        rel = offset - self.span_offset
        if 0 <= rel < len(self.span):
            chunk = bytes(memoryview(self.span)[rel:rel + HEADER_PROBE])
            end = _header_end(chunk)
            if end is not None:
                return fits.Header.fromstring(chunk[:end].decode("ascii")), offset + end
        n = HEADER_PROBE
        while True:
            chunk = self._pread(n, offset)
            end = _header_end(chunk)
            if end is not None:
                return fits.Header.fromstring(chunk[:end].decode("ascii")), offset + end
            if len(chunk) < n:
                raise FrameLayoutError(f"no END card after byte {offset}")
            n *= 2


def _image_hdu(header: fits.Header, data_offset: int, name: str) -> _Hdu:
    got = str(header.get("EXTNAME", "")).strip().upper()
    if got != name:
        raise FrameLayoutError(f"expected HDU {name}, found {got or '(unnamed)'}")
    bitpix = int(header["BITPIX"])
    if int(header.get("NAXIS", 0)) != 2 or bitpix not in _DTYPES:
        raise FrameLayoutError(f"{name}: not a 2-D image of a known BITPIX")
    if float(header.get("BSCALE", 1.0)) != 1.0 or float(header.get("BZERO", 0.0)) != 0.0:
        raise FrameLayoutError(f"{name}: scaled data (BSCALE/BZERO)")
    return _Hdu(header, data_offset, (int(header["NAXIS2"]), int(header["NAXIS1"])),
                _DTYPES[bitpix])


class LocalFrame:
    """Headers and pixel planes of one local L2 MEF, read with raw ``pread``.

    Opening reads the PRIMARY and IMAGE headers (one request).
    :meth:`load` then reads the pixels: ``"planes"`` makes one request for the
    IMAGE through ZODI planes and the PSF header after them; ``"bands"`` walks
    the three remaining headers (one small request each) and reads the given
    detector row ranges of every plane. :meth:`cut` hands out copies.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            self._r = _Reader(self._fd)
            self._r.load(0, HEADER_PROBE)          # PRIMARY and IMAGE headers: one request
            self.primary_header, pos = self._r.header(0)
            pos += _padded_data_size(self.primary_header)
            header, data_offset = self._r.header(pos)
            self.hdus = {"IMAGE": _image_hdu(header, data_offset, "IMAGE")}
        except Exception:
            os.close(self._fd)
            raise
        self.image_header = header
        self.shape = self.hdus["IMAGE"].shape
        self.wcs = WCS(header).celestial
        self.psf_header: fits.Header | None = None
        self._planes: dict[str, np.ndarray] = {}
        self._bands: dict[str, list[tuple[int, int, np.ndarray]]] = {}
        self.strategy: str | None = None

    # -- life cycle ---------------------------------------------------------
    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            self._r.span = b""

    def __enter__(self) -> LocalFrame:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def requests(self) -> int:
        return self._r.requests

    @property
    def bytes_read(self) -> int:
        return self._r.nbytes

    # -- reads --------------------------------------------------------------
    def _walk_rest(self) -> None:
        """FLAGS, VARIANCE, ZODI headers, then the PSF header (served from a resident span when inside it)."""
        img = self.hdus["IMAGE"]
        pos = img.data_offset + _padded_data_size(img.header)
        for name in PIXEL_PLANES[1:]:
            header, data_offset = self._r.header(pos)
            hdu = _image_hdu(header, data_offset, name)
            if hdu.shape != img.shape:
                raise FrameLayoutError(f"{name} is {hdu.shape}, IMAGE is {img.shape}")
            self.hdus[name] = hdu
            pos = data_offset + _padded_data_size(header)
        header, _ = self._r.header(pos)
        if str(header.get("EXTNAME", "")).strip().upper() not in _PSF_EXTNAMES:
            raise FrameLayoutError("no PSF/EPSF HDU after ZODI")
        self.psf_header = header

    def load(self, strategy: str, rows: list[tuple[int, int]] | None = None) -> None:
        """Read the pixels: ``"planes"`` (whole planes) or ``"bands"`` (the given row ranges)."""
        img = self.hdus["IMAGE"]
        if strategy == "planes":
            plane = _padded_data_size(img.header)
            self._r.load(img.data_offset, 4 * plane + 4 * HEADER_PROBE)
            self._walk_rest()
            for name in PIXEL_PLANES:
                hdu = self.hdus[name]
                rel = hdu.data_offset - self._r.span_offset
                if rel + hdu.nbytes <= len(self._r.span):
                    arr = np.frombuffer(self._r.span, dtype=hdu.dtype,
                                        count=hdu.shape[0] * hdu.shape[1], offset=rel)
                else:  # longer headers than the span allowed for: read this plane alone
                    arr = np.frombuffer(self._r._pread(hdu.nbytes, hdu.data_offset), dtype=hdu.dtype)
                self._planes[name] = arr.reshape(hdu.shape)
        elif strategy == "bands":
            self._walk_rest()
            for r0, r1 in _merge_ranges(rows or []):
                for name in PIXEL_PLANES:
                    hdu = self.hdus[name]
                    raw = self._r.read(hdu.data_offset + r0 * hdu.row_bytes, (r1 - r0) * hdu.row_bytes)
                    band = np.frombuffer(raw, dtype=hdu.dtype).reshape(r1 - r0, hdu.shape[1])
                    self._bands.setdefault(name, []).append((r0, r1, band))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy

    def cut(self, name: str, rows: slice, cols: slice) -> np.ndarray:
        """A copy of plane ``name`` over ``rows`` x ``cols`` (detector pixels)."""
        if name in self._planes:
            return np.array(self._planes[name][rows, cols])
        for r0, r1, band in self._bands.get(name, []):
            if r0 <= rows.start and rows.stop <= r1:
                return np.array(band[rows.start - r0:rows.stop - r0, cols])
        raise KeyError(f"{name} rows {rows.start}:{rows.stop} were not read")

    def psf_fields_full(self) -> dict:
        """The file's own PSF product (read with astropy: the rare, verifying path)."""
        with fits.open(self.path, memmap=False, lazy_load_hdus=True) as hdul:
            return _psf_fields_from_hdu(_find_psf_hdu(hdul))


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for r0, r1 in sorted(ranges):
        if out and r0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], r1)
        else:
            out.append([r0, r1])
    return [(a, b) for a, b in out]


# --------------------------------------------------------------------------- #
# One frame, many targets
# --------------------------------------------------------------------------- #

def iter_frame_cutouts(
    frame_path: str | os.PathLike,
    coords: SkyCoord,
    size: u.Quantity,
    *,
    obs_id: str,
    detector: int,
    collection: str,
    time_bounds_lower: float = float("nan"),
    access_strategy: AccessStrategy = "auto",
    psf_registry=None,
    data_release: str = "qr2",
    calibration_release: str | None = None,
    gain_correction: bool = False,
    include_wavelength: bool = True,
    include_sapm: bool = False,
    sapm_cal_token: str | None = None,
    subset_psf: bool = True,
    zone_margin: int = ZONE_MARGIN_DEFAULT,
    cache_dir=None,
    stats: dict | None = None,
) -> Iterator[Bundle]:
    """Yield one :class:`Bundle` per target in ``coords`` from a single read of the frame.

    Targets whose box misses the frame get status ``out_of_bounds``. The
    bundles match :func:`spherex_retrieval.retrieve` with
    ``cutout_backend="local"`` bit for bit. ``stats`` (a dict), if given,
    receives the strategy, requests, bytes and seconds spent reading.
    """
    from .core import complete_bundle

    if coords.isscalar:
        coords = coords.reshape((1,))
    path = str(frame_path)

    def bundle_for(coord) -> Bundle:
        return Bundle(obs_id=obs_id, detector=int(detector), collection=collection,
                      access_url=path, cloud_uri="", time_bounds_lower=float(time_bounds_lower),
                      coord_ra=coord.icrs.ra.to_value(u.deg), coord_dec=coord.icrs.dec.to_value(u.deg))

    t0 = time.perf_counter()
    try:
        frame = LocalFrame(path)
    except FrameLayoutError:
        yield from _astropy_fallback(path, coords, size, bundle_for, complete_bundle, locals())
        return
    with frame:
        n_xy = box_pixels(size, frame.wcs)
        windows = []
        for coord in coords:
            # one target at a time: all_world2pix iterates a batch until every
            # member converges, which moves an early one by up to its tolerance
            x, y = frame.wcs.world_to_pixel(coord)
            windows.append(irsa_window(float(x), float(y), n_xy, frame.shape))
        inside = [w for w in windows if w is not None]
        strategy = access_strategy
        if strategy == "auto":
            strategy = "bands" if len(inside) < BANDS_BELOW else "planes"
        if inside:
            try:
                frame.load(strategy, rows=[(w[0].start, w[0].stop) for w in inside])
            except FrameLayoutError:
                frame.close()
                yield from _astropy_fallback(path, coords, size, bundle_for, complete_bundle,
                                             locals())
                return
        t_read = time.perf_counter() - t0
        if stats is not None:
            stats.update(strategy=strategy if inside else "none", requests=frame.requests,
                         bytes=frame.bytes_read, read_s=t_read, targets=len(windows),
                         inside=len(inside))

        full_fields = None

        def fields_full():
            nonlocal full_fields
            if full_fields is None:
                full_fields = frame.psf_fields_full()
            return dict(full_fields)

        for coord, window in zip(coords, windows):
            bundle = bundle_for(coord)
            if window is None:
                bundle.status = RetrievalStatus.OUT_OF_BOUNDS
                bundle.message = (f"cutout failed: the {size} box around "
                                  f"{coord.to_string('decimal')} does not overlap the frame")
                yield bundle
                continue
            rows, cols = window

            def make(psf_fields, rows=rows, cols=cols):
                header = _shift_reference_pixels(frame.image_header, cols.start, rows.start)
                image = frame.cut("IMAGE", rows, cols)
                header["NAXIS1"] = image.shape[1]
                header["NAXIS2"] = image.shape[0]
                return CutoutPayload(
                    image=image, flags=frame.cut("FLAGS", rows, cols),
                    variance=frame.cut("VARIANCE", rows, cols), zodi=frame.cut("ZODI", rows, cols),
                    image_header=header, primary_header=frame.primary_header.copy(),
                    spatial_wcs=WCS(header).celestial,
                    detector=int(frame.image_header.get("DETECTOR", -1)),
                    pixel_origin=(cols.start, rows.start), **psf_fields)

            try:
                if psf_registry is None:
                    payload = make(fields_full())
                else:
                    payload, source = psf_registry.fetch(
                        int(detector),
                        fetch_full=lambda: make(fields_full()),
                        fetch_light=lambda product: make(
                            _psf_fields_from_product(product, frame.psf_header.copy())),
                        cube_of=lambda p: p.psf_cube,
                        provenance_of=_psf_provenance,
                    )
                    payload.psf_source = source
            except Exception as exc:  # noqa: BLE001 - one target must not lose the frame
                bundle.status = RetrievalStatus.DOWNLOAD_FAILED
                bundle.message = f"cutout failed: {exc}"
                yield bundle
                continue
            bundle.cutout = payload
            yield complete_bundle(
                bundle, coord=coord, cutout_backend="local",
                include_wavelength=include_wavelength, include_sapm=include_sapm,
                sapm_cal_token=sapm_cal_token, subset_psf=subset_psf, zone_margin=zone_margin,
                data_release=data_release, calibration_release=calibration_release,
                gain_correction=gain_correction, cache_dir=cache_dir, fsspec_kwargs=None,
                query_backend="local")


def _astropy_fallback(path, coords, size, bundle_for, complete_bundle, kw) -> Iterator[Bundle]:
    """A file whose layout the raw reader does not know: the per-target local backend."""
    from .cutout import NoOverlapError, fetch_cutout

    for coord in coords:
        bundle = bundle_for(coord)
        try:
            bundle.cutout = fetch_cutout(access_url=path, cloud_uri="", coord=coord, size=size,
                                         backend="local", psf_registry=kw["psf_registry"],
                                         detector=bundle.detector)
        except NoOverlapError as exc:
            bundle.status = RetrievalStatus.OUT_OF_BOUNDS
            bundle.message = f"cutout failed: {exc}"
            yield bundle
            continue
        except Exception as exc:  # noqa: BLE001
            bundle.status = RetrievalStatus.DOWNLOAD_FAILED
            bundle.message = f"cutout failed: {exc}"
            yield bundle
            continue
        yield complete_bundle(
            bundle, coord=coord, cutout_backend="local",
            include_wavelength=kw["include_wavelength"], include_sapm=kw["include_sapm"],
            sapm_cal_token=kw["sapm_cal_token"], subset_psf=kw["subset_psf"],
            zone_margin=kw["zone_margin"], data_release=kw["data_release"],
            calibration_release=kw["calibration_release"], gain_correction=kw["gain_correction"],
            cache_dir=kw["cache_dir"], fsspec_kwargs=None, query_backend="local")


# --------------------------------------------------------------------------- #
# A catalogue
# --------------------------------------------------------------------------- #

def retrieve_catalog(
    coords: SkyCoord,
    size: u.Quantity,
    *,
    index,
    archive_root: str | os.PathLike | None = None,
    release: str | None = None,
    psf_source: PsfSource = "epsf-cal",
    psf_verify_every: int = PSF_VERIFY_EVERY_DEFAULT,
    psf_cal_token: str | None = None,
    epsf_release: str = EPSF_RELEASE_DEFAULT,
    calibration_release: str | None = "qr3",
    gain_correction: bool = True,
    include_wavelength: bool = True,
    include_sapm: bool = False,
    sapm_cal_token: str | None = None,
    subset_psf: bool = True,
    zone_margin: int = ZONE_MARGIN_DEFAULT,
    access_strategy: AccessStrategy = "auto",
    margin_pix: float = 10.0,
    max_workers: int = 8,
    cal_roots: dict | None = None,
    cache_dir=None,
    frame_stats: list | None = None,
) -> Iterator[tuple[int, Bundle]]:
    """Cut every target of a catalogue out of a local archive, reading each frame once.

    Targets are paired with frames by
    :func:`~spherex_retrieval.index.find_overlapping_many`; the frames are
    read in index (directory) order on ``max_workers`` threads, and
    ``(target index, Bundle)`` pairs are yielded frame by frame. The options
    mean what they mean in :func:`spherex_retrieval.retrieve` (same defaults:
    the R7 ePSF and R7 calibration on every image). ``frame_stats`` (a list),
    if given, receives one dict per frame with the read strategy, requests,
    bytes and seconds.
    """
    from . import index as sidx
    from .cal_index import set_local_cal_roots
    from .core import psf_registry_for

    if cal_roots is not None:
        set_local_cal_roots(cal_roots)
    meta = {}
    if isinstance(index, (str, os.PathLike)):
        meta = sidx.index_metadata(index)
        index = sidx.read_index(index)
    else:
        meta = sidx.index_metadata(index)
    root = Path(archive_root or meta.get("archive_root") or "")
    release = release or meta.get("release")
    if not release:
        raise ValueError("the index records no release; pass release= (e.g. 'qr2')")
    if coords.isscalar:
        coords = coords.reshape((1,))
    ra = coords.icrs.ra.to_value(u.deg)
    dec = coords.icrs.dec.to_value(u.deg)
    pairs = sidx.find_overlapping_many(ra, dec, size.to_value(u.arcsec), index,
                                       margin_pix=margin_pix)
    frames, starts = np.unique(pairs["frame"], return_index=True)
    ends = list(starts[1:]) + [len(pairs)]
    registry = psf_registry_for(release, psf_source=psf_source, psf_verify_every=psf_verify_every,
                                psf_cal_token=psf_cal_token, epsf_release=epsf_release,
                                cache_dir=cache_dir)
    cols = {c: index.column(c) for c in ("path", "obs_id", "detector", "t_min")}

    def one(k: int) -> list[tuple[int, Bundle]]:
        f = int(frames[k])
        targets = pairs["target"][starts[k]:ends[k]]
        st: dict = {"frame": f}
        out = list(zip(targets.tolist(), iter_frame_cutouts(
            root / cols["path"][f].as_py(), coords[targets], size,
            obs_id=cols["obs_id"][f].as_py(), detector=int(cols["detector"][f].as_py()),
            collection=f"spherex_{release}", time_bounds_lower=float(cols["t_min"][f].as_py()),
            access_strategy=access_strategy, psf_registry=registry, data_release=release,
            calibration_release=calibration_release or release, gain_correction=gain_correction,
            include_wavelength=include_wavelength, include_sapm=include_sapm,
            sapm_cal_token=sapm_cal_token, subset_psf=subset_psf, zone_margin=zone_margin,
            cache_dir=cache_dir, stats=st)))
        if frame_stats is not None:
            frame_stats.append(st)
        return out

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        # a bounded window of frames in flight keeps at most that many planes in memory
        window = max(1, 2 * max_workers)
        pending = []
        k = 0
        while k < len(frames) or pending:
            while k < len(frames) and len(pending) < window:
                pending.append(ex.submit(one, k))
                k += 1
            for item in pending.pop(0).result():
                yield item


def write_catalog_bundles(results, output_dir: str | os.PathLike) -> dict[int, Path]:
    """Write ``(target index, Bundle)`` pairs as one directory per target.

    Each ``target_<index>/`` looks like a :func:`spherex_retrieval.retrieve`
    output: ``cutout_<k>_<obs>_D<det>.fits`` numbered in time order, and
    ``summary.ecsv``. Returns ``{target index: directory}``.
    """
    per_target: dict[int, list[Bundle]] = {}
    for t, b in results:
        per_target.setdefault(int(t), []).append(b)
    out = {}
    for t, bundles in per_target.items():
        d = Path(output_dir) / f"target_{t:06d}"
        d.mkdir(parents=True, exist_ok=True)
        bundles.sort(key=lambda b: (b.time_bounds_lower, b.detector))
        for i, b in enumerate(bundles, start=1):
            if b.is_ok and b.cutout is not None:
                write_bundle(b, d / cutout_filename(b, i))
        write_summary(bundles, d / "summary.ecsv")
        out[t] = d
    return out
