"""One PSF product per detector, shared across cutouts.

The PSF inside every L2 MEF is a per-detector calibration constant that
IRSA's cutout service passes through uncropped, so a small cutout is mostly
PSF, re-sent on every request:

* QR2 (pipeline 6.x): the 121 x 101 x 101 *optical* PSF cube, byte-identical
  to the ``PSF-DATA-CUBE`` of the ``average_psf_D[Det]_spx_cal-psf-...``
  product — 4.94 MB of a ~5.1 MB cutout;
* QR3 / DR1 (pipeline R7): the ``EPSF`` binary table of *effective* PSFs
  (441 rows of 33 x 33 at 5x, 11 x 41 rows on D3), identical to the
  ``epsf_D[Det]_spx_cal-epsf-...`` library — 3.86 MB of a ~4.4 MB cutout.

With ``psf_source="cal"`` the product is taken from the cal file, fetched
once per detector, and each cutout download stops right after the PSF
*header* (which still comes from the L2 file, so the zone table and the
erratum handling are unchanged).

Verification differs by kind. Nothing in the QR2 headers names the PSF cal
version a file was built with, so the optical cube is checked by sampling:
the first cutout of each detector, and every ``verify_every``-th one after
it, is downloaded in full and its cube compared against the cal product;
those cutouts keep their own L2 cube. The R7 ``EPSF`` header carries
``HISTORY Calibration source file: epsf_<det>_<date>.fits``, which the
truncated stream still delivers, so every effective-PSF cutout is checked by
that provenance string (after the same first full download and array
comparison), and a mismatch is caught on the cutout itself. On a mismatch
the detector falls back to full downloads for the rest of the process and
a warning is emitted.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeVar

import numpy as np
from astropy.io import fits

from .io import open_fits

PsfSource = Literal["epsf-cal", "cal", "l2"]
PsfKind = Literal["optical", "effective"]
PSF_VERIFY_EVERY_DEFAULT = 200

#: Cal family per PSF kind.
PSF_FAMILY = {"optical": "average_psf", "effective": "epsf"}
#: Data release whose library ``psf_source="epsf-cal"`` attaches to any image.
EPSF_RELEASE_DEFAULT = "qr3"

_P = TypeVar("_P")


def psf_kind_of_release(data_release: str) -> PsfKind:
    """``qr2`` files carry the optical PSF cube; ``qr3`` and later the R7 ePSF."""
    return "optical" if str(data_release).lower() in ("qr1", "qr2") else "effective"


@dataclass(frozen=True)
class EpsfLibrary:
    """One detector's R7 effective-PSF library: the ``EPSF`` binary table.

    ``table`` is the structured array of the bintable (``BINX``, ``BINY``,
    ``XCENTER``, ``YCENTER``, ``XWIDTH``, ``YWIDTH``, ``NSTAR``,
    ``NEFF_MEAN``, ``CWAVE``, ``CBAND``, ``EPSF``); ``cube`` is its ``EPSF``
    column as ``(n_zones, 33, 33)``; ``source_file`` is the provenance string
    from the header (``HISTORY Calibration source file: ...``), which the L2
    ``EPSF`` extension repeats and which identifies the product.
    """

    table: np.ndarray
    header: fits.Header
    cube: np.ndarray
    source_file: str | None

    @property
    def shape(self):
        return self.cube.shape


def epsf_source_file(header: fits.Header) -> str | None:
    """The ``Calibration source file`` named in an ``EPSF`` header's HISTORY
    (the last such card, should a re-issue append another)."""
    found = None
    for card in header.get("HISTORY", []):
        text = str(card)
        if "Calibration source file" in text:
            found = text.split(":", 1)[1].strip()
    return found


def find_psf_product(
    kind: PsfKind,
    detector: int,
    *,
    data_release: str = "qr2",
    cal_token: str | None = None,
) -> tuple[str, str]:
    """Return ``(http_url, s3_uri)`` for the detector's PSF product of ``kind``.

    Resolved from the local cal tree of ``data_release`` when one is
    configured (see :func:`~spherex_retrieval.cal_index.set_local_cal_roots`),
    else from ``cal_token`` or the IRSA directory listing only.  SIA2 is
    deliberately skipped: a positional ``spherex_qr2_cal`` query does not
    return ``average_psf`` rows, and a position-less one scans the whole
    collection (~3 min). The listing is resolved per detector because a
    version can be re-issued for one detector only (``qr3`` D3 is on
    ``cal-epsf-v2``, the others on ``v1``).
    """
    from .cal_index import (
        cal_http_url,
        cal_s3_uri,
        latest_cal_token_via_listing,
        local_cal_product,
    )

    family = PSF_FAMILY[kind]
    local = local_cal_product(family, detector, data_release=data_release, cal_token=cal_token)
    if local is not None:
        return local[0], ""
    token = cal_token or latest_cal_token_via_listing(
        family, data_release=data_release, detector=detector)
    if token is None:
        raise RuntimeError(
            f"could not list {family} cal products for D{detector} under {data_release} on IRSA"
        )
    return (
        cal_http_url(family, detector, token, data_release=data_release),
        cal_s3_uri(family, detector, token, data_release=data_release),
    )


def find_average_psf_product(detector: int, *, data_release: str = "qr2",
                             cal_token: str | None = None) -> tuple[str, str]:
    """QR2 optical cube; see :func:`find_psf_product`."""
    return find_psf_product("optical", detector, data_release=data_release, cal_token=cal_token)


def load_average_psf_cube(
    cal_target: str,
    *,
    cache_dir=None,
    fsspec_kwargs: dict | None = None,
) -> np.ndarray:
    """Read the ``PSF-DATA-CUBE`` of an ``average_psf`` cal product."""
    with open_fits(cal_target, mode="auto", cache_dir=cache_dir,
                   fsspec_kwargs=fsspec_kwargs) as hdul:
        hdu = hdul["PSF-DATA-CUBE"] if "PSF-DATA-CUBE" in hdul else hdul[1]
        cube = np.array(hdu.data, copy=True)
    cube.flags.writeable = False   # shared by every cutout of the detector
    return cube


def epsf_library_from_hdu(hdu) -> EpsfLibrary:
    """Build an :class:`EpsfLibrary` from an ``EPSF`` binary-table HDU."""
    table = np.array(hdu.data, copy=True)
    n = len(table)
    cube = np.ascontiguousarray(np.asarray(table["EPSF"], dtype=np.float64).reshape(n, 33, 33))
    table.flags.writeable = False
    cube.flags.writeable = False
    return EpsfLibrary(table=table, header=hdu.header.copy(), cube=cube,
                       source_file=epsf_source_file(hdu.header))


def load_epsf_library(
    cal_target: str,
    *,
    cache_dir=None,
    fsspec_kwargs: dict | None = None,
) -> EpsfLibrary:
    """Read the ``EPSF`` table of an ``epsf`` cal product."""
    with open_fits(cal_target, mode="auto", cache_dir=cache_dir,
                   fsspec_kwargs=fsspec_kwargs) as hdul:
        hdu = hdul["EPSF"] if "EPSF" in hdul else hdul[1]
        return epsf_library_from_hdu(hdu)


def load_psf_product(kind: PsfKind, cal_target: str, *, cache_dir=None,
                     fsspec_kwargs: dict | None = None):
    if kind == "optical":
        return load_average_psf_cube(cal_target, cache_dir=cache_dir, fsspec_kwargs=fsspec_kwargs)
    return load_epsf_library(cal_target, cache_dir=cache_dir, fsspec_kwargs=fsspec_kwargs)


def _cube_of_product(product) -> np.ndarray:
    return product.cube if isinstance(product, EpsfLibrary) else product


def cubes_identical(a, b) -> bool:
    a = _cube_of_product(a)
    b = _cube_of_product(b)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


@dataclass
class _DetectorState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    cube: np.ndarray | None = None
    source: str = ""
    verified: bool = False
    disabled: bool = False
    count: int = 0


class SharedPsfRegistry:
    """Per-detector cal products plus the checks against the L2 copy.

    ``kind`` selects the product: the QR2 optical cube (``"optical"``,
    verified by sampled full downloads) or the R7 effective-PSF library
    (``"effective"``, verified by the ``EPSF`` header's provenance string on
    every cutout after the first full download). ``verify=False`` attaches
    the product without any check, for ``psf_source="epsf-cal"``, where the
    L2 file does not carry the product being attached.

    Thread-safe; one instance is meant to live for a whole ``retrieve()``
    call or longer — :func:`get_registry` keeps one per process so a
    many-target campaign pays the first-cutout check once per detector, not
    once per target.
    """

    def __init__(
        self,
        *,
        verify_every: int = PSF_VERIFY_EVERY_DEFAULT,
        cal_token: str | None = None,
        data_release: str = "qr2",
        kind: PsfKind | None = None,
        verify: bool = True,
        cache_dir=None,
        use_s3: bool = False,
        fsspec_kwargs: dict | None = None,
        loader: Callable[[int], tuple[np.ndarray, str]] | None = None,
    ):
        if verify_every < 0:
            raise ValueError("verify_every must be >= 0 (0 = check the first cutout only)")
        self.verify_every = verify_every
        self.cal_token = cal_token
        self.data_release = data_release
        self.kind: PsfKind = kind or psf_kind_of_release(data_release)
        self.verify = verify
        self.cache_dir = cache_dir
        self.use_s3 = use_s3
        self.fsspec_kwargs = fsspec_kwargs
        self._loader = loader or self._load_from_irsa
        self._states: dict[int, _DetectorState] = {}
        self._states_lock = threading.Lock()

    def _load_from_irsa(self, detector: int):
        http_url, s3_uri = find_psf_product(
            self.kind, detector, data_release=self.data_release, cal_token=self.cal_token
        )
        target = s3_uri if (self.use_s3 and s3_uri) else http_url
        product = load_psf_product(
            self.kind, target, cache_dir=self.cache_dir, fsspec_kwargs=self.fsspec_kwargs
        )
        return product, target

    def _state(self, detector: int) -> _DetectorState:
        with self._states_lock:
            return self._states.setdefault(detector, _DetectorState())

    def _source_tag(self, source: str) -> str:
        name = source.rsplit("/", 1)[-1]
        return f"epsf:{name}" if self.kind == "effective" else f"cal:{name}"

    def source_tag(self, detector: int) -> str:
        """The ``PSFSRC`` tag of the product loaded for ``detector``."""
        return self._source_tag(self._state(detector).source)

    def product(self, detector: int):
        """The shared product for ``detector`` (loaded on first use), or
        ``None`` when it could not be loaded."""
        st = self._state(detector)
        with st.lock:
            if st.cube is None and not st.disabled:
                try:
                    st.cube, st.source = self._loader(detector)
                except Exception as exc:
                    st.disabled = True
                    warnings.warn(
                        f"D{detector}: {PSF_FAMILY[self.kind]} cal product unavailable ({exc})",
                        RuntimeWarning, stacklevel=2,
                    )
            return st.cube

    def fetch(
        self,
        detector: int,
        *,
        fetch_full: Callable[[], _P],
        fetch_light: Callable[[np.ndarray], _P],
        cube_of: Callable[[_P], np.ndarray],
        provenance_of: Callable[[_P], str | None] | None = None,
    ) -> tuple[_P, str]:
        """Run one cutout download; return ``(payload, psf_source_tag)``.

        ``fetch_full()`` downloads the cutout with its own PSF product;
        ``fetch_light(product)`` downloads it without and attaches
        ``product``; ``provenance_of(payload)`` (optional) returns the
        product provenance the light payload's PSF header names, checked
        against the shared product's when both exist (the R7 ``EPSF``
        header carries one; the QR2 ``PSF`` header does not).
        """
        st = self._state(detector)
        if not self.verify:
            # attach without any check (the L2 file does not carry this product)
            product = self.product(detector)
            if product is None:
                return fetch_full(), "l2"
            return fetch_light(product), self._source_tag(st.source)
        with st.lock:
            if not st.disabled and not st.verified:
                # First cutout of this detector: others wait here so none of
                # them uses the cal product before it has been checked once.
                payload = fetch_full()
                self._check(st, detector, cube_of(payload), load=True)
                return payload, "l2"
            disabled = st.disabled
            if not disabled:
                st.count += 1
                sample = self.verify_every > 0 and st.count % self.verify_every == 0
                cube, source = st.cube, st.source
        if disabled:
            return fetch_full(), "l2"
        if sample:
            payload = fetch_full()
            with st.lock:
                if not st.disabled:
                    self._check(st, detector, cube_of(payload), load=False)
            return payload, "l2"
        payload = fetch_light(cube)
        expected = getattr(cube, "source_file", None)
        if provenance_of is not None and expected:
            got = provenance_of(payload)
            if got is not None and got != expected:
                with st.lock:
                    st.disabled, st.verified, st.cube = True, False, None
                warnings.warn(
                    f"D{detector}: L2 EPSF header names {got!r}, the shared library is "
                    f"{expected!r}; downloading the PSF with every cutout from here on",
                    RuntimeWarning, stacklevel=2,
                )
                return fetch_full(), "l2"
        return payload, self._source_tag(source)

    def _check(self, st: _DetectorState, detector: int, l2_cube: np.ndarray, *, load: bool) -> None:
        if load:
            try:
                st.cube, st.source = self._loader(detector)
            except Exception as exc:
                st.disabled = True
                warnings.warn(
                    f"D{detector}: {PSF_FAMILY[self.kind]} cal product unavailable ({exc}); "
                    "downloading the PSF with every cutout",
                    RuntimeWarning, stacklevel=3,
                )
                return
        if cubes_identical(st.cube, l2_cube):
            st.verified = True
            return
        was_trusted = st.verified
        st.disabled, st.verified, st.cube = True, False, None
        msg = (f"D{detector}: L2 PSF cube differs from {st.source}; "
               "downloading the PSF cube with every cutout from here on")
        if was_trusted:
            msg += (f" — up to {max(self.verify_every, 1) - 1} D{detector} cutouts since the "
                    "last passing check carry the cal cube (PSFSRC='cal:...') and "
                    "should be re-retrieved with psf_source='l2'")
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def is_disabled(self, detector: int) -> bool:
        return self._state(detector).disabled


_REGISTRIES: dict[tuple, SharedPsfRegistry] = {}
_REGISTRIES_LOCK = threading.Lock()


def get_registry(
    *,
    verify_every: int = PSF_VERIFY_EVERY_DEFAULT,
    cal_token: str | None = None,
    data_release: str = "qr2",
    kind: PsfKind | None = None,
    verify: bool = True,
    cache_dir=None,
    use_s3: bool = False,
    fsspec_kwargs: dict | None = None,
) -> SharedPsfRegistry:
    """Process-wide registry for this cal selection (created on first use)."""
    kind = kind or psf_kind_of_release(data_release)
    key = (data_release, kind, bool(verify), cal_token, use_s3)
    with _REGISTRIES_LOCK:
        reg = _REGISTRIES.get(key)
        if reg is None:
            reg = _REGISTRIES[key] = SharedPsfRegistry(
                verify_every=verify_every, cal_token=cal_token,
                data_release=data_release, kind=kind, verify=verify,
                cache_dir=cache_dir, use_s3=use_s3, fsspec_kwargs=fsspec_kwargs,
            )
        else:
            if verify_every < 0:
                raise ValueError("verify_every must be >= 0 (0 = check the first cutout only)")
            reg.verify_every = verify_every
            reg.cache_dir = cache_dir
            reg.fsspec_kwargs = fsspec_kwargs
        return reg
