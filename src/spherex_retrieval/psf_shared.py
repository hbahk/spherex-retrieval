"""One PSF cube per detector, shared across cutouts.

The 121 x 101 x 101 PSF cube inside every L2 MEF is a per-detector
calibration constant: it is byte-identical to the ``PSF-DATA-CUBE`` of the
standalone ``average_psf_D[Det]_spx_cal-psf-...`` product, and IRSA's cutout
service passes it through uncropped.  A small cutout is therefore ~97 % PSF
(4.94 MB of ~5.1 MB), re-sent on every request.

With ``psf_source="cal"`` the cube is taken from the cal product, fetched
once per detector, and each cutout download stops right after the PSF
*header* (which still comes from the L2 file, so the zone table and the
erratum handling are unchanged).

Nothing in the L2 headers names the PSF cal version a file was built with,
so the identity cannot be checked without downloading a cube.  It is
instead sampled: the first cutout of each detector, and every
``verify_every``-th one after it, is downloaded in full and its cube
compared against the cal product.  Those cutouts keep their own L2 cube.
On a mismatch the detector falls back to full downloads for the rest of
the process and a warning is emitted.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeVar

import numpy as np

from .io import open_fits

PsfSource = Literal["cal", "l2"]
PSF_VERIFY_EVERY_DEFAULT = 200

_P = TypeVar("_P")


def find_average_psf_product(
    detector: int,
    *,
    data_release: str = "qr2",
    cal_token: str | None = None,
) -> tuple[str, str]:
    """Return ``(http_url, s3_uri)`` for the detector's ``average_psf`` file.

    Resolved from ``cal_token`` or the IRSA directory listing only.  SIA2 is
    deliberately skipped: a positional ``spherex_qr2_cal`` query does not
    return ``average_psf`` rows, and a position-less one scans the whole
    collection (~3 min).
    """
    from .cal_index import cal_http_url, cal_s3_uri, latest_cal_token_via_listing

    token = cal_token or latest_cal_token_via_listing("average_psf", data_release=data_release)
    if token is None:
        raise RuntimeError(
            f"could not list average_psf cal products for D{detector} on IRSA"
        )
    return (
        cal_http_url("average_psf", detector, token, data_release=data_release),
        cal_s3_uri("average_psf", detector, token, data_release=data_release),
    )


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


def cubes_identical(a: np.ndarray, b: np.ndarray) -> bool:
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
    """Per-detector cal cubes plus the sampling check against the L2 cube.

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
        self.cache_dir = cache_dir
        self.use_s3 = use_s3
        self.fsspec_kwargs = fsspec_kwargs
        self._loader = loader or self._load_from_irsa
        self._states: dict[int, _DetectorState] = {}
        self._states_lock = threading.Lock()

    def _load_from_irsa(self, detector: int) -> tuple[np.ndarray, str]:
        http_url, s3_uri = find_average_psf_product(
            detector, data_release=self.data_release, cal_token=self.cal_token
        )
        target = s3_uri if (self.use_s3 and s3_uri) else http_url
        cube = load_average_psf_cube(
            target, cache_dir=self.cache_dir, fsspec_kwargs=self.fsspec_kwargs
        )
        return cube, target

    def _state(self, detector: int) -> _DetectorState:
        with self._states_lock:
            return self._states.setdefault(detector, _DetectorState())

    def fetch(
        self,
        detector: int,
        *,
        fetch_full: Callable[[], _P],
        fetch_light: Callable[[np.ndarray], _P],
        cube_of: Callable[[_P], np.ndarray],
    ) -> tuple[_P, str]:
        """Run one cutout download; return ``(payload, psf_source_tag)``.

        ``fetch_full()`` downloads the cutout with its own PSF cube;
        ``fetch_light(cube)`` downloads it without and attaches ``cube``.
        """
        st = self._state(detector)
        with st.lock:
            if not st.disabled and not st.verified:
                # First cutout of this detector: others wait here so none of
                # them uses the cal cube before it has been checked once.
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
        return fetch_light(cube), f"cal:{source.rsplit('/', 1)[-1]}"

    def _check(self, st: _DetectorState, detector: int, l2_cube: np.ndarray, *, load: bool) -> None:
        if load:
            try:
                st.cube, st.source = self._loader(detector)
            except Exception as exc:
                st.disabled = True
                warnings.warn(
                    f"D{detector}: average_psf cal product unavailable ({exc}); "
                    "downloading the PSF cube with every cutout",
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
    cache_dir=None,
    use_s3: bool = False,
    fsspec_kwargs: dict | None = None,
) -> SharedPsfRegistry:
    """Process-wide registry for this cal selection (created on first use)."""
    key = (data_release, cal_token, use_s3)
    with _REGISTRIES_LOCK:
        reg = _REGISTRIES.get(key)
        if reg is None:
            reg = _REGISTRIES[key] = SharedPsfRegistry(
                verify_every=verify_every, cal_token=cal_token,
                data_release=data_release, cache_dir=cache_dir,
                use_s3=use_s3, fsspec_kwargs=fsspec_kwargs,
            )
        else:
            if verify_every < 0:
                raise ValueError("verify_every must be >= 0 (0 = check the first cutout only)")
            reg.verify_every = verify_every
            reg.cache_dir = cache_dir
            reg.fsspec_kwargs = fsspec_kwargs
        return reg
