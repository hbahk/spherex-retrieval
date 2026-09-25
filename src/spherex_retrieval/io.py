"""Remote-FITS open helpers with URL-keyed local cache.

Two access modes are supported:

* ``http``  — vanilla HTTP/HTTPS download; the file is fetched once and cached
  locally on disk by URL hash.
* ``s3``    — fsspec / s3fs streaming with byte-range reads.  Astropy opens
  the file lazily (`use_fsspec=True`).

The default cache lives under ``$SPHEREX_RETRIEVAL_CACHE`` (or
``~/.cache/spherex-retrieval``) but callers can override per-call.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import urllib.parse
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import numpy as np
import requests
from astropy.io import fits


def default_cache_dir() -> Path:
    env = os.environ.get("SPHEREX_RETRIEVAL_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "spherex-retrieval"
    base.mkdir(parents=True, exist_ok=True)
    return base


def url_to_cache_path(url: str, cache_dir: Path | None = None) -> Path:
    """Deterministic on-disk path for a given URL."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".fits"
    return cache_dir / f"{digest}{suffix}"


def is_s3_uri(target: str) -> bool:
    return target.startswith("s3://") or target.startswith("gs://")


_RETRYABLE_STATUS = {500, 502, 503, 504}
_T = TypeVar("_T")


def http_download(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    max_retries: int = 4,
    backoff: float = 2.0,
) -> Path:
    """Download ``url`` to ``dest`` atomically; return the final path.

    Retries on transient network failures and 5xx server errors with
    exponential backoff (``backoff * 2**attempt`` seconds, jittered by
    ±20 %).  IRSA's ibe service occasionally returns 503 under load; this
    loop hides those from callers.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    def _attempt() -> Path:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=dest.parent, prefix=dest.name + ".", suffix=".part"
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                _raise_for_status(resp, url)
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            shutil.move(str(tmp_path), str(dest))
            return dest
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return _with_retries(_attempt, url=url, max_retries=max_retries, backoff=backoff)


def http_fetch_until(
    url: str,
    stop: Callable[[bytearray], int | None],
    *,
    timeout: float = 120.0,
    max_retries: int = 4,
    backoff: float = 2.0,
    chunk_size: int = 1 << 14,
) -> tuple[bytes, bool]:
    """Stream ``url`` into memory and hang up as soon as ``stop`` is satisfied.

    ``stop(buffer)`` is called after every chunk (with the live, growing
    ``bytearray`` — it must not keep a view of it); once it returns a byte
    count ``n`` the connection is closed and ``(buffer[:n], True)`` is
    returned.  If the stream ends first the whole body comes back as
    ``(body, False)``.  IRSA's cutout service ignores ``Range`` (it answers
    200 with the full chunked body), so closing the stream early is the only
    way not to receive the trailing HDUs.  Nothing is written to the disk
    cache.  Retries follow :func:`http_download`.
    """
    def _attempt() -> tuple[bytes, bool]:
        buf = bytearray()
        with requests.get(url, stream=True, timeout=timeout) as resp:
            _raise_for_status(resp, url)
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                buf += chunk
                n = stop(buf)
                if n is not None:
                    return bytes(buf[:n]), True
        return bytes(buf), False

    return _with_retries(_attempt, url=url, max_retries=max_retries, backoff=backoff)


def _raise_for_status(resp: requests.Response, url: str) -> None:
    if resp.status_code in _RETRYABLE_STATUS:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for {url}", response=resp
        )
    resp.raise_for_status()


def _with_retries(attempt: Callable[[], _T], *, url: str, max_retries: int, backoff: float) -> _T:
    """Run ``attempt`` with exponential backoff on transient HTTP failures."""
    import random
    import time

    last_exc: Exception | None = None
    for i in range(max_retries):
        try:
            return attempt()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # Don't retry on 4xx (except 408 Request Timeout, 429 Too Many Requests).
            if status is not None and status not in _RETRYABLE_STATUS and status not in (408, 429):
                raise
            if i == max_retries - 1:
                raise
            time.sleep(backoff * (2 ** i) * random.uniform(0.8, 1.2))
    raise RuntimeError(f"http request exhausted retries for {url}") from last_exc


@contextmanager
def open_fits(
    target: str,
    *,
    mode: str = "auto",
    cache_dir: Path | None = None,
    fsspec_kwargs: dict | None = None,
) -> Iterator[fits.HDUList]:
    """Open a FITS file from HTTP, S3, or a local path.

    Parameters
    ----------
    target : str
        Local path, ``http(s)://...`` URL, or ``s3://...`` URI.
    mode : {"auto", "http", "s3", "local"}
        Force a specific access mode; ``auto`` picks based on the prefix.
    cache_dir : Path, optional
        Where to keep HTTP downloads.  Ignored for S3/local.
    fsspec_kwargs : dict, optional
        Forwarded to ``fits.open`` when streaming via fsspec; defaults to
        ``{"anon": True}`` for public S3 buckets.
    """
    if mode == "auto":
        if is_s3_uri(target):
            mode = "s3"
        elif target.startswith(("http://", "https://")):
            mode = "http"
        else:
            mode = "local"

    if mode == "s3":
        kw = {"anon": True}
        if fsspec_kwargs:
            kw.update(fsspec_kwargs)
        with fits.open(target, use_fsspec=True, fsspec_kwargs=kw) as hdul:
            yield hdul
    elif mode == "http":
        local = url_to_cache_path(target, cache_dir=cache_dir)
        http_download(target, local)
        with fits.open(local) as hdul:
            yield hdul
    else:
        with fits.open(target) as hdul:
            yield hdul


# --------------------------------------------------------------------------- #
# Whole calibration planes held in memory
# --------------------------------------------------------------------------- #

#: Full HDUs kept per process. A detector's CWAVE, CBAND, SAPM or flux-correction
#: plane is 16.6 MB; 64 holds every family for six detectors in two releases.
HDU_CACHE_MAX = 64
_HDU_CACHE: OrderedDict = OrderedDict()
_HDU_CACHE_LOCK = threading.Lock()


def cached_hdu_data(target: str, names: tuple[str, ...], fallback_index: int, *,
                    cache_dir: Path | None = None,
                    fsspec_kwargs: dict | None = None) -> tuple[np.ndarray, fits.Header]:
    """The full data and header of one HDU of a calibration file, read once per process.

    The first EXTNAME in ``names`` present in the file is used, else HDU
    ``fallback_index``. The array is shared and read-only; callers slice it and
    copy what they keep. A cutout's crop from it holds the same values as a
    ``.section`` read of the file, without reopening the file for every cutout.
    """
    key = (str(target), tuple(names), int(fallback_index))
    with _HDU_CACHE_LOCK:
        hit = _HDU_CACHE.get(key)
        if hit is not None:
            _HDU_CACHE.move_to_end(key)
            return hit
    with open_fits(target, mode="auto", cache_dir=cache_dir, fsspec_kwargs=fsspec_kwargs) as hdul:
        hdu = None
        for name in names:
            if name in hdul:
                hdu = hdul[name]
                break
        if hdu is None:
            hdu = hdul[fallback_index]
        data = np.array(hdu.data, copy=True)
        header = hdu.header.copy()
    data.flags.writeable = False
    with _HDU_CACHE_LOCK:
        _HDU_CACHE[key] = (data, header)
        _HDU_CACHE.move_to_end(key)
        while len(_HDU_CACHE) > HDU_CACHE_MAX:
            _HDU_CACHE.popitem(last=False)
    return data, header


def clear_hdu_cache() -> None:
    """Drop the in-memory calibration planes."""
    with _HDU_CACHE_LOCK:
        _HDU_CACHE.clear()
