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
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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


def http_download(url: str, dest: Path, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` atomically; return the final path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    with tempfile.NamedTemporaryFile(
        delete=False, dir=dest.parent, prefix=dest.name + ".", suffix=".part"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        shutil.move(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return dest


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
