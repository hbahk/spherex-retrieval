"""End-to-end retrieval orchestrator."""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Literal

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Row

from .bundle import Bundle, RetrievalStatus, cutout_filename, write_bundle, write_summary
from .cutout import CutoutBackend, fetch_cutout
from .psf import (ZONE_MARGIN_DEFAULT, fix_psf_header_if_needed,
                  subset_zones_for_cutout, zone_table_from_epsf)
from .psf_shared import (EPSF_RELEASE_DEFAULT, PSF_VERIFY_EVERY_DEFAULT, PsfSource,
                         SharedPsfRegistry, get_registry)
from .query import (SUPPORTED_COLLECTIONS, find_overlapping, release_of_collection,
                    release_of_url)
from .sapm import crop_sapm, find_sapm_product
from .wavelength import crop_wavelength_maps, find_cal_product

QueryBackend = Literal["astroquery", "pyvo"]


def default_output_dir(coord: SkyCoord) -> Path:
    """Path under cwd, namespaced by coordinate."""
    ra = coord.icrs.ra.to_value(u.deg)
    dec = coord.icrs.dec.to_value(u.deg)
    return Path.cwd() / f"spherex_cutouts_{ra:.5f}_{dec:+.5f}"


def retrieve(
    coord: SkyCoord,
    size: u.Quantity,
    *,
    output_dir: Path | str | None = None,
    query_backend: QueryBackend = "astroquery",
    cutout_backend: CutoutBackend = "irsa",
    collections: tuple[str, ...] = SUPPORTED_COLLECTIONS,
    bandpass: str | None = None,
    include_wavelength: bool = True,
    include_sapm: bool = False,
    sapm_cal_token: str | None = None,
    subset_psf: bool = True,
    zone_margin: int = ZONE_MARGIN_DEFAULT,
    psf_source: PsfSource = "cal",
    psf_verify_every: int = PSF_VERIFY_EVERY_DEFAULT,
    psf_cal_token: str | None = None,
    epsf_release: str = EPSF_RELEASE_DEFAULT,
    max_workers: int = 8,
    cache_dir: Path | str | None = None,
    fsspec_kwargs: dict | None = None,
    remote_timeout: float = 120.0,
) -> tuple[list[Bundle], Path]:
    """Retrieve all SPHEREx cutouts overlapping ``coord`` within ``size``.

    Parameters
    ----------
    bandpass : str, optional
        Restrict to one SPHEREx detector (e.g. ``'SPHEREx-D2'``).  Filter
        is applied at query time by both backends.
    include_sapm : bool
        When True, also fetch the matching Solid Angle Pixel Map (SAPM)
        cal product per detector and store the cropped (arcsec^2) array
        as a ``SAPM`` HDU.  Useful for converting MJy/sr to flux density
        (uJy) before forced photometry.
    sapm_cal_token : str, optional
        Pin the SAPM cal version, e.g. ``'cal-sapm-v2-2025-164'``.  When
        omitted, the latest SAPM available via SIA2 for each detector is
        used.
    psf_source : {"cal", "l2", "epsf-cal"}
        ``"cal"`` (default) takes the PSF product from the per-detector cal
        file of the image's own release, fetched once, and stops each cutout
        download right after the PSF header — the product is identical in
        every L2 file of a detector and most of a small cutout's bytes (the
        QR2 121-plane optical cube, ``average_psf``, 4.9 MB; the R7 ``EPSF``
        table, ``epsf``, 3.9 MB).  The PSF *header* (zone table, provenance)
        still comes from the L2 file.  ``"l2"`` downloads the product with
        every cutout, as before.  ``"epsf-cal"`` attaches the R7 effective
        PSF library of ``epsf_release`` to EVERY image, QR2 ones included
        (the diagnostic that separates a PSF-product error from a WCS
        offset, and a way to use the ePSF on QR2 images before DR1); no
        check against the L2 file is possible then.  The output primary
        header records the choice in ``PSFSRC`` and the product kind in
        ``PSFKIND``.
    psf_verify_every : int
        With ``psf_source="cal"``, the first cutout of each detector and
        every N-th one after it is downloaded in full and its product
        compared with the cal file; a mismatch switches that detector back
        to full downloads with a warning.  ``0`` checks the first cutout
        only.  R7 cutouts are additionally checked on every download by the
        ``EPSF`` header's calibration source file.
    psf_cal_token : str, optional
        Pin the PSF cal version, e.g. ``'cal-psf-v5-2026-082'`` or
        ``'cal-epsf-v1-2026-191'`` (applies to every release retrieved).
    epsf_release : str
        Release whose ``epsf`` library ``psf_source="epsf-cal"`` attaches
        (default ``"qr3"``).
    remote_timeout : float
        Sets ``astropy.utils.data.conf.remote_timeout``; SPHEREx reads
        often exceed the default, hence 120 s is the recommended floor
        in the IRSA tutorials.

    Returns
    -------
    bundles : list[Bundle]
    output_dir : Path
        Directory containing per-cutout MEFs and ``summary.ecsv``.
    """
    # Tutorials warn that SPHEREx remote reads can exceed astropy's default.
    from astropy.utils.data import conf as _astropy_data_conf
    _astropy_data_conf.remote_timeout = max(remote_timeout, _astropy_data_conf.remote_timeout)

    if output_dir is None:
        output_dir = default_output_dir(coord)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cache_dir) if cache_dir else None

    if psf_source not in ("cal", "l2", "epsf-cal"):
        raise ValueError(f"unknown psf_source: {psf_source!r}")

    def _registry_for(release: str) -> SharedPsfRegistry | None:
        """The shared-PSF registry for an image of ``release`` (None: l2)."""
        if psf_source == "l2":
            return None
        if psf_source == "epsf-cal":
            return get_registry(
                verify_every=psf_verify_every, cal_token=psf_cal_token,
                data_release=epsf_release, kind="effective", verify=False,
                cache_dir=cache_dir, use_s3=(cutout_backend == "fsspec"),
                fsspec_kwargs=fsspec_kwargs,
            )
        return get_registry(
            verify_every=psf_verify_every, cal_token=psf_cal_token,
            data_release=release, cache_dir=cache_dir,
            use_s3=(cutout_backend == "fsspec"), fsspec_kwargs=fsspec_kwargs,
        )

    overlap = find_overlapping(
        coord, size,
        backend=query_backend,
        collections=tuple(collections),
        bandpass=bandpass,
    )

    if len(overlap) == 0:
        write_summary([], output_dir / "summary.ecsv")
        return [], output_dir

    def _do_one(row: Row) -> Bundle:
        release = _release_of_row(row)
        return _retrieve_one(
            row=row,
            coord=coord,
            size=size,
            cutout_backend=cutout_backend,
            include_wavelength=include_wavelength,
            include_sapm=include_sapm,
            sapm_cal_token=sapm_cal_token,
            subset_psf=subset_psf,
            zone_margin=zone_margin,
            psf_registry=_registry_for(release),
            data_release=release,
            cache_dir=cache_dir,
            fsspec_kwargs=fsspec_kwargs,
            query_backend=query_backend,
        )

    bundles: list[Bundle] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, row) for row in overlap]
        for fut in cf.as_completed(futures):
            bundles.append(fut.result())

    bundles.sort(key=lambda b: (b.time_bounds_lower, b.detector))

    for i, b in enumerate(bundles, start=1):
        if b.is_ok and b.cutout is not None:
            try:
                write_bundle(b, output_dir / cutout_filename(b, i))
            except Exception as exc:  # pragma: no cover
                b.status = RetrievalStatus.DOWNLOAD_FAILED
                b.message = f"write failed: {exc}"

    write_summary(bundles, output_dir / "summary.ecsv")
    return bundles, output_dir


def _release_of_row(row) -> str:
    """Data release of a discovery row: from its collection name, else its URL."""
    try:
        return release_of_collection(str(row["collection"]))
    except (ValueError, KeyError):
        return release_of_url(str(row["access_url"])) or "qr2"


def _retrieve_one(
    *,
    row: Row,
    coord: SkyCoord,
    size: u.Quantity,
    cutout_backend: CutoutBackend,
    include_wavelength: bool,
    include_sapm: bool,
    sapm_cal_token: str | None,
    subset_psf: bool,
    zone_margin: int = ZONE_MARGIN_DEFAULT,
    psf_registry: SharedPsfRegistry | None = None,
    data_release: str = "qr2",
    cache_dir: Path | None,
    fsspec_kwargs: dict | None,
    query_backend: QueryBackend,
) -> Bundle:
    bundle = Bundle(
        obs_id=str(row["obs_id"]),
        detector=int(row["detector"]),
        collection=str(row["collection"]),
        access_url=str(row["access_url"]),
        cloud_uri=str(row["cloud_uri"]),
        time_bounds_lower=float(row["time_bounds_lower"]),
        coord_ra=coord.icrs.ra.to_value(u.deg),
        coord_dec=coord.icrs.dec.to_value(u.deg),
    )

    try:
        bundle.cutout = fetch_cutout(
            access_url=bundle.access_url,
            cloud_uri=bundle.cloud_uri,
            coord=coord,
            size=size,
            backend=cutout_backend,
            cache_dir=cache_dir,
            fsspec_kwargs=fsspec_kwargs,
            psf_registry=psf_registry,
            detector=bundle.detector,
        )
    except Exception as exc:
        bundle.status = RetrievalStatus.DOWNLOAD_FAILED
        bundle.message = f"cutout failed: {exc}"
        return bundle

    if bundle.cutout is not None and bundle.cutout.psf_kind == "optical":
        # Apply the SPHEREx PSF header erratum fix in-place if the file
        # is from VERSION <= 6.5.5 without "+psffix1".  Without this the
        # XCTR_i / YCTR_i mapping is wrong and zone selection is wrong.
        # (R7 files carry the lattice in the EPSF table; nothing to fix.)
        try:
            fixed_hdr, was_fixed = fix_psf_header_if_needed(
                bundle.cutout.psf_header,
                bundle.cutout.primary_header,
                n_planes=bundle.cutout.psf_cube.shape[0],
            )
            if was_fixed:
                bundle.cutout.psf_header = fixed_hdr
                bundle.extras["psf_header_fixed"] = True
        except Exception as exc:
            bundle.message = f"psf header fix skipped: {exc}"

    if subset_psf and bundle.cutout is not None:
        try:
            zone_table = (zone_table_from_epsf(bundle.cutout.psf_table)
                          if bundle.cutout.psf_table is not None else None)
            bundle.psf_subset = subset_zones_for_cutout(
                bundle.cutout.psf_cube,
                bundle.cutout.psf_header,
                cutout_shape=bundle.cutout.image.shape,
                pixel_origin=bundle.cutout.pixel_origin,
                zone_margin=zone_margin,
                zone_table=zone_table,
            )
        except Exception as exc:
            bundle.message = f"psf subset failed: {exc}"

    if include_wavelength and bundle.cutout is not None:
        try:
            cal_http, cal_s3 = find_cal_product(
                bundle.detector, backend=query_backend, coord=coord,
                data_release=data_release,
            )
            cal_target = cal_s3 if (cutout_backend == "fsspec" and cal_s3) else cal_http
            bundle.wavelength = crop_wavelength_maps(
                cal_target,
                pixel_origin=bundle.cutout.pixel_origin,
                cutout_shape=bundle.cutout.image.shape,
                cache_dir=cache_dir,
                fsspec_kwargs=fsspec_kwargs,
            )
        except Exception as exc:
            bundle.status = RetrievalStatus.WAVELENGTH_MISSING
            bundle.message = f"wavelength fetch failed: {exc}"

    if include_sapm and bundle.cutout is not None:
        try:
            sapm_http, sapm_s3 = find_sapm_product(
                bundle.detector,
                backend=query_backend,
                cal_token=sapm_cal_token,
                coord=coord,
                data_release=data_release,
            )
            sapm_target = (
                sapm_s3 if (cutout_backend == "fsspec" and sapm_s3) else sapm_http
            )
            bundle.sapm = crop_sapm(
                sapm_target,
                pixel_origin=bundle.cutout.pixel_origin,
                cutout_shape=bundle.cutout.image.shape,
                cache_dir=cache_dir,
                fsspec_kwargs=fsspec_kwargs,
            )
        except Exception as exc:
            # SAPM failure is non-fatal — note it in the message but
            # don't override an existing status.
            note = f"sapm fetch failed: {exc}"
            bundle.message = (bundle.message + "; " + note) if bundle.message else note

    return bundle
