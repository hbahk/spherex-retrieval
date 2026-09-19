"""End-to-end retrieval orchestrator."""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Row

from .bundle import (
    Bundle,
    RetrievalStatus,
    cutout_filename,
    write_bundle,
    write_summary,
)
from .cutout import CutoutBackend, fetch_cutout
from .flux_correction import (
    apply_flux_correction,
    crop_flux_correction,
    find_flux_correction_product,
)
from .psf import (
    ZONE_MARGIN_DEFAULT,
    fix_psf_header_if_needed,
    subset_zones_for_cutout,
    zone_table_from_epsf,
)
from .psf_shared import (
    EPSF_RELEASE_DEFAULT,
    PSF_VERIFY_EVERY_DEFAULT,
    PsfSource,
    SharedPsfRegistry,
    get_registry,
    psf_kind_of_release,
)
from .query import (
    SUPPORTED_COLLECTIONS,
    find_overlapping,
    release_of_collection,
    release_of_url,
)
from .sapm import crop_sapm, find_sapm_product
from .wavelength import crop_wavelength_maps, find_cal_product

QueryBackend = Literal["astroquery", "pyvo"]

#: Release whose detector calibrations (spectral WCS, SAPM, flux corrections)
#: are applied to every image by default: the newest on-sky calibration. Bump
#: when DR1 publishes newer products.
CALIBRATION_RELEASE_DEFAULT = "qr3"


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
    psf_source: PsfSource = "epsf-cal",
    psf_verify_every: int = PSF_VERIFY_EVERY_DEFAULT,
    psf_cal_token: str | None = None,
    epsf_release: str = EPSF_RELEASE_DEFAULT,
    calibration_release: str | None = CALIBRATION_RELEASE_DEFAULT,
    gain_correction: bool = True,
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
    psf_source : {"epsf-cal", "cal", "l2"}
        ``"epsf-cal"`` (default) gives every image the R7 effective PSF: an
        R7 image (QR3, DR1) shares its own per-detector ``epsf`` library,
        verified against the L2 file exactly as ``"cal"`` does; a QR2 image
        gets the R7 library of ``epsf_release`` attached in place of its
        optical cube, which is never downloaded.  Measured on A2537 QR2
        images, the R7 ePSF fits stars better than the QR2 cube even after
        the core re-registration (chi2/dof 1.6 vs 2.9, central residual
        1 % vs 4 %) and needs no re-registration at all; no check against
        the L2 file is possible for that attachment.  ``"cal"`` takes the
        PSF product of the image's OWN release from its per-detector cal
        file (the QR2 121-plane optical cube, ``average_psf``, or the R7
        ``EPSF`` table, ``epsf``), fetched once, and stops each cutout
        download right after the PSF header — the product is identical in
        every L2 file of a detector and most of a small cutout's bytes; the
        PSF *header* (zone table, provenance) still comes from the L2 file.
        This is the mode that reproduces the paper's QR2 configuration of
        record.  ``"l2"`` downloads the product with every cutout.  The
        output primary header records the choice in ``PSFSRC`` and the
        product kind in ``PSFKIND``.
    psf_verify_every : int
        Whenever the shared product is the image's own (``"cal"``, and
        ``"epsf-cal"`` on R7 images), the first cutout of each detector and
        every N-th one after it is downloaded in full and its product
        compared with the cal file; a mismatch switches that detector back
        to full downloads with a warning.  ``0`` checks the first cutout
        only.  R7 cutouts are additionally checked on every download by the
        ``EPSF`` header's calibration source file.
    psf_cal_token : str, optional
        Pin the PSF cal version, e.g. ``'cal-psf-v5-2026-082'`` or
        ``'cal-epsf-v1-2026-191'`` (applies to every release retrieved).
    epsf_release : str
        Release whose ``epsf`` library ``psf_source="epsf-cal"`` attaches to
        images of a release without one (default ``"qr3"``).
    calibration_release : str or None
        Release whose per-detector calibrations are used for EVERY image:
        the spectral WCS (``CWAVE``/``CBAND``) and the solid-angle map.
        Default ``"qr3"``, the R7 on-sky calibration, also on QR2 images:
        the maps are detector properties in the L2 pixel frame
        (``DETCOORD='sky'`` in both releases; D1-D3 unchanged between them,
        bands 5 and 6 shifted by a constant, +0.0064 um on D5). ``None``
        takes each image's own release, the paper's configuration of record.
    gain_correction : bool
        Apply the R7 ``l3_flux_corrections`` of ``calibration_release`` to
        images of an earlier release (QR2): IMAGE times the per-pixel
        factor, VARIANCE times its square, so QR2 and R7 fluxes share the R7
        absolute gain (D1 median factor 0.970, D5 1.007). R7 images are
        never touched. Recorded in the bundle as ``FLXCORR`` (cal token),
        ``FLXCMED`` (median factor over the cutout) and ``FLXCSRC``.
        ``False`` keeps the QR2 gains (the paper's configuration).
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
        if psf_source == "epsf-cal" and psf_kind_of_release(release) == "optical":
            # a QR2 image: attach the R7 library (no check against the file possible)
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
            calibration_release=calibration_release or release,
            gain_correction=gain_correction,
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
    calibration_release: str | None = None,
    gain_correction: bool = False,
    cache_dir: Path | None,
    fsspec_kwargs: dict | None,
    query_backend: QueryBackend,
) -> Bundle:
    calibration_release = calibration_release or data_release
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

    if (gain_correction and bundle.cutout is not None
            and psf_kind_of_release(data_release) == "optical"
            and psf_kind_of_release(calibration_release) == "effective"):
        # a pre-R7 image: bring its pixels to the R7 absolute gain
        try:
            http, s3, token = find_flux_correction_product(
                bundle.detector, data_release=calibration_release)
            target = s3 if (cutout_backend == "fsspec" and s3) else http
            corr = crop_flux_correction(
                target, pixel_origin=bundle.cutout.pixel_origin,
                cutout_shape=bundle.cutout.image.shape, token=token,
                cache_dir=cache_dir, fsspec_kwargs=fsspec_kwargs)
            apply_flux_correction(bundle.cutout, corr)
            bundle.extras["flux_correction"] = {
                "token": token, "source_file": corr.source_file,
                "median": float(np.median(corr.data)), "n_replaced": corr.n_replaced}
        except Exception as exc:  # noqa: BLE001 - a cal-product hiccup must not lose the cutout
            note = f"flux correction failed: {exc}"
            bundle.message = (bundle.message + "; " + note) if bundle.message else note

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
                data_release=calibration_release,
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
                data_release=calibration_release,
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
