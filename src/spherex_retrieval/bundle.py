"""Per-cutout retrieval bundle and on-disk MEF writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .cutout import CutoutPayload
from .psf import PSFZoneSubset
from .sapm import SolidAnglePixelMap
from .wavelength import WavelengthMaps


class RetrievalStatus(str, Enum):
    OK = "ok"
    OUT_OF_BOUNDS = "out_of_bounds"
    DOWNLOAD_FAILED = "download_failed"
    QA_EXCLUDED = "qa_excluded"
    WAVELENGTH_MISSING = "wavelength_missing"


@dataclass
class Bundle:
    """A single overlapping pointing's contribution to the retrieval."""

    obs_id: str
    detector: int
    collection: str
    access_url: str
    cloud_uri: str
    time_bounds_lower: float
    coord_ra: float
    coord_dec: float
    cutout: Optional[CutoutPayload] = None
    psf_subset: Optional[PSFZoneSubset] = None
    wavelength: Optional[WavelengthMaps] = None
    sapm: Optional[SolidAnglePixelMap] = None
    status: RetrievalStatus = RetrievalStatus.OK
    message: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status == RetrievalStatus.OK


# --------------------------------------------------------------------------- #
# On-disk layout
# --------------------------------------------------------------------------- #

def cutout_filename(bundle: Bundle, cutout_index: int) -> str:
    safe_obs = bundle.obs_id.replace("/", "_") if bundle.obs_id else "unknown"
    return f"cutout_{cutout_index:04d}_{safe_obs}_D{bundle.detector}.fits"


def write_bundle(bundle: Bundle, path: Path) -> Path:
    """Write a per-cutout MEF.

    Layout (HDUs after PSF_ZONES are present only when the corresponding
    side data was retrieved):

        HDU 0  PRIMARY    (header carries provenance)
        HDU 1  IMAGE
        HDU 2  FLAGS
        HDU 3  VARIANCE
        HDU 4  ZODI
        HDU 5  PSF        (subsetted cube)
        HDU 6  PSF_ZONES  (lookup table for the PSF subset)
        HDU ?  CWAVE      (per-pixel central wavelength, microns)
        HDU ?  CBAND      (per-pixel bandwidth, microns)
        HDU ?  SAPM       (solid-angle per pixel, arcsec^2)

    ``FLXCORR`` / ``FLXCMED`` / ``FLXCSRC`` / ``FLXCNREP`` are present when the
    R7 flux correction was applied to a pre-R7 image (``gain_correction``):
    IMAGE and VARIANCE then carry the R7 absolute gain, not the QR2 one.

    The layout is the same for both PSF kinds; the PRIMARY header says which:

        PSFKIND  'OPTICAL' (QR2 cube, 10x, the pixel response NOT included)
                 or 'EPSF' (R7 effective PSF, 5x, pixel response included:
                 render by point sampling, never integrate again)
        OVERSAMP oversampling of the PSF planes (10 or 5)
        PSFNORM  'hr-sum-1': each plane sums to 1 on its own oversampled grid
        EPSFCAL  the ePSF calibration source file (R7 only)
        DETCOORD 'sky' (R7 only): the arrays and zone centres are in the L2
                 image orientation for every detector, no mirroring needed
        ZONENX / ZONENY  lattice size of the full product (11x11, 21x21, 11x41)

    ``PSF_ZONES`` carries ``zone_id, x, y, plane_idx`` (0-based detector px
    centres) and, for the R7 product, ``xwidth, ywidth, nstar, neff``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundle.cutout is None:
        raise ValueError("cannot write an empty bundle")

    primary = fits.PrimaryHDU()
    h = primary.header
    h["OBSID"] = (bundle.obs_id, "SPHEREx Observation ID")
    h["DETECTOR"] = (bundle.detector, "Detector index 1..6")
    h["COLLECT"] = (bundle.collection, "IRSA collection")
    h["ACCESS"] = (bundle.access_url[:68], "L2 MEF access URL (truncated)")
    h["CLOUDURI"] = (bundle.cloud_uri[:68], "Cloud URI (truncated)")
    h["TMINMJD"] = (bundle.time_bounds_lower, "Lower time bound, MJD")
    h["RA_REQ"] = (bundle.coord_ra, "Requested RA (deg)")
    h["DEC_REQ"] = (bundle.coord_dec, "Requested Dec (deg)")
    h["STATUS"] = (bundle.status.value, "Retrieval status")
    if bundle.cutout is not None:
        for key in ("VERSION", "OBSDATE", "DATE", "PROCDATE"):
            if key in bundle.cutout.primary_header:
                h[key] = bundle.cutout.primary_header[key]
        h["PSFFIXED"] = (
            bool(bundle.extras.get("psf_header_fixed", False)),
            "True if PSF erratum fix was applied locally",
        )
        h["OVERSAMP"] = (bundle.cutout.psf_oversamp, "PSF oversampling factor")
        h["PSFSRC"] = (bundle.cutout.psf_source[:68], "PSF cube origin")
        fc = bundle.extras.get("flux_correction")
        if fc:
            h["FLXCORR"] = (str(fc["token"])[:68], "R7 flux correction applied to IMAGE/VARIANCE")
            h["FLXCMED"] = (float(fc["median"]), "median correction factor over the cutout")
            if fc.get("source_file"):
                h["FLXCSRC"] = (str(fc["source_file"])[:68], "flux-correction calibration source file")
            h["FLXCNREP"] = (int(fc.get("n_replaced", 0)), "wild factors replaced by their row median")
        effective = bundle.cutout.psf_kind == "effective"
        h["PSFKIND"] = ("EPSF" if effective else "OPTICAL",
                        "EPSF: pixel response included, point-sample it")
        h["PSFNORM"] = ("hr-sum-1", "each PSF plane sums to 1 on its oversampled grid")
        if effective:
            from .psf_shared import epsf_source_file
            src = epsf_source_file(bundle.cutout.psf_header)
            if src:
                h["EPSFCAL"] = (src[:68], "ePSF calibration source file")
            if "DETCOORD" in bundle.cutout.psf_header:
                h["DETCOORD"] = (str(bundle.cutout.psf_header["DETCOORD"]),
                                 "ePSF coordinate frame (matches the L2 image)")
        if bundle.cutout.psf_table is not None:
            from .psf import zone_lattice, zone_table_from_epsf
            ix, iy = zone_lattice(zone_table_from_epsf(bundle.cutout.psf_table))
            h["ZONENX"] = (int(ix.max()), "PSF zone lattice size along x")
            h["ZONENY"] = (int(iy.max()), "PSF zone lattice size along y")

    cut = bundle.cutout
    image_hdu = fits.ImageHDU(cut.image, header=cut.image_header, name="IMAGE")
    flags_hdu = fits.ImageHDU(cut.flags, name="FLAGS")
    var_hdu = fits.ImageHDU(cut.variance, name="VARIANCE")
    zodi_hdu = fits.ImageHDU(cut.zodi, name="ZODI")

    psf_hdu = fits.ImageHDU(
        bundle.psf_subset.cube if bundle.psf_subset else cut.psf_cube,
        header=_psf_image_header(cut.psf_header, cut.psf_kind),
        name="PSF",
    )
    if bundle.psf_subset is not None:
        psf_zones_hdu = fits.BinTableHDU(bundle.psf_subset.lookup, name="PSF_ZONES")
    elif cut.psf_table is not None:
        from .psf import zone_table_from_epsf
        full = zone_table_from_epsf(cut.psf_table)
        full["plane_idx"] = np.arange(len(full), dtype=np.int32)
        psf_zones_hdu = fits.BinTableHDU(full, name="PSF_ZONES")
    else:
        psf_zones_hdu = fits.BinTableHDU(Table(names=("zone_id", "x", "y", "plane_idx")),
                                         name="PSF_ZONES")

    hdus = [primary, image_hdu, flags_hdu, var_hdu, zodi_hdu, psf_hdu, psf_zones_hdu]

    if bundle.wavelength is not None:
        hdus.append(fits.ImageHDU(bundle.wavelength.cwave, name="CWAVE"))
        hdus.append(fits.ImageHDU(bundle.wavelength.cband, name="CBAND"))

    if bundle.sapm is not None:
        sapm_hdu = fits.ImageHDU(bundle.sapm.data, name="SAPM")
        sapm_hdu.header["BUNIT"] = bundle.sapm.bunit
        hdus.append(sapm_hdu)

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


_EPSF_HEADER_KEYS = ("DETECTOR", "OVSMPX", "OVSMPY", "MAXIT", "SMOOTH", "JUNKCLN", "KEEPNAT",
                     "KEEPOS", "DETCOORD", "ORDERING", "BINSRC", "GEOMSRC", "NEFFSRC", "WVMSRC",
                     "CWAVESRC", "CBANDSRC")


def _psf_image_header(psf_header: fits.Header, psf_kind: str) -> fits.Header:
    """The header of the bundle's ``PSF`` image HDU.

    The QR2 ``PSF`` header is an image header and is kept as is (the zone
    keywords live there). The R7 ``EPSF`` header describes a binary table,
    so only its descriptive cards and HISTORY are carried over."""
    if psf_kind != "effective":
        return psf_header
    out = fits.Header()
    for key in _EPSF_HEADER_KEYS:
        if key in psf_header:
            out[key] = (psf_header[key], psf_header.comments[key])
    out["OVERSAMP"] = (int(psf_header.get("OVSMPX", 5)), "PSF oversampling factor")
    for card in psf_header.get("HISTORY", []):
        out["HISTORY"] = str(card)
    return out


def write_summary(bundles: list[Bundle], path: Path) -> Path:
    """Write a summary table with one row per overlapping pointing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, b in enumerate(bundles, start=1):
        ny = nx = 0
        xlo = ylo = -1
        if b.cutout is not None:
            ny, nx = b.cutout.image.shape
            xlo, ylo = b.cutout.pixel_origin
        rows.append(
            {
                "cutout_index": i,
                "obs_id": b.obs_id,
                "detector": b.detector,
                "collection": b.collection,
                "time_bounds_lower": b.time_bounds_lower,
                "ra_req": b.coord_ra,
                "dec_req": b.coord_dec,
                "nx": nx,
                "ny": ny,
                "x_orig_lo": xlo,
                "y_orig_lo": ylo,
                "status": b.status.value,
                "message": b.message,
                "access_url": b.access_url,
                "cloud_uri": b.cloud_uri,
            }
        )
    Table(rows=rows).write(path, overwrite=True)
    return path
