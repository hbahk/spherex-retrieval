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

    cut = bundle.cutout
    image_hdu = fits.ImageHDU(cut.image, header=cut.image_header, name="IMAGE")
    flags_hdu = fits.ImageHDU(cut.flags, name="FLAGS")
    var_hdu = fits.ImageHDU(cut.variance, name="VARIANCE")
    zodi_hdu = fits.ImageHDU(cut.zodi, name="ZODI")

    psf_hdu = fits.ImageHDU(
        bundle.psf_subset.cube if bundle.psf_subset else cut.psf_cube,
        header=cut.psf_header,
        name="PSF",
    )
    if bundle.psf_subset is not None:
        psf_zones_hdu = fits.BinTableHDU(bundle.psf_subset.lookup, name="PSF_ZONES")
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
