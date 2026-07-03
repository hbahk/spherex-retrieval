"""Regression tests for the cutout -> detector pixel-origin mapping.

These lock in the sign convention that broke CWAVE/CBAND (and, more subtly,
SAPM and PSF-zone selection): the IRSA cutout header's alternate 'A' WCS has
its reference at detector pixel (1,1), so the 0-based detector origin of cutout
pixel (0,0) is ``-(CRPIX*A - 1)``, NOT the naive ``CRPIX*A - 1``.  With the old
sign an off-origin cutout (CRPIX2A strongly negative) produced a negative
origin, and ``cal.section[ylo:ylo+ny]`` negative-wrapped to the vertically
mirrored detector rows -> a within-detector wavelength reversal.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from spherex_retrieval.cutout import _payload_from_irsa_hdul
from spherex_retrieval.psf import cutout_to_orig
from spherex_retrieval.wavelength import crop_wavelength_maps


def _fake_irsa_hdul(crpix1a: int, crpix2a: int, ny: int = 8, nx: int = 6) -> fits.HDUList:
    """Minimal 6-extension MEF mimicking an IRSA cutout-service response."""
    img_hdr = fits.Header()
    # Minimal celestial WCS so WCS(header).celestial is well-defined.
    img_hdr["CTYPE1"] = "RA---TAN"
    img_hdr["CTYPE2"] = "DEC--TAN"
    img_hdr["CRVAL1"] = 150.0
    img_hdr["CRVAL2"] = 2.0
    img_hdr["CRPIX1"] = 1.0
    img_hdr["CRPIX2"] = 1.0
    img_hdr["CD1_1"] = -1.7e-4
    img_hdr["CD2_2"] = 1.7e-4
    img_hdr["DETECTOR"] = 1
    # Alternate 'A' detector-pixel WCS reference lands off the cutout.
    img_hdr["CRPIX1A"] = crpix1a
    img_hdr["CRPIX2A"] = crpix2a

    data = np.zeros((ny, nx), dtype=np.float32)
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data.copy(), img_hdr, name="IMAGE")
    flags = fits.ImageHDU(data.astype(np.int32), name="FLAGS")
    var = fits.ImageHDU(data.copy(), name="VARIANCE")
    zodi = fits.ImageHDU(data.copy(), name="ZODI")
    psf = fits.ImageHDU(np.zeros((1, 4, 4), dtype=np.float32), name="PSF")
    psf.header["OVERSAMP"] = 10
    return fits.HDUList([primary, image, flags, var, zodi, psf])


@pytest.mark.parametrize(
    "crpix1a, crpix2a",
    [(-1838, -1839), (1, 1), (-5, -5), (0, -100)],
)
def test_pixel_origin_is_negated_crpix(crpix1a, crpix2a):
    hdul = _fake_irsa_hdul(crpix1a, crpix2a)
    payload = _payload_from_irsa_hdul(hdul)
    assert payload.pixel_origin == (1 - crpix1a, 1 - crpix2a)


def test_pixel_origin_matches_cutout_to_orig():
    """pixel_origin must equal the detector coord of cutout pixel (0,0)."""
    crpix1a, crpix2a = -1838, -1839
    payload = _payload_from_irsa_hdul(_fake_irsa_hdul(crpix1a, crpix2a))
    x0, y0 = cutout_to_orig(0.0, 0.0, crpix1a=crpix1a, crpix2a=crpix2a)
    assert payload.pixel_origin == (x0, y0)


def test_off_origin_cutout_has_nonneg_origin():
    """The historical failure case: strongly negative CRPIX2A must not wrap."""
    payload = _payload_from_irsa_hdul(_fake_irsa_hdul(-1838, -1839))
    xlo, ylo = payload.pixel_origin
    assert xlo >= 0 and ylo >= 0
    assert (xlo, ylo) == (1839, 1840)


def test_crop_rejects_negative_origin(tmp_path):
    """The guard turns a silent negative-wrap mirror into a loud error."""
    cal = tmp_path / "cwave.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((2040, 2040), dtype=np.float32), name="CWAVE"),
            fits.ImageHDU(np.zeros((2040, 2040), dtype=np.float32), name="CBAND"),
        ]
    ).writeto(cal)
    with pytest.raises(ValueError, match="non-negative"):
        crop_wavelength_maps(str(cal), pixel_origin=(-1840, -1840), cutout_shape=(8, 6))
