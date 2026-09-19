"""R7 flux corrections on QR2 images, and the calibration-release policy.

The SSDC's ``l3_flux_corrections`` map brings QR2 pixels to the R7 absolute
gain. These tests pin down that the map is cropped like the other detector
products, that its dead/hot-pixel values are replaced by the row median, that
IMAGE goes with the factor and VARIANCE with its square (ZODI untouched), that
the bundle records what was applied, and that the spectral WCS / SAPM lookups
follow ``calibration_release`` while the PSF follows the image's release.
"""
import io

import numpy as np
import pytest
from astropy.io import fits

from spherex_retrieval import flux_correction as fc
from spherex_retrieval.bundle import Bundle, write_bundle
from spherex_retrieval.cutout import _payload_from_irsa_hdul
from tests.test_shared_psf import _mef_bytes as _qr2_mef_bytes


def _factor_file(tmp_path, ny=2040, nx=2040):
    rng = np.random.default_rng(0)
    rows = 0.97 - 0.05 * (np.arange(ny) / ny)                 # varies along y like the product
    f = np.repeat(rows[:, None], nx, axis=1) + 1e-3 * rng.normal(size=(ny, nx))
    f[210, 1841] = 3653.0                                     # a hot pixel inside the crop
    f[205, 1842] = 0.011                                      # a dead one
    f[300, 5] = np.nan                                        # outside the crop
    hdu = fits.ImageHDU(f.astype(np.float32), name="IMAGE")
    hdu.header["DETECTOR"] = "4"
    hdu.header["DETCOORD"] = "sky"
    hdu.header["COMMENT"] = "Multiplicative flux correction factor for pre-QR3 data"
    hdu.header["HISTORY"] = "Calibration source file: l3_flux_corrections_4_v2_20260528.fits"
    p = tmp_path / "flxc_D4.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(p)
    return p, f


def test_crop_replaces_wild_factors_with_the_row_median(tmp_path):
    p, f = _factor_file(tmp_path)
    # the QR2 test MEF's cutout sits at detector origin (1840, 201), 3 x 15 px
    corr = fc.crop_flux_correction(str(p), pixel_origin=(1840, 201), cutout_shape=(15, 3),
                                   token="cal-flxc-v1-2026-191")
    assert corr.data.shape == (15, 3) and corr.data.dtype == np.float32
    assert corr.n_replaced == 2 and corr.source_file == "l3_flux_corrections_4_v2_20260528.fits"
    # untouched pixels are the product's values; the two wild ones the row median
    assert np.isclose(corr.data[0, 0], f[201, 1840], rtol=1e-6)
    y_hot, x_hot = 210 - 201, 1841 - 1840
    assert 0.5 < corr.data[y_hot, x_hot] < 2.0
    assert np.isclose(corr.data[y_hot, x_hot], np.median(f[210, (f[210] > 0.5) & (f[210] < 2)]), rtol=1e-4)
    with pytest.raises(ValueError):
        fc.crop_flux_correction(str(p), pixel_origin=(-1, 0), cutout_shape=(2, 2))


def test_apply_scales_image_and_variance_only(tmp_path):
    p, _ = _factor_file(tmp_path)
    with fits.open(io.BytesIO(_qr2_mef_bytes())) as hdul:
        payload = _payload_from_irsa_hdul(hdul)
    img0, var0, zodi0 = payload.image.copy(), payload.variance.copy(), payload.zodi.copy()
    corr = fc.crop_flux_correction(str(p), pixel_origin=payload.pixel_origin,
                                   cutout_shape=payload.image.shape, token="t")
    fc.apply_flux_correction(payload, corr)
    np.testing.assert_allclose(payload.image, img0 * corr.data, rtol=1e-6)
    np.testing.assert_allclose(payload.variance, var0 * corr.data ** 2, rtol=1e-6)
    np.testing.assert_array_equal(payload.zodi, zodi0)
    assert payload.image.dtype == img0.dtype and payload.variance.dtype == var0.dtype
    # shape mismatch is an error, not a broadcast
    bad = fc.FluxCorrection(data=np.ones((2, 2), np.float32), source_url="", token="",
                            source_file="", n_replaced=0)
    with pytest.raises(ValueError):
        fc.apply_flux_correction(payload, bad)


def test_bundle_records_the_correction(tmp_path):
    with fits.open(io.BytesIO(_qr2_mef_bytes())) as hdul:
        payload = _payload_from_irsa_hdul(hdul)
    b = Bundle(obs_id="o", detector=4, collection="spherex_qr2", access_url="u", cloud_uri="",
               time_bounds_lower=0.0, coord_ra=0.0, coord_dec=0.0, cutout=payload)
    b.extras["flux_correction"] = {"token": "cal-flxc-v1-2026-191", "median": 0.9696,
                                   "source_file": "l3_flux_corrections_4_v2_20260528.fits",
                                   "n_replaced": 2}
    out = write_bundle(b, tmp_path / "c.fits")
    with fits.open(out) as hdul:
        h = hdul[0].header
        assert h["FLXCORR"] == "cal-flxc-v1-2026-191" and abs(h["FLXCMED"] - 0.9696) < 1e-6
        assert h["FLXCSRC"].startswith("l3_flux_corrections_4") and h["FLXCNREP"] == 2
    b.extras.pop("flux_correction")
    out2 = write_bundle(b, tmp_path / "d.fits")
    with fits.open(out2) as hdul:
        assert "FLXCORR" not in hdul[0].header


def test_retrieve_one_threads_the_calibration_release(monkeypatch, tmp_path):
    """QR2 image: PSF from its own release, wavelength/SAPM from the calibration
    release, the gain correction applied; R7 image: no correction."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    from spherex_retrieval import core

    seen = {"wcs": [], "sapm": [], "flxc": []}
    with fits.open(io.BytesIO(_qr2_mef_bytes())) as hdul:
        payload = _payload_from_irsa_hdul(hdul)
    p, _ = _factor_file(tmp_path)

    monkeypatch.setattr(core, "fetch_cutout", lambda **kw: payload)
    monkeypatch.setattr(core, "find_cal_product",
                        lambda det, **kw: seen["wcs"].append(kw["data_release"]) or ("http://wcs", ""))
    monkeypatch.setattr(core, "crop_wavelength_maps", lambda *a, **kw: None)
    monkeypatch.setattr(core, "find_sapm_product",
                        lambda det, **kw: seen["sapm"].append(kw["data_release"]) or ("http://sapm", ""))
    monkeypatch.setattr(core, "crop_sapm", lambda *a, **kw: None)
    monkeypatch.setattr(core, "find_flux_correction_product",
                        lambda det, **kw: seen["flxc"].append(kw["data_release"]) or (str(p), "", "cal-flxc-v1-2026-191"))
    row = Table(rows=[("2025W23_1C_0164_1", 4, "spherex_qr2", "http://x/level2_x_D4.fits", "", 0.0)],
                names=("obs_id", "detector", "collection", "access_url", "cloud_uri", "time_bounds_lower"))[0]
    img0 = payload.image.copy()
    b = core._retrieve_one(row=row, coord=SkyCoord(10, 20, unit="deg"), size=1 * u.arcmin,
                           cutout_backend="irsa", include_wavelength=True, include_sapm=True,
                           sapm_cal_token=None, subset_psf=False, data_release="qr2",
                           calibration_release="qr3", gain_correction=True, cache_dir=None,
                           fsspec_kwargs=None, query_backend="astroquery")
    assert seen == {"wcs": ["qr3"], "sapm": ["qr3"], "flxc": ["qr3"]}
    assert "flux_correction" in b.extras and b.extras["flux_correction"]["token"] == "cal-flxc-v1-2026-191"
    assert not np.array_equal(b.cutout.image, img0)
    # an R7 image is never corrected, and 'own' calibration follows the image
    with fits.open(io.BytesIO(_qr2_mef_bytes())) as hdul:
        payload2 = _payload_from_irsa_hdul(hdul)
    monkeypatch.setattr(core, "fetch_cutout", lambda **kw: payload2)
    seen = {"wcs": [], "sapm": [], "flxc": []}
    b2 = core._retrieve_one(row=row, coord=SkyCoord(10, 20, unit="deg"), size=1 * u.arcmin,
                            cutout_backend="irsa", include_wavelength=True, include_sapm=False,
                            sapm_cal_token=None, subset_psf=False, data_release="qr3",
                            calibration_release=None, gain_correction=True, cache_dir=None,
                            fsspec_kwargs=None, query_backend="astroquery")
    assert "flux_correction" not in b2.extras and b2.status.value == "ok"
    assert seen["wcs"] == ["qr3"] and seen["flxc"] == []
