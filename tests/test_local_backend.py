"""Local archive backends: calibration trees, index queries and in-place crops.

A local archive serves the retrieval without the network: cal products come
from a tree laid out like IRSA's ``spherex/<release>/`` and the L2 files from
the archive index. These tests build a one-frame archive and never reach IRSA
(the network entry points are replaced by functions that fail the test).
"""
import os

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

pytest.importorskip("pyarrow")
pytest.importorskip("scipy")

from spherex_retrieval import cal_index, core, flux_correction, psf_shared, query  # noqa: E402
from spherex_retrieval import index as sidx  # noqa: E402
from spherex_retrieval.cutout import fetch_cutout  # noqa: E402

NAME = "level2_2025W30_1B_0325_1D4_spx_l2b-v20-2025-262.fits"
NPIX = 64
RNG = np.random.default_rng(7)
SIZE = 4.6 * 6.15 * u.arcsec   # 5 px (not on a whole-pixel boundary)


@pytest.fixture(autouse=True)
def _no_network_and_clean_roots(monkeypatch):
    def offline(*a, **kw):
        raise AssertionError("tried to reach IRSA")

    monkeypatch.setattr(cal_index, "find_via_sia", offline)
    monkeypatch.setattr(cal_index, "latest_cal_token_via_listing", offline)
    monkeypatch.delenv("SPHEREX_CAL_ROOTS", raising=False)
    cal_index.set_local_cal_roots(None)
    yield
    cal_index.set_local_cal_roots(None)


def _cal_file(root, family, token, det, hdus=None):
    d = root / family / token / str(det)
    d.mkdir(parents=True, exist_ok=True)
    path = d / cal_index.cal_filename(family, det, token)
    fits.HDUList(hdus or [fits.PrimaryHDU()]).writeto(path)
    return path


def test_local_cal_product_takes_the_latest_token_holding_the_detector(tmp_path):
    root = tmp_path / "r7"
    for det in (1, 2, 3):
        _cal_file(root, "epsf", "cal-epsf-v1-2026-191", det)
    _cal_file(root, "epsf", "cal-epsf-v2-2026-191", 3)     # re-issued for D3 alone
    (root / "epsf" / "not-a-token").mkdir()
    cal_index.set_local_cal_roots({"qr3": root})
    path, token = cal_index.local_cal_product("epsf", 3, data_release="qr3")
    assert token == "cal-epsf-v2-2026-191" and os.path.isfile(path)
    assert cal_index.local_cal_product("epsf", 1, data_release="qr3")[1] == "cal-epsf-v1-2026-191"
    pinned = cal_index.local_cal_product("epsf", 3, data_release="qr3",
                                         cal_token="cal-epsf-v1-2026-191")
    assert pinned[1] == "cal-epsf-v1-2026-191"
    with pytest.raises(FileNotFoundError):
        cal_index.local_cal_product("epsf", 4, data_release="qr3")
    assert cal_index.local_cal_product("epsf", 1, data_release="qr2") is None


def test_roots_from_the_environment_and_their_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHEREX_CAL_ROOTS", f"qr2={tmp_path / 'a'}, qr3={tmp_path / 'b'}")
    assert cal_index.local_cal_roots() == {"qr2": tmp_path / "a", "qr3": tmp_path / "b"}
    cal_index.set_local_cal_roots({"qr3": tmp_path / "c"})
    assert cal_index.local_cal_roots() == {"qr2": tmp_path / "a", "qr3": tmp_path / "c"}


def test_every_cal_family_resolves_locally_without_the_network(tmp_path):
    q2, r7 = tmp_path / "qr2", tmp_path / "r7"
    wcs_path = _cal_file(q2, "spectral_wcs", "cal-wcs-v4-2025-254", 4)
    _cal_file(q2, "spectral_wcs", "cal-wcs-v2-2025-246", 4)
    sapm_path = _cal_file(r7, "solid_angle_pixel_map", "cal-sapm-v3-2026-191", 4)
    psf_path = _cal_file(q2, "average_psf", "cal-psf-v5-2026-082", 4)
    epsf_path = _cal_file(r7, "epsf", "cal-epsf-v1-2026-191", 4)
    flxc_path = _cal_file(r7, "l3_flux_corrections", "cal-flxc-v1-2026-191", 4)
    cal_index.set_local_cal_roots({"qr2": q2, "qr3": r7})
    assert cal_index.discover_cal_product("spectral_wcs", 4, data_release="qr2") == (str(wcs_path), "")
    assert cal_index.discover_cal_product("solid_angle_pixel_map", 4, data_release="qr3") == \
        (str(sapm_path), "")
    assert psf_shared.find_psf_product("optical", 4, data_release="qr2") == (str(psf_path), "")
    assert psf_shared.find_psf_product("effective", 4, data_release="qr3") == (str(epsf_path), "")
    assert flux_correction.find_flux_correction_product(4, data_release="qr3") == \
        (str(flxc_path), "", "cal-flxc-v1-2026-191")


def _frame_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [NPIX / 2 + 0.5, NPIX / 2 + 0.5]
    w.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    return w


def _write_l2(path):
    hdr = _frame_wcs().to_header()
    hdr["DETECTOR"] = 4
    primary = fits.PrimaryHDU()
    primary.header["VERSION"] = "6.5.7"
    primary.header["MJD-BEG"] = 60880.1
    primary.header["MJD-END"] = 60880.2
    planes = {n: RNG.normal(size=(NPIX, NPIX)).astype(">f4") for n in ("IMAGE", "VARIANCE", "ZODI")}
    hdus = [primary, fits.ImageHDU(planes["IMAGE"], header=hdr, name="IMAGE"),
            fits.ImageHDU(RNG.integers(0, 4, (NPIX, NPIX)).astype(">i4"), name="FLAGS"),
            fits.ImageHDU(planes["VARIANCE"], name="VARIANCE"),
            fits.ImageHDU(planes["ZODI"], name="ZODI")]
    psf = fits.ImageHDU(np.ones((9, 11, 11), dtype=">f4"), name="PSF")
    psf.header["OVERSAMP"] = 10
    hdus += [psf, fits.BinTableHDU(Table({"x": [1.0]}), name="WCS-WAVE")]
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path)
    return planes["IMAGE"]


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    m = sidx.LEVEL2_NAME_RE.match(NAME)
    rel = f"{m['week']}/{m['procver']}/{m['det']}/{NAME}"
    image = _write_l2(root / "loads/L1/data/level2" / rel)
    corners = _frame_wcs().pixel_to_world_values([-0.5, NPIX - 0.5, NPIX - 0.5, -0.5],
                                                 [-0.5, -0.5, NPIX - 0.5, NPIX - 0.5])
    poly = " ".join(f"{a} {d}" for a, d in zip(*corners))
    obscore = root / "loads/L1/obscore/obscore_L1.tbl"
    obscore.parent.mkdir(parents=True)
    Table(rows=[{"calib_level": 2, "obs_id": "2025W30_1B_0325_1",
                 "obs_creator_did": "2025W30_1B_0325_1D4", "s_ra": 10.0, "s_dec": 20.0,
                 "t_min": 60880.1, "t_max": 60880.2, "em_min": 2.4e-6, "em_max": 3.8e-6,
                 "s_region": f"POLYGON ICRS {poly}",
                 "access_url": f"file:///data/level2/{rel}"}]).write(obscore, format="ipac")
    (root / "repo/level2").mkdir(parents=True)
    os.symlink(f"../../loads/L1/data/level2/{m['week']}", root / "repo/level2" / m["week"])
    table, report = sidx.build_index(root, workers=1, release="qr2")
    out = sidx.write_index(table, tmp_path / "index.parquet", report)
    return root, out, image


def test_query_local_labels_frames_with_the_release(archive):
    root, idx, _ = archive
    coord = SkyCoord(10.001, 20.002, unit="deg")
    t = query.find_overlapping(coord, 5 * 6.15 * u.arcsec, backend="local", index=idx)
    assert len(t) == 1
    row = t[0]
    # the recorded (resolved) root plus the index path: the week link stays unresolved
    assert row["access_url"] == str(root.resolve() / "repo/level2/2025W30_1B/l2b-v20-2025-262/4" / NAME)
    assert row["collection"] == "spherex_qr2" and row["detector"] == 4
    assert row["bandpass"] == "SPHEREx-D4" and row["obs_id"] == "2025W30_1B_0325_1"
    far = SkyCoord(12.0, 20.0, unit="deg")
    assert len(query.find_overlapping(far, 5 * 6.15 * u.arcsec, backend="local", index=idx)) == 0
    assert len(query.find_overlapping(coord, 5 * 6.15 * u.arcsec, backend="local", index=idx,
                                      bandpass="SPHEREx-D2")) == 0


def test_local_cutout_crops_the_file_in_place(archive):
    root, idx, image = archive
    coord = SkyCoord(10.001, 20.002, unit="deg")
    row = query.find_overlapping(coord, 5 * 6.15 * u.arcsec, backend="local", index=idx)[0]
    payload = fetch_cutout(access_url=row["access_url"], cloud_uri="", coord=coord,
                           size=SIZE, backend="local")
    x0, y0 = payload.pixel_origin
    ny, nx = payload.image.shape
    assert (ny, nx) == (5, 5)
    np.testing.assert_array_equal(payload.image, image[y0:y0 + ny, x0:x0 + nx])
    # the cutout WCS puts the target on the same sky position as the frame WCS
    fx, fy = _frame_wcs().world_to_pixel_values(10.001, 20.002)
    cx, cy = payload.spatial_wcs.world_to_pixel_values(10.001, 20.002)
    assert np.allclose([cx + x0, cy + y0], [fx, fy])
    assert payload.image_header["CRPIX1A"] == x0 + 1


def test_retrieve_end_to_end_offline(archive, tmp_path):
    root, idx, image = archive
    coord = SkyCoord(10.001, 20.002, unit="deg")
    bundles, out = core.retrieve(coord, SIZE, output_dir=tmp_path / "out",
                                 query_backend="local", cutout_backend="local", index=idx,
                                 psf_source="l2", subset_psf=False, include_wavelength=False,
                                 max_workers=1)
    assert [b.status.value for b in bundles] == ["ok"]
    files = sorted(out.glob("cutout_*.fits"))
    assert len(files) == 1
    with fits.open(files[0]) as h:
        x0, y0 = bundles[0].cutout.pixel_origin
        assert h["IMAGE"].data.shape == (5, 5)
        np.testing.assert_array_equal(h["IMAGE"].data, image[y0:y0 + 5, x0:x0 + 5])


def test_header_scan_reproduces_the_obscore_row(archive, tmp_path):
    root, idx, _ = archive
    from_obscore = sidx.read_index(idx).to_pylist()[0]
    table, report = sidx.build_index(root, obscore_glob="no-such-tables/*.tbl", workers=1,
                                     release="qr2")
    assert report["header_rows"] == 1 and report["unindexed"] == []
    from_headers = table.to_pylist()[0]
    assert from_headers["meta"] == "header" and from_obscore["meta"] == "obscore"
    assert from_headers["load"] == "L1"
    for key in ("obs_creator_did", "obs_id", "path", "size", "detector", "week", "procver"):
        assert from_headers[key] == from_obscore[key]
    for key in ("s_ra", "s_dec", *sidx._CORNER_COLUMNS):
        assert from_headers[key] == pytest.approx(from_obscore[key], abs=1e-9)
    assert (from_headers["em_min"], from_headers["em_max"]) == sidx.DETECTOR_BAND_M[4]
