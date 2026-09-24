"""Frame-level reads: raw pread with walked headers, many targets per read.

Synthetic full-frame L2 files with headers of different lengths stand in for
the five layouts of the QR2 archive. The raw reader must return exactly what
astropy returns, and a bundle cut from a frame read must equal the one the
per-target local backend cuts.
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

from spherex_retrieval import frame as sfr  # noqa: E402
from spherex_retrieval import index as sidx  # noqa: E402
from spherex_retrieval.bundle import RetrievalStatus  # noqa: E402
from spherex_retrieval.core import complete_bundle  # noqa: E402
from spherex_retrieval.cutout import fetch_cutout  # noqa: E402

N = 96
SIZE = 6.6 * 6.15 * u.arcsec          # 7 px
RNG = np.random.default_rng(11)


def _wcs(ra0=10.0, dec0=20.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [N / 2 + 0.5, N / 2 + 0.5]
    w.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    return w


def write_l2(path, *, pad_cards=(0, 0, 0, 0, 0), ra0=10.0, dec0=20.0, names=None):
    """A full-frame L2 MEF; ``pad_cards`` lengthen the IMAGE..PSF headers (layouts)."""
    names = names or ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "PSF")
    hdr = _wcs(ra0, dec0).to_header()
    hdr["DETECTOR"] = 4
    primary = fits.PrimaryHDU()
    primary.header["VERSION"] = "6.5.7"
    primary.header["MJD-BEG"], primary.header["MJD-END"] = 60880.1, 60880.2
    image = RNG.normal(size=(N, N)).astype(">f4")
    image[3, 5] = np.nan
    planes = [image, RNG.integers(0, 2**20, (N, N)).astype(">i4"),
              RNG.random((N, N)).astype(">f4"), RNG.random((N, N)).astype(">f4")]
    hdus = [primary, fits.ImageHDU(planes[0], header=hdr, name=names[0])]
    hdus += [fits.ImageHDU(p, name=n) for p, n in zip(planes[1:], names[1:4])]
    psf = fits.ImageHDU(np.ones((9, 11, 11), dtype=">f4"), name=names[4])
    psf.header["OVERSAMP"] = 10
    hdus += [psf, fits.BinTableHDU(Table({"x": [1.0]}), name="WCS-WAVE")]
    for hdu, n in zip(hdus[1:6], pad_cards):
        for i in range(n):
            hdu.header[f"PAD{i:05d}"] = (i, "lengthens the header like the real layouts")
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return planes


LAYOUTS = [(0, 0, 0, 0, 0), (40, 3, 0, 36, 5), (80, 80, 80, 80, 300)]


@pytest.mark.parametrize("pads", LAYOUTS)
@pytest.mark.parametrize("strategy", ["planes", "bands"])
def test_raw_reads_equal_astropy(tmp_path, pads, strategy):
    path = tmp_path / "level2_2025W30_1B_0325_1D4_spx_l2b-v20-2025-262.fits"
    write_l2(path, pad_cards=pads)
    with sfr.LocalFrame(path) as fr:
        fr.load(strategy, rows=[(0, N)] if strategy == "bands" else None)
        with fits.open(path) as h:
            for name in sfr.PIXEL_PLANES:
                got = fr.cut(name, slice(0, N), slice(0, N))
                np.testing.assert_array_equal(got, h[name].data)
            assert fr.image_header == h["IMAGE"].header
            assert fr.primary_header == h[0].header
            assert fr.psf_header == h["PSF"].header
        # planes: the header probe + one span; bands: at most the probe, four header
        # walks and one read per plane (tiny files put more inside the probe)
        assert fr.requests == 2 if strategy == "planes" else fr.requests <= 1 + 4 + 4


def test_bands_read_only_the_rows_asked(tmp_path):
    path = tmp_path / "f.fits"
    planes = write_l2(path, pad_cards=LAYOUTS[1])
    with sfr.LocalFrame(path) as fr:
        fr.load("bands", rows=[(10, 20), (15, 30), (60, 64)])     # merged to 10-30 and 60-64
        np.testing.assert_array_equal(fr.cut("VARIANCE", slice(12, 28), slice(3, 9)),
                                      planes[2][12:28, 3:9])
        with pytest.raises(KeyError):
            fr.cut("IMAGE", slice(40, 45), slice(0, 5))
        # only the merged row ranges were read, for every plane
        for name in sfr.PIXEL_PLANES:
            assert [(r0, r1) for r0, r1, _ in fr._bands[name]] == [(10, 30), (60, 64)]


def _archive(tmp_path, frames):
    """A local archive of full frames + its index. ``frames``: [(name, ra0, dec0, pads)]."""
    root = tmp_path / "archive"
    rows = []
    for name, ra0, dec0, pads in frames:
        m = sidx.LEVEL2_NAME_RE.match(name)
        rel = f"{m['week']}/{m['procver']}/{m['det']}/{name}"
        write_l2(root / "repo/level2" / rel, pad_cards=pads, ra0=ra0, dec0=dec0)
        corners = _wcs(ra0, dec0).pixel_to_world_values([-0.5, N - 0.5, N - 0.5, -0.5],
                                                        [-0.5, -0.5, N - 0.5, N - 0.5])
        rows.append({"calib_level": 2, "obs_id": f"{m['week']}_{m['expo']}_{m['dither']}",
                     "obs_creator_did": f"{m['week']}_{m['expo']}_{m['dither']}D{m['det']}",
                     "s_ra": ra0, "s_dec": dec0, "t_min": 60880.1, "t_max": 60880.2,
                     "em_min": 2.4e-6, "em_max": 3.8e-6,
                     "s_region": "POLYGON ICRS " + " ".join(f"{a} {d}" for a, d in zip(*corners)),
                     "access_url": f"file:///data/level2/{rel}"})
    (root / "loads/L1/obscore").mkdir(parents=True)
    Table(rows=rows).write(root / "loads/L1/obscore/obscore_L1.tbl", format="ipac")
    table, report = sidx.build_index(root, workers=1, release="qr2")
    return root, sidx.write_index(table, tmp_path / "index.parquet", report)


FRAME_A = "level2_2025W30_1B_0325_1D4_spx_l2b-v20-2025-262.fits"
FRAME_B = "level2_2025W30_1B_0326_2D4_spx_l2b-v20-2025-262.fits"
KW = dict(include_wavelength=False, subset_psf=False, data_release="qr2", gain_correction=False)


def _per_target(path, coord):
    """What retrieve(query_backend="local", cutout_backend="local") makes for one target."""
    from spherex_retrieval.bundle import Bundle

    b = Bundle(obs_id="x", detector=4, collection="spherex_qr2", access_url=str(path),
               cloud_uri="", time_bounds_lower=60880.1, coord_ra=coord.ra.deg,
               coord_dec=coord.dec.deg)
    b.cutout = fetch_cutout(access_url=str(path), cloud_uri="", coord=coord, size=SIZE,
                            backend="local")
    return complete_bundle(b, coord=coord, cutout_backend="local", include_sapm=False,
                           sapm_cal_token=None, cache_dir=None, fsspec_kwargs=None,
                           query_backend="local", **KW)


@pytest.mark.parametrize("strategy", ["planes", "bands", "auto"])
def test_frame_bundles_equal_the_per_target_ones(tmp_path, strategy):
    path = tmp_path / FRAME_A
    write_l2(path, pad_cards=LAYOUTS[1])
    w = _wcs()
    xy = [(3.2, 4.9), (47.5, 47.5), (90.1, 12.6), (50.0, 93.4), (20.7, 60.2)]
    coords = SkyCoord(*w.pixel_to_world_values(*np.array(xy).T), unit="deg")
    got = list(sfr.iter_frame_cutouts(path, coords, SIZE, obs_id="x", detector=4,
                                      collection="spherex_qr2", time_bounds_lower=60880.1,
                                      access_strategy=strategy, **KW))
    assert [b.status for b in got] == [RetrievalStatus.OK] * len(xy)
    for b, coord in zip(got, coords):
        ref = _per_target(path, coord)
        assert b.cutout.pixel_origin == ref.cutout.pixel_origin
        for key in ("image", "flags", "variance", "zodi", "psf_cube"):
            np.testing.assert_array_equal(getattr(b.cutout, key), getattr(ref.cutout, key))
        assert b.cutout.image_header == ref.cutout.image_header
        assert b.cutout.primary_header == ref.cutout.primary_header
        assert b.cutout.psf_header == ref.cutout.psf_header


def test_targets_off_the_frame_are_out_of_bounds(tmp_path):
    path = tmp_path / FRAME_A
    write_l2(path)
    coords = SkyCoord([10.0, 12.0], [20.0, 20.0], unit="deg")
    got = list(sfr.iter_frame_cutouts(path, coords, SIZE, obs_id="x", detector=4,
                                      collection="spherex_qr2", **KW))
    assert [b.status for b in got] == [RetrievalStatus.OK, RetrievalStatus.OUT_OF_BOUNDS]


def test_an_unknown_layout_falls_back_to_astropy(tmp_path):
    path = tmp_path / FRAME_A
    write_l2(path, names=("IMAGE", "FLAGS", "NOISE", "ZODI", "PSF"))   # not the L2 sequence
    coords = SkyCoord([10.0], [20.0], unit="deg")
    with pytest.raises(sfr.FrameLayoutError):
        with sfr.LocalFrame(path) as fr:
            fr.load("planes")
    (b,) = sfr.iter_frame_cutouts(path, coords, SIZE, obs_id="x", detector=4,
                                  collection="spherex_qr2", **KW)
    assert b.status == RetrievalStatus.OK
    np.testing.assert_array_equal(b.cutout.image, _per_target(path, coords[0]).cutout.image)


def test_retrieve_catalog_reads_each_frame_once(tmp_path):
    root, idx = _archive(tmp_path, [(FRAME_A, 10.0, 20.0, LAYOUTS[0]),
                                    (FRAME_B, 10.1, 20.05, LAYOUTS[2])])
    coords = SkyCoord([10.0, 10.05, 10.12, 30.0], [20.0, 20.02, 20.06, -5.0], unit="deg")
    stats = []
    results = list(sfr.retrieve_catalog(coords, SIZE, index=idx, psf_source="l2",
                                        calibration_release=None, max_workers=2,
                                        frame_stats=stats, **{k: v for k, v in KW.items()
                                                              if k != "data_release"}))
    # each frame read once; a band read per plane per distinct row range
    assert sorted(s["frame"] for s in stats) == [0, 1]
    assert all(s["requests"] <= 1 + 4 + 4 * s["inside"] for s in stats)
    ok = {(t, b.obs_id) for t, b in results if b.is_ok}
    assert {t for t, _ in ok} == {0, 1, 2}          # the far target has no frame
    for t, b in results:
        if b.is_ok:
            ref = _per_target(b.access_url, coords[t])
            np.testing.assert_array_equal(b.cutout.image, ref.cutout.image)
            assert b.cutout.image_header == ref.cutout.image_header
            assert b.collection == "spherex_qr2" and b.time_bounds_lower == 60880.1
    out = sfr.write_catalog_bundles(results, tmp_path / "out")
    assert sorted(out) == sorted({t for t, _ in results})
    for t, d in out.items():
        assert (d / "summary.ecsv").exists()
