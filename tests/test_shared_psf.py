"""psf_source="cal": share one PSF cube per detector, stop downloads before the PSF data.

The PSF cube IRSA passes through with every cutout is a per-detector
constant and ~97 % of a small cutout's bytes. These tests pin the three
pieces that make skipping it safe: the header scanner that decides where to
hang up, the streaming fetch that actually hangs up, and the registry that
samples full downloads to check the cal cube against the L2 one.
"""
import io
import threading
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from spherex_retrieval import io as sio
from spherex_retrieval.bundle import Bundle, write_bundle
from spherex_retrieval.cutout import (_payload_from_irsa_hdul, _PsfHeaderScanner,
                                      fetch_irsa_cutout)
from spherex_retrieval.psf_shared import SharedPsfRegistry

RNG = np.random.default_rng(0)
CUBE = RNG.normal(size=(121, 11, 11)).astype(">f4")


def _mef_bytes(cube=CUBE, *, name_psf=True) -> bytes:
    """A cutout MEF laid out the way IRSA's cutout service returns it."""
    img_hdr = fits.Header()
    img_hdr["DETECTOR"] = 4
    img_hdr["CRPIX1A"] = -1839
    img_hdr["CRPIX2A"] = -200
    for k, v in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 10.0), ("CRVAL2", 20.0),
                 ("CRPIX1", 2.0), ("CRPIX2", 8.0), ("CDELT1", -6.2 / 3600), ("CDELT2", 6.2 / 3600)):
        img_hdr[k] = v
    primary = fits.PrimaryHDU()
    primary.header["VERSION"] = "6.5.7"
    psf = fits.ImageHDU(cube, name="PSF" if name_psf else None)
    psf.header["OVERSAMP"] = 10
    for i in range(1, 122):          # a multi-block header, like the real one
        psf.header[f"XCTR_{i}"] = float(i)
        psf.header[f"YCTR_{i}"] = float(i)
    hdus = [primary, fits.ImageHDU(RNG.normal(size=(15, 3)).astype(">f4"), header=img_hdr, name="IMAGE")]
    hdus += [fits.ImageHDU(RNG.normal(size=(15, 3)).astype(">f4"), name=n)
             for n in ("FLAGS", "VARIANCE", "ZODI")]
    hdus += [psf, fits.BinTableHDU(Table({"x": [1.0, 2.0]}), name="WCS-WAVE")]
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    return buf.getvalue()


def _psf_data_offset(body: bytes) -> int:
    with fits.open(io.BytesIO(body)) as hdul:
        return hdul.fileinfo(5)["datLoc"]


@pytest.mark.parametrize("chunk", [1, 79, 2880, 5000, 10**7])
def test_scanner_stops_at_psf_data_start(chunk):
    body = _mef_bytes()
    scanner, buf, stop = _PsfHeaderScanner(), bytearray(), None
    for i in range(0, len(body), chunk):
        buf += body[i:i + chunk]
        stop = scanner(buf)
        if stop is not None:
            break
    assert stop == _psf_data_offset(body)
    assert len(buf) < stop + chunk          # never asked for more than one chunk past it
    hdr = fits.Header.fromstring(bytes(buf[slice(*scanner.psf_span)]).decode("ascii"))
    assert hdr["EXTNAME"] == "PSF" and hdr["XCTR_121"] == 121.0


def test_scanner_falls_back_to_hdu_index_when_extname_missing():
    body = _mef_bytes(name_psf=False)
    assert _PsfHeaderScanner()(bytearray(body)) == _psf_data_offset(body)


def test_scanner_never_fires_without_a_psf_hdu():
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((3, 3)), name="IMAGE")]).writeto(buf)
    assert _PsfHeaderScanner()(bytearray(buf.getvalue())) is None


class _FakeResponse:
    status_code, reason = 200, "OK"

    def __init__(self, body, log):
        self._body, self._log = body, log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._log["closed"] = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            self._log["sent"] = i + chunk_size
            yield self._body[i:i + chunk_size]


@pytest.fixture
def fake_irsa(monkeypatch, tmp_path):
    """Serve ``_mef_bytes()`` for any URL; record how much of it was pulled."""
    log = {"sent": 0, "requests": 0}
    body = _mef_bytes()

    def _get(url, **kw):
        log["requests"] += 1
        log["sent"] = 0
        return _FakeResponse(body, log)

    monkeypatch.setattr(sio.requests, "get", _get)
    monkeypatch.setenv("SPHEREX_RETRIEVAL_CACHE", str(tmp_path / "cache"))
    return log, body


def test_http_fetch_until_hangs_up_early(fake_irsa):
    log, body = fake_irsa
    got, stopped = sio.http_fetch_until("http://x", _PsfHeaderScanner(), chunk_size=4096)
    off = _psf_data_offset(body)
    assert stopped and got == body[:off]
    assert log["closed"] and log["sent"] < off + 4096 < len(body)


def test_http_fetch_until_returns_whole_body_if_never_stopped(fake_irsa):
    _, body = fake_irsa
    got, stopped = sio.http_fetch_until("http://x", lambda buf: None)
    assert not stopped and got == body


def _registry(cube=CUBE, **kw):
    shared = np.array(cube, copy=True)
    shared.flags.writeable = False
    return SharedPsfRegistry(loader=lambda det: (shared, f"average_psf_D{det}_spx_cal-psf-v5.fits"), **kw)


URL = "https://irsa/level2_2025W22_2B_0161_1D4_spx_l2b-v20-2025-251.fits"


def test_light_payload_matches_full_payload(fake_irsa):
    log, body = fake_irsa
    coord = SkyCoord(10.0, 20.0, unit="deg")
    with fits.open(io.BytesIO(body)) as hdul:
        ref = _payload_from_irsa_hdul(hdul)

    reg = _registry(verify_every=0)
    first = fetch_irsa_cutout(URL, coord, 1 * u.arcmin, psf_registry=reg)
    assert first.psf_source == "l2" and log["sent"] >= len(body)

    # a different position -> a different cutout URL, so the disk cache is not hit
    second = fetch_irsa_cutout(URL, SkyCoord(10.1, 20.0, unit="deg"), 1 * u.arcmin,
                               psf_registry=reg)
    assert second.psf_source == "cal:average_psf_D4_spx_cal-psf-v5.fits"
    # hung up within one 16 KiB chunk of the PSF data start, short of the body's end
    assert log["sent"] < _psf_data_offset(body) + (1 << 14) < len(body)
    for name in ("image", "flags", "variance", "zodi", "psf_cube"):
        np.testing.assert_array_equal(getattr(second, name), getattr(ref, name))
    assert second.psf_cube.dtype == ref.psf_cube.dtype
    assert second.psf_header.tostring() == ref.psf_header.tostring()
    assert second.pixel_origin == ref.pixel_origin == (1840, 201)
    assert second.detector == 4 and second.psf_oversamp == 10


def _drive(reg, n, *, l2_cube=CUBE, det=4):
    """Run n fetches; return the list of 'full' / 'light' decisions."""
    calls = []

    def full():
        calls.append("full")
        return {"cube": l2_cube}

    def light(cube):
        calls.append("light")
        return {"cube": cube}

    tags = [reg.fetch(det, fetch_full=full, fetch_light=light, cube_of=lambda p: p["cube"])[1]
            for _ in range(n)]
    return calls, tags


def test_registry_checks_first_cutout_then_samples():
    calls, tags = _drive(_registry(verify_every=4), 10)
    #        first   1        2        3        4th     5 ...
    assert calls == ["full", "light", "light", "light", "full", "light", "light", "light", "full", "light"]
    assert [t == "l2" for t in tags] == [c == "full" for c in calls]


def test_registry_is_per_detector():
    reg = _registry(verify_every=0)
    assert _drive(reg, 2, det=1)[0] == ["full", "light"]
    assert _drive(reg, 2, det=2)[0] == ["full", "light"]


def test_first_cutout_mismatch_disables_sharing():
    reg = _registry(verify_every=0)
    with pytest.warns(RuntimeWarning, match="differs from"):
        calls, tags = _drive(reg, 3, l2_cube=CUBE + 1)
    assert calls == ["full"] * 3 and set(tags) == {"l2"} and reg.is_disabled(4)


def test_sampled_mismatch_disables_sharing_and_names_the_exposure():
    reg = _registry(verify_every=3)
    assert _drive(reg, 3)[0] == ["full", "light", "light"]
    with pytest.warns(RuntimeWarning, match="re-retrieved"):
        calls, _ = _drive(reg, 3, l2_cube=CUBE + 1)     # 3rd counted fetch is the sample
    assert calls == ["full", "full", "full"]


def test_nan_cubes_compare_equal():
    cube = CUBE.copy()
    cube[0, 0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _drive(_registry(cube, verify_every=0), 2, l2_cube=cube.copy())[0] == ["full", "light"]


def test_unavailable_cal_product_falls_back_to_full_downloads():
    def boom(det):
        raise RuntimeError("no listing")

    reg = SharedPsfRegistry(loader=boom)
    with pytest.warns(RuntimeWarning, match="unavailable"):
        calls, _ = _drive(reg, 3)
    assert calls == ["full"] * 3


def test_failed_first_download_leaves_detector_unverified():
    reg = _registry(verify_every=0)

    def broken():
        raise OSError("503")

    with pytest.raises(OSError):
        reg.fetch(4, fetch_full=broken, fetch_light=lambda c: None, cube_of=lambda p: p)
    assert _drive(reg, 2)[0] == ["full", "light"]


def test_concurrent_first_use_checks_once():
    reg = _registry(verify_every=0)
    calls, lock = [], threading.Lock()

    def full():
        with lock:
            calls.append("full")
        return {"cube": CUBE}

    def light(cube):
        with lock:
            calls.append("light")
        return {"cube": cube}

    threads = [threading.Thread(target=lambda: reg.fetch(
        4, fetch_full=full, fetch_light=light, cube_of=lambda p: p["cube"])) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls.count("full") == 1 and calls.count("light") == 15


def test_bundle_records_psf_source_and_writes_readonly_cube(tmp_path):
    with fits.open(io.BytesIO(_mef_bytes())) as hdul:
        shared = np.array(CUBE, copy=True)
        shared.flags.writeable = False
        payload = _payload_from_irsa_hdul(hdul, psf_cube=shared, psf_header=hdul["PSF"].header.copy())
    payload.psf_source = "cal:average_psf_D4_spx_cal-psf-v5-2026-082.fits"
    b = Bundle(obs_id="o", detector=4, collection="c", access_url="u", cloud_uri="s",
               time_bounds_lower=0.0, coord_ra=0.0, coord_dec=0.0, cutout=payload)
    out = write_bundle(b, tmp_path / "c.fits")
    with fits.open(out) as hdul:
        assert hdul[0].header["PSFSRC"] == payload.psf_source
        np.testing.assert_array_equal(hdul["PSF"].data, CUBE)


def test_cal_discovery_is_memoised_per_detector(monkeypatch):
    from spherex_retrieval import cal_index

    n = {"sia": 0}

    def fake_sia(family, detector, **kw):
        n["sia"] += 1
        return (f"http://{family}/D{detector}", f"s3://{family}/D{detector}")

    monkeypatch.setattr(cal_index, "find_via_sia", fake_sia)
    monkeypatch.setattr(cal_index, "_DISCOVERED", {})
    for ra in (1.0, 2.0, 3.0):
        got = cal_index.discover_cal_product("spectral_wcs", 4, coord=SkyCoord(ra, 0.0, unit="deg"))
    assert got == ("http://spectral_wcs/D4", "s3://spectral_wcs/D4") and n["sia"] == 1
    cal_index.discover_cal_product("spectral_wcs", 5)
    cal_index.discover_cal_product("solid_angle_pixel_map", 4)
    assert n["sia"] == 3
