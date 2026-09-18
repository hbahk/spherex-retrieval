"""R7 (QR3/DR1) effective PSF: the ``EPSF`` binary table through the retrieval.

An R7 L2 MEF carries its PSF as a binary table of 33x33 effective PSFs at 5x,
one row per zone of a 21x21 (D3: 11x41) lattice, instead of the QR2 optical
cube. These tests pin down that the header scanner hangs up at that table,
that a light download can carry the shared per-detector library in its place
and is checked by the library's provenance string, that the zone lattice is
read from the table (any size), and that the bundle says which kind of PSF it
holds so downstream photometry point-samples it instead of integrating it.
"""
import io
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from spherex_retrieval import io as sio
from spherex_retrieval.bundle import Bundle, write_bundle
from spherex_retrieval.cutout import (
    _payload_from_irsa_hdul,
    _PsfHeaderScanner,
    fetch_irsa_cutout,
)
from spherex_retrieval.psf import (
    subset_zones_for_cutout,
    zone_lattice,
    zone_table_from_epsf,
)
from spherex_retrieval.psf_shared import (
    EpsfLibrary,
    SharedPsfRegistry,
    epsf_library_from_hdu,
    epsf_source_file,
    psf_kind_of_release,
)
from spherex_retrieval.query import (
    observation_id_from_filename,
    release_of_collection,
    release_of_url,
)

RNG = np.random.default_rng(1)
SOURCE_FILE = "epsf_4_20260507.fits"


def _epsf_table(nx=21, ny=21, det=4, seed=1):
    """A synthetic EPSF bintable: BINX/BINY lattice, 0-based centres, unit-sum arrays."""
    rng = np.random.default_rng(seed)
    xw, yw = 2040.0 / nx, 2040.0 / ny
    binx, biny, arrs = [], [], []
    yy, xx = np.indices((33, 33))
    for by in range(ny):
        for bx in range(nx):
            sig = 2.0 + 0.5 * bx / max(nx - 1, 1)
            arr = np.exp(-0.5 * ((xx - 16) ** 2 + (yy - 16) ** 2) / sig ** 2)
            arr = arr / arr.sum() + 1e-6 * rng.normal(size=(33, 33))
            binx.append(bx)
            biny.append(by)
            arrs.append(arr.ravel())
    binx, biny = np.array(binx, np.int32), np.array(biny, np.int32)
    t = Table({
        "BINX": binx, "BINY": biny,
        "XCENTER": ((binx + 0.5) * xw - 0.5).astype(np.float32),
        "YCENTER": ((biny + 0.5) * yw - 0.5).astype(np.float32),
        "XWIDTH": np.full(len(binx), xw, np.float32), "YWIDTH": np.full(len(binx), yw, np.float32),
        "NSTAR": (500 + binx).astype(np.int32), "NEFF_MEAN": np.full(len(binx), 3.4, np.float32),
        "CWAVE": (1.0 + 0.01 * binx).astype(np.float32), "CBAND": np.full(len(binx), 0.03, np.float32),
        "EPSF": np.stack(arrs).astype(np.float64),
    })
    hdu = fits.BinTableHDU(t, name="EPSF")
    hdu.header["DETECTOR"] = str(det)
    hdu.header["OVSMPX"] = 5
    hdu.header["OVSMPY"] = 5
    hdu.header["DETCOORD"] = "sky"
    hdu.header["ORDERING"] = "x-fast then y"
    hdu.header["HISTORY"] = "Headers standardized by SSDC on 2026-06-26T22:53:35.593 (UTC)"
    hdu.header["HISTORY"] = f"Calibration source file: {SOURCE_FILE}"
    return hdu


def _mef_bytes(epsf_hdu=None, *, det=4, version="7.0.5") -> bytes:
    """An R7 cutout MEF laid out the way IRSA's cutout service returns it."""
    epsf_hdu = epsf_hdu if epsf_hdu is not None else _epsf_table(det=det)
    img_hdr = fits.Header()
    img_hdr["DETECTOR"] = det
    img_hdr["CRPIX1A"] = -931
    img_hdr["CRPIX2A"] = -933
    for k, v in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 10.0), ("CRVAL2", 20.0),
                 ("CRPIX1", 2.0), ("CRPIX2", 8.0), ("CDELT1", -6.2 / 3600), ("CDELT2", 6.2 / 3600)):
        img_hdr[k] = v
    primary = fits.PrimaryHDU()
    primary.header["VERSION"] = version
    hdus = [primary, fits.ImageHDU(RNG.normal(size=(15, 3)).astype(">f4"), header=img_hdr, name="IMAGE")]
    hdus += [fits.ImageHDU(RNG.normal(size=(15, 3)).astype(">f4"), name=n)
             for n in ("FLAGS", "VARIANCE", "ZODI")]
    hdus += [epsf_hdu, fits.BinTableHDU(Table({"x": [1.0, 2.0]}), name="WCS-WAVE")]
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    return buf.getvalue()


def _epsf_data_offset(body: bytes) -> int:
    with fits.open(io.BytesIO(body)) as hdul:
        return hdul.fileinfo(5)["datLoc"]


# --------------------------------------------------------------------------- #
# release / kind bookkeeping
# --------------------------------------------------------------------------- #
def test_release_and_kind_helpers():
    assert release_of_collection("spherex_qr3_deep") == "qr3"
    assert release_of_collection("spherex_qr2") == "qr2"
    assert release_of_collection("spherex_dr1") == "dr1"
    with pytest.raises(ValueError):
        release_of_collection("wise_allsky")
    assert release_of_url("https://irsa.ipac.caltech.edu/ibe/data/spherex/qr3/level2/x.fits") == "qr3"
    assert release_of_url("https://example.org/x.fits") is None
    assert observation_id_from_filename(
        "ibe/data/spherex/qr3/level2/2026W32_1A/l2b-v27-2026-223/1/"
        "level2_2026W32_1A_0001_1D1_spx_l2b-v27-2026-223.fits") == "2026W32_1A_0001_1"
    assert psf_kind_of_release("qr2") == "optical"
    assert psf_kind_of_release("qr3") == "effective"
    assert psf_kind_of_release("dr1") == "effective"


# --------------------------------------------------------------------------- #
# scanner + payload
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", [79, 2880, 10**7])
def test_scanner_stops_at_the_epsf_table(chunk):
    body = _mef_bytes()
    scanner, buf, stop = _PsfHeaderScanner(), bytearray(), None
    for i in range(0, len(body), chunk):
        buf += body[i:i + chunk]
        stop = scanner(buf)
        if stop is not None:
            break
    assert stop == _epsf_data_offset(body)
    assert scanner.extname == "EPSF"
    hdr = fits.Header.fromstring(bytes(buf[slice(*scanner.psf_span)]).decode("ascii"))
    assert hdr["XTENSION"] == "BINTABLE" and hdr["OVSMPX"] == 5
    assert epsf_source_file(hdr) == SOURCE_FILE


def test_full_payload_is_effective_kind():
    with fits.open(io.BytesIO(_mef_bytes())) as hdul:
        p = _payload_from_irsa_hdul(hdul)
    assert p.psf_kind == "effective" and p.psf_oversamp == 5
    assert p.psf_cube.shape == (441, 33, 33) and p.psf_cube.dtype == np.float64
    assert p.psf_table is not None and len(p.psf_table) == 441
    assert abs(p.psf_cube[0].sum() - 1.0) < 1e-3
    assert p.pixel_origin == (932, 934) and p.detector == 4
    assert epsf_source_file(p.psf_header) == SOURCE_FILE


def test_library_from_hdu_and_shared_payload():
    hdu = _epsf_table()
    lib = epsf_library_from_hdu(hdu)
    assert isinstance(lib, EpsfLibrary) and lib.source_file == SOURCE_FILE
    assert lib.cube.shape == (441, 33, 33) and not lib.cube.flags.writeable
    with fits.open(io.BytesIO(_mef_bytes(hdu))) as hdul:
        ref = _payload_from_irsa_hdul(hdul)
        light = _payload_from_irsa_hdul(hdul, psf_product=lib, psf_header=hdul["EPSF"].header.copy())
    np.testing.assert_array_equal(light.psf_cube, ref.psf_cube)
    assert light.psf_kind == "effective" and light.psf_oversamp == 5
    assert light.psf_table is lib.table


# --------------------------------------------------------------------------- #
# registry: provenance check on the light payload
# --------------------------------------------------------------------------- #
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


def _serve(monkeypatch, tmp_path, body):
    log = {"sent": 0, "requests": 0}

    def _get(url, **kw):
        log["requests"] += 1
        log["sent"] = 0
        return _FakeResponse(body, log)

    monkeypatch.setattr(sio.requests, "get", _get)
    monkeypatch.setenv("SPHEREX_RETRIEVAL_CACHE", str(tmp_path / "cache"))
    return log


URL = "https://irsa/level2_2026W32_1A_0001_1D4_spx_l2b-v27-2026-223.fits"


def _registry(lib, **kw):
    return SharedPsfRegistry(kind="effective", data_release="qr3",
                             loader=lambda det: (lib, f"epsf_D{det}_spx_cal-epsf-v1-2026-191.fits"), **kw)


def test_light_epsf_payload_matches_full_and_hangs_up(monkeypatch, tmp_path):
    hdu = _epsf_table()
    body = _mef_bytes(hdu)
    log = _serve(monkeypatch, tmp_path, body)
    lib = epsf_library_from_hdu(hdu)
    with fits.open(io.BytesIO(body)) as hdul:
        ref = _payload_from_irsa_hdul(hdul)
    reg = _registry(lib, verify_every=0)
    coord = SkyCoord(10.0, 20.0, unit="deg")
    first = fetch_irsa_cutout(URL, coord, 1 * u.arcmin, psf_registry=reg)
    assert first.psf_source == "l2" and first.psf_kind == "effective"
    second = fetch_irsa_cutout(URL, SkyCoord(10.1, 20.0, unit="deg"), 1 * u.arcmin, psf_registry=reg)
    assert second.psf_source == "epsf:epsf_D4_spx_cal-epsf-v1-2026-191.fits"
    assert log["sent"] < _epsf_data_offset(body) + (1 << 14) < len(body)
    for name in ("image", "flags", "variance", "zodi", "psf_cube"):
        np.testing.assert_array_equal(getattr(second, name), getattr(ref, name))
    assert second.psf_kind == "effective" and second.psf_oversamp == 5
    assert second.psf_header.tostring() == ref.psf_header.tostring()


def test_provenance_mismatch_falls_back_to_full_download(monkeypatch, tmp_path):
    hdu = _epsf_table()
    body = _mef_bytes(hdu)
    _serve(monkeypatch, tmp_path, body)
    # a library with a different source file but the same arrays: the first
    # full-download comparison passes, the provenance check must still catch it
    other = _epsf_table()
    other.header["HISTORY"] = "Calibration source file: epsf_4_20261231.fits"
    lib = epsf_library_from_hdu(other)
    assert lib.source_file == "epsf_4_20261231.fits"
    reg = _registry(lib, verify_every=0)
    fetch_irsa_cutout(URL, SkyCoord(10.0, 20.0, unit="deg"), 1 * u.arcmin, psf_registry=reg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = fetch_irsa_cutout(URL, SkyCoord(10.2, 20.0, unit="deg"), 1 * u.arcmin, psf_registry=reg)
    assert p.psf_source == "l2" and any("names" in str(x.message) for x in w)
    assert reg.is_disabled(4)


def test_unverified_registry_attaches_the_library_to_any_image(monkeypatch, tmp_path):
    """psf_source='epsf-cal': the L2 file need not carry the product."""
    from tests.test_shared_psf import _mef_bytes as _qr2_mef_bytes
    body = _qr2_mef_bytes()                      # a QR2 cutout with the optical cube
    _serve(monkeypatch, tmp_path, body)
    lib = epsf_library_from_hdu(_epsf_table())
    reg = _registry(lib, verify=False)
    p = fetch_irsa_cutout(URL, SkyCoord(10.0, 20.0, unit="deg"), 1 * u.arcmin, psf_registry=reg)
    assert p.psf_source == "epsf:epsf_D4_spx_cal-epsf-v1-2026-191.fits"
    assert p.psf_kind == "effective" and p.psf_cube.shape == (441, 33, 33)
    assert p.psf_oversamp == 5
    assert epsf_source_file(p.psf_header) == SOURCE_FILE   # the library's header stands in


# --------------------------------------------------------------------------- #
# zones from the table, any lattice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nx,ny", [(21, 21), (11, 41)])
def test_zone_lattice_from_epsf_table(nx, ny):
    tab = zone_table_from_epsf(np.array(_epsf_table(nx, ny).data))
    ix, iy = zone_lattice(tab)
    assert ix.max() == nx and iy.max() == ny and len(tab) == nx * ny
    assert set(tab.colnames) >= {"zone_id", "x", "y", "xwidth", "ywidth", "nstar", "neff"}
    # the first row is the lower-left zone, x-fast
    assert (ix[0], iy[0]) == (1, 1) and (ix[1], iy[1]) == (2, 1)


def test_subset_uses_the_finer_lattice():
    hdu = _epsf_table(21, 21)
    table = np.array(hdu.data)
    cube = np.asarray(table["EPSF"], dtype=np.float64).reshape(-1, 33, 33)
    ztab = zone_table_from_epsf(table)
    # a 175-px cutout at detector (932, 934) spans ~2 zones of 97 px; with one
    # margin ring the subset is 4x4 zones
    sub = subset_zones_for_cutout(cube, hdu.header, cutout_shape=(175, 175),
                                  pixel_origin=(932, 934), zone_margin=1, zone_table=ztab)
    assert sub.cube.shape[1:] == (33, 33) and sub.cube.dtype == np.float64
    assert 9 <= len(sub.lookup) <= 25
    assert "neff" in sub.lookup.colnames and "xwidth" in sub.lookup.colnames
    xs = np.unique(sub.zone_grid_xy[:, 0])
    assert xs.min() >= 9 and xs.max() <= 13         # zones around x ~ 932..1107
    sub0 = subset_zones_for_cutout(cube, hdu.header, cutout_shape=(175, 175),
                                   pixel_origin=(932, 934), zone_margin=0, zone_table=ztab)
    assert len(sub0.lookup) < len(sub.lookup)


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def test_bundle_records_the_epsf_kind(tmp_path):
    hdu = _epsf_table()
    with fits.open(io.BytesIO(_mef_bytes(hdu))) as hdul:
        payload = _payload_from_irsa_hdul(hdul)
    payload.psf_source = "epsf:epsf_D4_spx_cal-epsf-v1-2026-191.fits"
    ztab = zone_table_from_epsf(payload.psf_table)
    subset = subset_zones_for_cutout(payload.psf_cube, payload.psf_header, cutout_shape=(15, 3),
                                     pixel_origin=payload.pixel_origin, zone_margin=1, zone_table=ztab)
    b = Bundle(obs_id="2026W32_1A_0001_1", detector=4, collection="spherex_qr3", access_url="u",
               cloud_uri="", time_bounds_lower=0.0, coord_ra=0.0, coord_dec=0.0,
               cutout=payload, psf_subset=subset)
    out = write_bundle(b, tmp_path / "c.fits")
    with fits.open(out) as hdul:
        h = hdul[0].header
        assert h["PSFKIND"] == "EPSF" and h["OVERSAMP"] == 5 and h["PSFNORM"] == "hr-sum-1"
        assert h["EPSFCAL"] == SOURCE_FILE and h["DETCOORD"] == "sky"
        assert (h["ZONENX"], h["ZONENY"]) == (21, 21)
        assert h["PSFSRC"] == payload.psf_source and h["VERSION"] == "7.0.5"
        psf = hdul["PSF"]
        assert psf.data.dtype == np.dtype(">f8") and psf.data.shape[1:] == (33, 33)
        assert psf.header["OVERSAMP"] == 5 and psf.header["DETCOORD"] == "sky"
        assert "TFIELDS" not in psf.header and "XTENSION" in psf.header
        zones = Table(hdul["PSF_ZONES"].data)
        assert {"zone_id", "x", "y", "plane_idx", "xwidth", "neff"} <= set(zones.colnames)
        assert len(zones) == psf.data.shape[0]


def test_qr2_bundle_keywords_say_optical(tmp_path):
    from tests.test_shared_psf import CUBE
    from tests.test_shared_psf import _mef_bytes as _qr2_mef_bytes
    with fits.open(io.BytesIO(_qr2_mef_bytes())) as hdul:
        payload = _payload_from_irsa_hdul(hdul)
    assert payload.psf_kind == "optical" and payload.psf_table is None
    b = Bundle(obs_id="o", detector=4, collection="spherex_qr2", access_url="u", cloud_uri="",
               time_bounds_lower=0.0, coord_ra=0.0, coord_dec=0.0, cutout=payload)
    out = write_bundle(b, tmp_path / "q.fits")
    with fits.open(out) as hdul:
        h = hdul[0].header
        assert h["PSFKIND"] == "OPTICAL" and h["OVERSAMP"] == 10
        assert "EPSFCAL" not in h and "ZONENX" not in h
        np.testing.assert_array_equal(hdul["PSF"].data, CUBE)
