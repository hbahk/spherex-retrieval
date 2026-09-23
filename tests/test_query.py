"""Discovery: one row per L2 file, and TAP collections that do not bleed into each other.

IRSA's TAP artifact paths carry the release (``/spherex/qr3/``) but not the
collection, so ``spherex_qr3`` and ``spherex_qr3_deep`` cannot be told apart by
the path. Filtering on the path alone made both collections return every QR3
image, and ``find_overlapping`` stacked the two lists: each QR3 exposure was
retrieved, and photometered, twice (measured on QSO J0233+0653: 479 rows, 364
files). These tests run without the network.
"""
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from spherex_retrieval import query

COORD = SkyCoord(38.34366, 6.89102, unit="deg")
SIZE = 92.25 * u.arcsec
QR3_URL = ("https://irsa.ipac.caltech.edu/ibe/data/spherex/qr3/level2/2026W30_2A/"
           "l2b-v27-2026-224/{det}/level2_2026W30_2A_0166_1D{det}_spx_l2b-v27-2026-224.fits")
QR2_URL = ("https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/level2/2025W30_1B/"
           "l2b-v20-2025-262/{det}/level2_2025W30_1B_0325_1D{det}_spx_l2b-v20-2025-262.fits")


def _canonical(urls, collection, t0=60000.0):
    n = len(urls)
    return Table({
        "access_url": np.asarray(urls, dtype=str),
        "cloud_uri": np.asarray([""] * n, dtype=str),
        "obs_id": np.asarray([query.observation_id_from_filename(u) for u in urls], dtype=str),
        "bandpass": np.asarray([f"SPHEREx-D{i + 1}" for i in range(n)], dtype=str),
        "detector": np.arange(1, n + 1, dtype=np.int32),
        "time_bounds_lower": t0 + np.arange(n, dtype=np.float64),
        "collection": np.asarray([collection] * n, dtype=str),
    })


def test_query_tap_selects_the_collection_not_only_the_release(monkeypatch):
    seen = []

    def fake_adql(adql, *, timeout=120.0):
        seen.append(adql)
        return Table({"access_path": np.array([], dtype=str),
                      "time_bounds_lower": np.array([], dtype=float),
                      "obsid": np.array([], dtype=str),
                      "energy_bandpassname": np.array([], dtype=str),
                      "provenance_version": np.array([], dtype=str)})

    monkeypatch.setattr(query, "_run_adql", fake_adql)
    query.query_tap(COORD, SIZE, collection="spherex_qr3")
    query.query_tap(COORD, SIZE, collection="spherex_qr3_deep")
    wide, deep = seen
    assert "JOIN spherex.observation o ON o.obsid = p.obsid" in wide
    assert "o.collection = 'spherex_qr3'" in wide
    assert "o.collection = 'spherex_qr3_deep'" in deep
    # the release is still pinned by the artifact path
    assert "'%/spherex/qr3/level2/%'" in wide and "'%/spherex/qr3/level2/%'" in deep


def test_find_overlapping_keeps_each_file_once(monkeypatch):
    qr3 = [QR3_URL.format(det=d) for d in (1, 4)]

    def fake_tap(coord, size, *, collection, bandpass=None, timeout=120.0):
        # what the path-only filter did: every QR3 collection returns every QR3 image
        return _canonical(qr3, collection, t0=61000.0)

    def fake_sia2(coord, size, *, collection, bandpass=None, timeout=120.0):
        urls = [QR2_URL.format(det=d) for d in (1, 4)] if collection == "spherex_qr2" else []
        return _canonical(urls, collection) if urls else query._empty_canonical_table()

    monkeypatch.setattr(query, "query_tap", fake_tap)
    monkeypatch.setattr(query, "query_sia2", fake_sia2)
    t = query.find_overlapping(COORD, SIZE)
    names = [u.rsplit("/", 1)[-1] for u in t["access_url"]]
    assert len(t) == 4 and len(set(names)) == 4
    # the first collection in the search order keeps the row
    assert set(t["collection"][np.char.find(t["access_url"].astype(str), "/qr3/") >= 0]) == {"spherex_qr3"}
    assert list(t["time_bounds_lower"]) == sorted(t["time_bounds_lower"])


def test_drop_duplicate_files_ignores_query_strings_and_keeps_order():
    a = QR2_URL.format(det=1)
    b = QR2_URL.format(det=2)
    t = _canonical([a, b, a + "?center=38.3,6.9&size=0.0256deg", b], "spherex_qr2")
    out = query._drop_duplicate_files(t)
    assert list(out["access_url"]) == [a, b]
    untouched = _canonical([a, b], "spherex_qr2")
    assert query._drop_duplicate_files(untouched) is untouched
