"""find_overlapping_many: cone + polygon candidates for many targets at once.

Frames are exact squares in the gnomonic plane of their centre (the plane in
which great-circle edges are straight). The corner order is deliberately
inconsistent — the production ``s_region`` winding is — and frames sit on the
RA=0 seam and next to the pole.
"""
import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("scipy")

from spherex_retrieval import index as sidx  # noqa: E402

H = 1.75  # half side of a frame in the tangent plane, deg (a SPHEREx frame is ~3.5 deg)


def _deproject(ra0, dec0, xi_deg, eta_deg):
    """Inverse gnomonic projection about (ra0, dec0)."""
    xi, eta = np.radians(xi_deg), np.radians(eta_deg)
    c = sidx._unit_vectors(ra0, dec0)
    e, n = sidx._tangent_basis(ra0, dec0)
    v = c + xi[..., None] * e + eta[..., None] * n
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0, np.degrees(np.arcsin(v[..., 2]))


def _frame(ra0, dec0, order=(0, 1, 2, 3)):
    square = np.tan(np.radians(np.array([[-H, -H], [H, -H], [H, H], [-H, H]])))
    ra, dec = _deproject(ra0, dec0, np.degrees(square[:, 0]), np.degrees(square[:, 1]))
    corners = np.stack([ra, dec], axis=1)[list(order)]
    return {"s_ra": ra0, "s_dec": dec0, **{f"c{i + 1}_{ax}": corners[i, j]
                                          for i in range(4) for j, ax in enumerate(("ra", "dec"))}}


def _index(frames):
    cols = {k: [f[k] for f in frames] for k in frames[0]}
    return pa.table({k: pa.array(v, type=pa.float64()) for k, v in cols.items()})


def _pairs(result):
    return sorted((int(t), int(f)) for t, f in zip(result["target"], result["frame"]))


def _offset(ra0, dec0, xi_deg, eta_deg):
    ra, dec = _deproject(ra0, dec0, np.atleast_1d(np.degrees(np.tan(np.radians(xi_deg)))),
                         np.atleast_1d(np.degrees(np.tan(np.radians(eta_deg)))))
    return float(ra[0]), float(dec[0])


def test_winding_and_vertex_order_do_not_matter():
    idx = _index([_frame(10, 0), _frame(10, 0, (3, 2, 1, 0)), _frame(10, 0, (0, 2, 1, 3))])
    inside = _offset(10, 0, 1.0, -1.2)
    outside = _offset(10, 0, 2.2, 0.0)
    res = sidx.find_overlapping_many([inside[0], outside[0]], [inside[1], outside[1]], 0.0, idx,
                                     margin_pix=0.0)
    assert _pairs(res) == [(0, 0), (0, 1), (0, 2)]


def test_margin_is_half_the_box_diagonal_plus_pixels():
    idx = _index([_frame(200, -30)])
    size = 60.0
    margin = size / np.sqrt(2) + 10 * sidx.PIXEL_SCALE_ARCSEC      # ~104 arcsec
    near = _offset(200, -30, H + 0.8 * margin / 3600, 0.0)
    far = _offset(200, -30, H + 1.3 * margin / 3600, 0.0)
    res = sidx.find_overlapping_many([near[0], far[0]], [near[1], far[1]], size, idx, margin_pix=10)
    assert _pairs(res) == [(0, 0)]


def test_ra_seam_and_pole():
    idx = _index([_frame(0.5, 0.0), _frame(45.0, 89.0), _frame(180.0, 0.0)])
    # (225, 89.5) is 1.5 deg from the (45, 89) centre across the pole: inside;
    # (45, 87) is 2 deg south of it: outside
    t = [(359.2, 0.3), (1.9, -1.0), (225.0, 89.5), (45.0, 87.0), (0.5, 3.0)]
    res = sidx.find_overlapping_many([a for a, _ in t], [b for _, b in t], 0.0, idx, margin_pix=0)
    assert _pairs(res) == [(0, 0), (1, 0), (2, 1)]


def test_result_is_file_major():
    idx = _index([_frame(10, 0), _frame(11, 0.5)])
    ra = [10.2, 10.8, 11.1, 9.9]
    dec = [0.1, 0.2, 0.3, -0.2]
    res = sidx.find_overlapping_many(ra, dec, 0.0, idx, margin_pix=0)
    order = list(zip(res["frame"], res["target"]))
    assert order == sorted(order) and len(order) == 8


def test_agrees_with_the_spherical_great_circle_test():
    rng = np.random.default_rng(3)
    frames = [_frame(ra0, dec0, tuple(rng.permutation(4)) if k % 2 else (0, 1, 2, 3))
              for k, (ra0, dec0) in enumerate(zip(rng.uniform(0, 360, 40), rng.uniform(-80, 80, 40)))]
    idx = _index(frames)
    # targets scattered around the frames
    k = rng.integers(0, 40, 3000)
    ra = np.array([frames[j]["s_ra"] for j in k]) + rng.normal(0, 2.5, k.size) / np.cos(
        np.radians(np.array([frames[j]["s_dec"] for j in k])))
    dec = np.clip(np.array([frames[j]["s_dec"] for j in k]) + rng.normal(0, 2.5, k.size), -89, 89)
    res = sidx.find_overlapping_many(ra % 360, dec, 0.0, idx, margin_pix=0)
    got = set(_pairs(res))

    # independent test: inside every great circle through consecutive corners
    p = sidx._unit_vectors(ra % 360, dec)
    want = set()
    for j, f in enumerate(frames):
        corners = np.array([[f[f"c{i}_ra"], f[f"c{i}_dec"]] for i in range(1, 5)])
        v = sidx._unit_vectors(corners[:, 0], corners[:, 1])
        c = sidx._unit_vectors(f["s_ra"], f["s_dec"])
        # counter-clockwise order around the centre, as seen from outside the sphere
        e, n = sidx._tangent_basis(f["s_ra"], f["s_dec"])
        v = v[np.argsort(np.arctan2(v @ n, v @ e))]
        normals = np.cross(v, np.roll(v, -1, axis=0))
        inside = np.all(p @ normals.T >= 0, axis=1) & (p @ c > 0)
        want.update((int(t), j) for t in np.flatnonzero(inside))
    assert len(want) > 500
    assert got == want
