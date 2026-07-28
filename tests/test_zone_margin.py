"""zone_margin: keep the neighbouring PSF zones, not just the spanned ones.

Without a margin ``subset_zones_for_cutout`` keeps only the zone-index
rectangle between the zones nearest the cutout's two corners. For a cutout
smaller than the ~185 detector px zone pitch both corners resolve to the same
zone, so a single plane is stored — enough to pick a nearest-zone PSF, not
enough for downstream photometry to interpolate between zones.
"""
import numpy as np
import pytest
from astropy.io import fits

from spherex_retrieval.psf import (ZONE_X_INDEX, ZONE_Y_INDEX,
                                   subset_zones_for_cutout)

PITCH = 2048.0 / 11.0


def _header():
    """An 11x11 zone grid tagged the way the L2 PSF HDU tags it."""
    h = fits.Header()
    for i, (gx, gy) in enumerate(zip(ZONE_X_INDEX, ZONE_Y_INDEX), start=1):
        h[f"XCTR_{i}"] = (gx - 0.5) * PITCH
        h[f"YCTR_{i}"] = (gy - 0.5) * PITCH
    return h


@pytest.fixture
def cube():
    return np.arange(121 * 4 * 4, dtype=np.float32).reshape(121, 4, 4)


def test_small_cutout_gains_a_zone_ring(cube):
    """A 100 px cutout mid-detector: 1 plane without a margin, 9 with one."""
    hdr = _header()
    kw = dict(cutout_shape=(100, 100), pixel_origin=(1000, 1000))
    assert len(subset_zones_for_cutout(cube, hdr, zone_margin=0, **kw).lookup) == 1
    assert len(subset_zones_for_cutout(cube, hdr, zone_margin=1, **kw).lookup) == 9
    assert len(subset_zones_for_cutout(cube, hdr, zone_margin=2, **kw).lookup) == 25


def test_margin_is_the_default(cube):
    hdr = _header()
    kw = dict(cutout_shape=(100, 100), pixel_origin=(1000, 1000))
    assert (len(subset_zones_for_cutout(cube, hdr, **kw).lookup)
            == len(subset_zones_for_cutout(cube, hdr, zone_margin=1, **kw).lookup))


def test_margin_clips_at_the_lattice_edge(cube):
    """A corner cutout cannot grow past the 11x11 grid."""
    hdr = _header()
    sub = subset_zones_for_cutout(cube, hdr, cutout_shape=(100, 100),
                                  pixel_origin=(0, 0), zone_margin=1)
    assert len(sub.lookup) == 4                       # 2x2, not 3x3
    assert sub.zone_grid_xy[:, 0].min() >= ZONE_X_INDEX.min()
    assert sub.zone_grid_xy[:, 1].min() >= ZONE_Y_INDEX.min()


def test_margin_zero_reproduces_the_old_subset(cube):
    """Regression guard for bundles written before the default changed."""
    hdr = _header()
    kw = dict(cutout_shape=(600, 600), pixel_origin=(700, 700))
    old = subset_zones_for_cutout(cube, hdr, zone_margin=0, **kw)
    new = subset_zones_for_cutout(cube, hdr, zone_margin=1, **kw)
    assert set(np.asarray(old.lookup["zone_id"])) <= set(np.asarray(new.lookup["zone_id"]))
    assert len(new.lookup) > len(old.lookup)
    # planes are carried through unchanged, just more of them
    ids = list(np.asarray(old.lookup["zone_id"]))
    keep = [i for i, z in enumerate(np.asarray(new.lookup["zone_id"])) if z in ids]
    assert np.array_equal(new.cube[keep], old.cube)
