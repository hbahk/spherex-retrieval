"""PSF zone bookkeeping for SPHEREx cutouts.

The L2 MEF stores 121 oversampled PSFs in an 11x11 detector grid.  Each
plane is tagged in the PSF HDU header with ``XCTR_i``/``YCTR_i`` (0-based
detector pixel coordinates of the zone center, even though ``i`` itself is
1-based).  This module provides:

* :func:`build_zone_table` — turn the header into a tidy table.
* :func:`fix_psf_header_if_needed` — apply the QR-2 PSF erratum rewrite
  for spectral images with ``VERSION <= 6.5.5`` (no ``+psffix1`` local
  tag).  See https://irsa.ipac.caltech.edu/data/SPHEREx/docs/psfhdrerr.html
* :func:`subset_zones_for_cutout` — pick the zones overlapping a cutout
  bounding box, return the cropped cube + the matching lookup.
* :func:`select_zone_for_source` — pick the best PSF plane for a source
  given its position (in either cutout or original detector pixels).
* :func:`resample_psf_to_native` — downsample an oversampled PSF onto the
  native detector grid for use with forward-modelling tools (Tractor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.table import Table
from packaging.version import Version

PSF_VERSION_FIXED = Version("6.5.6")
PSF_FIX_TAG = "psffix1"

ZONE_GRID_X, ZONE_GRID_Y = np.meshgrid(np.arange(11), np.arange(11))
ZONE_X_INDEX = ZONE_GRID_X.flatten() + 1   # 1..11
ZONE_Y_INDEX = ZONE_GRID_Y.flatten() + 1


@dataclass
class PSFZoneSubset:
    cube: np.ndarray            # (n_zones, 101, 101)
    lookup: Table               # zone_id, x, y, plane_idx
    zone_grid_xy: np.ndarray    # (n_zones, 2) integer 1..11 indices


def build_zone_table(psf_header: fits.Header) -> Table:
    """Return ``zone_id``, ``x``, ``y`` (0-based detector pixels) for all 121 zones."""
    xctr: dict[int, float] = {}
    yctr: dict[int, float] = {}
    for key, val in psf_header.items():
        if re.match(r"XCTR_\d+", key):
            xctr[int(key.split("_")[1])] = float(val)
        elif re.match(r"YCTR_\d+", key):
            yctr[int(key.split("_")[1])] = float(val)
    if len(xctr) != len(yctr):
        raise ValueError("PSF header has mismatched XCTR/YCTR entries")

    rows = sorted(xctr.keys())
    return Table(
        {
            "zone_id": np.array(rows, dtype=np.int32),
            "x": np.array([xctr[i] for i in rows], dtype=np.float64),
            "y": np.array([yctr[i] for i in rows], dtype=np.float64),
        }
    )


def cutout_to_orig(x_cut: float, y_cut: float, *, crpix1a: float, crpix2a: float) -> tuple[float, float]:
    """Map a 0-based cutout pixel coord to a 0-based original-detector pixel coord."""
    return (1.0 + (x_cut - crpix1a), 1.0 + (y_cut - crpix2a))


def nearest_zone(x_orig: float, y_orig: float, table: Table) -> int:
    """Return the ``zone_id`` (1-based) whose center is closest to (x, y) in original pixels."""
    dx = table["x"] - x_orig
    dy = table["y"] - y_orig
    return int(table["zone_id"][np.argmin(dx * dx + dy * dy)])


def subset_zones_for_cutout(
    psf_cube: np.ndarray,
    psf_header: fits.Header,
    *,
    cutout_shape: tuple[int, int],     # (ny, nx)
    pixel_origin: tuple[int, int],     # (xlo, ylo) in 0-based detector pixels
) -> PSFZoneSubset:
    """Slice the PSF cube down to zones overlapping the cutout bbox."""
    table = build_zone_table(psf_header)

    ny, nx = cutout_shape
    xlo, ylo = pixel_origin
    xhi = xlo + nx
    yhi = ylo + ny

    zid_ll = nearest_zone(xlo, ylo, table)
    zid_ur = nearest_zone(xhi, yhi, table)
    zx_ll, zy_ll = ZONE_X_INDEX[zid_ll - 1], ZONE_Y_INDEX[zid_ll - 1]
    zx_ur, zy_ur = ZONE_X_INDEX[zid_ur - 1], ZONE_Y_INDEX[zid_ur - 1]
    if zx_ur < zx_ll:
        zx_ll, zx_ur = zx_ur, zx_ll
    if zy_ur < zy_ll:
        zy_ll, zy_ur = zy_ur, zy_ll

    sel = (
        (ZONE_X_INDEX >= zx_ll)
        & (ZONE_X_INDEX <= zx_ur)
        & (ZONE_Y_INDEX >= zy_ll)
        & (ZONE_Y_INDEX <= zy_ur)
    )
    plane_idx = np.where(sel)[0]
    if plane_idx.size == 0:
        plane_idx = np.array([nearest_zone((xlo + xhi) / 2, (ylo + yhi) / 2, table) - 1])

    cube = np.asarray(psf_cube[plane_idx, :, :], dtype=np.float32)
    lookup = Table(
        {
            "zone_id": table["zone_id"][plane_idx],
            "x": table["x"][plane_idx],
            "y": table["y"][plane_idx],
            "plane_idx": np.arange(plane_idx.size, dtype=np.int32),
        }
    )
    zone_grid_xy = np.column_stack([ZONE_X_INDEX[plane_idx], ZONE_Y_INDEX[plane_idx]])
    return PSFZoneSubset(cube=cube, lookup=lookup, zone_grid_xy=zone_grid_xy)


def select_zone_for_source(
    subset: PSFZoneSubset,
    *,
    x_orig: float,
    y_orig: float,
) -> int:
    """Return the local plane index in ``subset.cube`` closest to (x_orig, y_orig)."""
    dx = subset.lookup["x"] - x_orig
    dy = subset.lookup["y"] - y_orig
    return int(subset.lookup["plane_idx"][np.argmin(dx * dx + dy * dy)])


# --------------------------------------------------------------------------- #
# PSF erratum fix for VERSION <= 6.5.5
# --------------------------------------------------------------------------- #

_ZONE_COMMENT_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


def psf_fix_applied(primary_header: fits.Header) -> bool:
    """Return True iff the PSF zone-indexing fix is present.

    Per the SPHEREx erratum, files with ``VERSION >= 6.5.6`` are correct
    out of the box, and earlier files that have been reprocessed carry a
    ``+psffix1`` local version tag.
    """
    if "VERSION" not in primary_header:
        return False
    v = Version(str(primary_header["VERSION"]))
    if v >= PSF_VERSION_FIXED:
        return True
    return v.local is not None and PSF_FIX_TAG in v.local


def fix_psf_header_if_needed(
    psf_header: fits.Header,
    primary_header: fits.Header,
    *,
    n_planes: int,
) -> tuple[fits.Header, bool]:
    """Rewrite the per-plane ``XCTR_i``/``YCTR_i``/``XWID_i``/``YWID_i`` mapping.

    Returns ``(header, was_fixed)``.  When the fix is already applied (or
    unnecessary) the header is returned unchanged with ``was_fixed=False``.

    Plane k0 (0-based) becomes x-fast ordered::

        ix = k0 % bins_x
        iy = k0 // bins_x
    """
    if psf_fix_applied(primary_header):
        return psf_header, False

    bins_x, bins_y = _infer_bins_from_comments(psf_header, n_planes)
    x_centers, y_centers, x_widths, y_widths = _collect_axis_values(psf_header, n_planes)

    out = psf_header.copy()
    for k0 in range(n_planes):
        ix = k0 % bins_x
        iy = k0 // bins_x
        k1 = k0 + 1
        out[f"XCTR_{k1}"] = (x_centers[ix], f"Center of x zone ({ix}, {iy})")
        out[f"YCTR_{k1}"] = (y_centers[iy], f"Center of y zone ({ix}, {iy})")
        if ix in x_widths:
            out[f"XWID_{k1}"] = (x_widths[ix], f"Width of x zone ({ix}, {iy})")
        if iy in y_widths:
            out[f"YWID_{k1}"] = (y_widths[iy], f"Width of y zone ({ix}, {iy})")
    out["HISTORY"] = "Rewrote PSF per-plane zone metadata to x-fast ordering (psffix1)."
    return out, True


def _parse_zone_comment(comment: str) -> tuple[int, int]:
    m = _ZONE_COMMENT_RE.search(str(comment))
    if not m:
        raise ValueError(f"could not parse zone indices from comment: {comment!r}")
    return int(m.group(1)), int(m.group(2))


def _infer_bins_from_comments(hdr: fits.Header, nzone: int) -> tuple[int, int]:
    max_ix = max_iy = -1
    for k1 in range(1, nzone + 1):
        key = f"XCTR_{k1}"
        if key not in hdr:
            raise KeyError(f"missing required PSF header key: {key}")
        ix, iy = _parse_zone_comment(hdr.comments[key])
        max_ix = max(max_ix, ix)
        max_iy = max(max_iy, iy)
    bins_x, bins_y = max_ix + 1, max_iy + 1
    if bins_x * bins_y != nzone:
        raise ValueError(
            f"inconsistent grid inferred from comments: bins_x={bins_x}, "
            f"bins_y={bins_y}, nzone={nzone}"
        )
    return bins_x, bins_y


def _collect_axis_values(
    hdr: fits.Header, nzone: int
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    x_centers: dict[int, float] = {}
    y_centers: dict[int, float] = {}
    x_widths: dict[int, float] = {}
    y_widths: dict[int, float] = {}
    for k1 in range(1, nzone + 1):
        ix, iy = _parse_zone_comment(hdr.comments[f"XCTR_{k1}"])
        if f"XCTR_{k1}" in hdr:
            x_centers[ix] = float(hdr[f"XCTR_{k1}"])
        if f"YCTR_{k1}" in hdr:
            y_centers[iy] = float(hdr[f"YCTR_{k1}"])
        if f"XWID_{k1}" in hdr:
            x_widths[ix] = float(hdr[f"XWID_{k1}"])
        if f"YWID_{k1}" in hdr:
            y_widths[iy] = float(hdr[f"YWID_{k1}"])
    return x_centers, y_centers, x_widths, y_widths


# --------------------------------------------------------------------------- #
# Oversampled-PSF -> native-grid resampler (for forward-modelling)
# --------------------------------------------------------------------------- #

def resample_psf_to_native(
    psf_oversamp: np.ndarray,
    *,
    oversamp: int = 10,
    sub_pixel_shift: tuple[float, float] = (0.0, 0.0),
    output_size: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Downsample an oversampled SPHEREx PSF onto the native detector grid.

    The PSF cube delivered in the L2 MEF is super-resolved by
    ``OVERSAMP`` (=10 for QR-2): 10 PSF pixels span one native detector
    pixel.  Forward-modelling tools (Tractor) need the PSF
    pixel-integrated at the native resolution and evaluated at the source
    sub-pixel phase.

    Parameters
    ----------
    psf_oversamp : ndarray (N x N)
        Single oversampled PSF plane (e.g. 101 x 101).
    oversamp : int
        Oversampling factor.  Defaults to 10.
    sub_pixel_shift : (dx, dy)
        Source's sub-pixel offset *in native pixels* (0..1 each).
    output_size : int, optional
        Side length of the output PSF in native pixels.  Defaults to
        ``floor(N / oversamp)`` (10 for the QR-2 default).
    normalize : bool
        If True (default), rescale to sum to 1.
    """
    arr = np.asarray(psf_oversamp, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square 2D PSF, got shape {arr.shape}")
    n_over = arr.shape[0]
    if output_size is None:
        output_size = n_over // oversamp

    # Apply a sub-pixel shift in oversampled units.
    dx_over = sub_pixel_shift[0] * oversamp
    dy_over = sub_pixel_shift[1] * oversamp
    shifted = _shift_image(arr, dx=dx_over, dy=dy_over)

    # Block-integrate down to native pixels.
    needed = output_size * oversamp
    if needed > n_over:
        pad = needed - n_over
        shifted = np.pad(shifted, ((0, pad), (0, pad)), mode="constant")
    elif needed < n_over:
        # Crop centred to the integer block.
        start = (n_over - needed) // 2
        shifted = shifted[start:start + needed, start:start + needed]
    out = shifted.reshape(
        output_size, oversamp, output_size, oversamp
    ).sum(axis=(1, 3))

    if normalize:
        s = out.sum()
        if s > 0:
            out = out / s
    return out


def _shift_image(arr: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    """Bilinear sub-pixel shift; positive dx shifts right, positive dy shifts up."""
    if dx == 0.0 and dy == 0.0:
        return arr
    ny, nx = arr.shape
    yy, xx = np.indices(arr.shape, dtype=np.float64)
    xs = xx - dx
    ys = yy - dy
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < nx) & (y1 < ny)
    out = np.zeros_like(arr)
    x0c = np.clip(x0, 0, nx - 1)
    x1c = np.clip(x1, 0, nx - 1)
    y0c = np.clip(y0, 0, ny - 1)
    y1c = np.clip(y1, 0, ny - 1)
    out[valid] = (
        arr[y0c[valid], x0c[valid]] * (1 - fx[valid]) * (1 - fy[valid])
        + arr[y0c[valid], x1c[valid]] * fx[valid] * (1 - fy[valid])
        + arr[y1c[valid], x0c[valid]] * (1 - fx[valid]) * fy[valid]
        + arr[y1c[valid], x1c[valid]] * fx[valid] * fy[valid]
    )
    return out
