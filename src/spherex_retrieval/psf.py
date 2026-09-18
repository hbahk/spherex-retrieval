"""PSF zone bookkeeping for SPHEREx cutouts.

A QR2 L2 MEF stores 121 oversampled optical PSFs in an 11x11 detector grid,
each plane tagged in the PSF HDU header with ``XCTR_i``/``YCTR_i`` (0-based
detector pixel coordinates of the zone center, even though ``i`` itself is
1-based).  An R7 (QR3/DR1) MEF stores the effective PSFs as the ``EPSF``
binary table instead, one row per zone of a 21x21 lattice (11x41 on D3)
with ``BINX``/``BINY``/``XCENTER``/``YCENTER``/``XWIDTH``/``YWIDTH``; the
lattice is read from the table, never assumed.  This module provides:

* :func:`build_zone_table` — turn the QR2 header into a tidy table;
  :func:`zone_table_from_epsf` does the same for the R7 table.
* :func:`zone_lattice` — 1-based (ix, iy) lattice indices of every zone,
  inferred from the distinct centre coordinates of either table.
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

import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from packaging.version import Version

PSF_VERSION_FIXED = Version("6.5.6")
PSF_FIX_TAG = "psffix1"

# The QR2 lattice (x-fast plane order); kept for callers that import them.
ZONE_GRID_X, ZONE_GRID_Y = np.meshgrid(np.arange(11), np.arange(11))
ZONE_X_INDEX = ZONE_GRID_X.flatten() + 1   # 1..11
ZONE_Y_INDEX = ZONE_GRID_Y.flatten() + 1

#: Optional per-zone metadata columns carried from the R7 table into the subset lookup.
ZONE_EXTRA_COLUMNS = ("xwidth", "ywidth", "nstar", "neff")


@dataclass
class PSFZoneSubset:
    cube: np.ndarray            # (n_zones, S, S): 101x101 (QR2) or 33x33 (R7)
    lookup: Table               # zone_id, x, y, plane_idx (+ xwidth, ywidth, nstar, neff for R7)
    zone_grid_xy: np.ndarray    # (n_zones, 2) integer 1-based lattice indices


def zone_table_from_epsf(psf_table: np.ndarray) -> Table:
    """``zone_id`` (1-based row number), ``x``, ``y`` (0-based detector pixels)
    and the per-zone metadata of an R7 ``EPSF`` table."""
    n = len(psf_table)
    cols = {
        "zone_id": np.arange(1, n + 1, dtype=np.int32),
        "x": np.asarray(psf_table["XCENTER"], dtype=np.float64),
        "y": np.asarray(psf_table["YCENTER"], dtype=np.float64),
    }
    for col, src in (("xwidth", "XWIDTH"), ("ywidth", "YWIDTH"), ("nstar", "NSTAR"),
                     ("neff", "NEFF_MEAN")):
        if src in psf_table.dtype.names:
            cols[col] = np.asarray(psf_table[src])
    return Table(cols)


def zone_lattice(table: Table) -> tuple[np.ndarray, np.ndarray]:
    """1-based lattice indices ``(ix, iy)`` of every zone row, from the ranks of
    its centre among the distinct centre coordinates (11x11 for QR2, 21x21 or
    11x41 for R7)."""
    x = np.asarray(table["x"], dtype=np.float64)
    y = np.asarray(table["y"], dtype=np.float64)
    ux, uy = np.unique(np.round(x, 3)), np.unique(np.round(y, 3))
    ix = np.searchsorted(ux, np.round(x, 3)) + 1
    iy = np.searchsorted(uy, np.round(y, 3)) + 1
    return ix.astype(np.int64), iy.astype(np.int64)


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


ZONE_MARGIN_DEFAULT = 1


def subset_zones_for_cutout(
    psf_cube: np.ndarray,
    psf_header: fits.Header,
    *,
    cutout_shape: tuple[int, int],     # (ny, nx)
    pixel_origin: tuple[int, int],     # (xlo, ylo) in 0-based detector pixels
    zone_margin: int = ZONE_MARGIN_DEFAULT,
    zone_table: Table | None = None,
) -> PSFZoneSubset:
    """Slice the PSF cube down to zones overlapping the cutout bbox.

    The zone table comes from ``zone_table`` when given (the R7 ``EPSF``
    rows, see :func:`zone_table_from_epsf`) and from the QR2 PSF header
    otherwise; the lattice is inferred from it (:func:`zone_lattice`).

    ``zone_margin`` widens the retained rectangle by that many zones on every
    side (clipped to the lattice). Without it the rectangle spans only
    the zones nearest the cutout's two corners, so a cutout smaller than the
    zone pitch (~185 detector px on QR2, ~97 px on R7) keeps a SINGLE plane —
    enough to pick a nearest-zone PSF, but not enough to interpolate between
    zones, which downstream forced photometry wants (a tile can otherwise sit
    ~93 px from the kernel it uses). One margin ring takes a small cutout from
    1 plane to up to 9; each QR2 plane is 101x101 float32 = 41 kB, an R7 one
    33x33 float64 = 9 kB. Pass ``zone_margin=0`` to reproduce bundles written
    before this default changed.
    """
    table = zone_table if zone_table is not None else build_zone_table(psf_header)
    zone_ix, zone_iy = zone_lattice(table)
    table_index = {int(z): i for i, z in enumerate(table["zone_id"])}

    ny, nx = cutout_shape
    xlo, ylo = pixel_origin
    xhi = xlo + nx
    yhi = ylo + ny

    zid_ll = nearest_zone(xlo, ylo, table)
    zid_ur = nearest_zone(xhi, yhi, table)
    r_ll, r_ur = table_index[zid_ll], table_index[zid_ur]
    zx_ll, zy_ll = zone_ix[r_ll], zone_iy[r_ll]
    zx_ur, zy_ur = zone_ix[r_ur], zone_iy[r_ur]
    if zx_ur < zx_ll:
        zx_ll, zx_ur = zx_ur, zx_ll
    if zy_ur < zy_ll:
        zy_ll, zy_ur = zy_ur, zy_ll

    m = max(int(zone_margin), 0)
    zx_ll = max(int(zx_ll) - m, int(zone_ix.min()))
    zx_ur = min(int(zx_ur) + m, int(zone_ix.max()))
    zy_ll = max(int(zy_ll) - m, int(zone_iy.min()))
    zy_ur = min(int(zy_ur) + m, int(zone_iy.max()))

    sel = (
        (zone_ix >= zx_ll)
        & (zone_ix <= zx_ur)
        & (zone_iy >= zy_ll)
        & (zone_iy <= zy_ur)
    )
    plane_idx = np.where(sel)[0]
    if plane_idx.size == 0:
        plane_idx = np.array([table_index[nearest_zone((xlo + xhi) / 2, (ylo + yhi) / 2, table)]])

    dtype = np.float64 if psf_cube.dtype == np.float64 else np.float32
    cube = np.asarray(psf_cube[plane_idx, :, :], dtype=dtype)
    cols = {
        "zone_id": table["zone_id"][plane_idx],
        "x": table["x"][plane_idx],
        "y": table["y"][plane_idx],
        "plane_idx": np.arange(plane_idx.size, dtype=np.int32),
    }
    for col in ZONE_EXTRA_COLUMNS:
        if col in table.colnames:
            cols[col] = table[col][plane_idx]
    lookup = Table(cols)
    zone_grid_xy = np.column_stack([zone_ix[plane_idx], zone_iy[plane_idx]])
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
        Side length of the output PSF in native pixels.  Defaults to the
        smallest ODD size covering the input, ``11`` for the QR-2 default
        (101 oversampled px = 10.1 native px).  The size must be odd for the
        result to be centred at all: the PSF centre lands on output index
        ``output_size // 2``, and only for odd sizes is that the array's
        geometric centre.  An even size is accepted but warns, because a
        caller using the usual "array centre = source position" convention
        would then be off by half a native pixel.
    normalize : bool
        If True (default), rescale to sum to 1.
    """
    arr = np.asarray(psf_oversamp, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square 2D PSF, got shape {arr.shape}")
    n_over = arr.shape[0]
    if output_size is None:
        output_size = n_over // oversamp
        if output_size % 2 == 0:
            output_size += 1
    elif output_size % 2 == 0:
        warnings.warn(
            f"output_size={output_size} is even, so the PSF centre lands on "
            f"index {output_size // 2} while the array's geometric centre is "
            f"{(output_size - 1) / 2} — half a native pixel apart. Pass an "
            "odd output_size unless you are tracking that offset yourself.",
            stacklevel=2,
        )

    # Apply a sub-pixel shift in oversampled units.
    dx_over = sub_pixel_shift[0] * oversamp
    dy_over = sub_pixel_shift[1] * oversamp
    shifted = _shift_image(arr, dx=dx_over, dy=dy_over)

    # Pixel-integrate onto a grid CENTRED on the array centre (see
    # _integrate_centred).  The previous implementation reshaped into
    # ``output_size x oversamp`` blocks aligned to index 0, which forced two
    # asymmetries whenever ``output_size * oversamp != n_over``: it cropped
    # with ``start = (n_over - needed) // 2`` -- for the QR-2 default
    # 101 -> 100 that drops only the LAST row and column, not a centred crop --
    # and it padded with ``((0, pad), (0, pad))``, i.e. entirely on the
    # bottom/right.  Either one displaces the PSF relative to the output array
    # centre by a fraction of an oversampled pixel, which a forced-photometry
    # caller then reads as an astrometric offset.
    out = _integrate_centred(shifted, oversamp=oversamp,
                             output_size=output_size)

    if normalize:
        s = out.sum()
        if s > 0:
            out = out / s
    return out


def _integrate_centred(
    arr: np.ndarray, *, oversamp: int, output_size: int
) -> np.ndarray:
    """Pixel-integrate an oversampled image onto a grid centred on its centre.

    Output pixel ``m`` integrates the continuous window of width ``oversamp``
    centred on input index ``c + oversamp * (m - output_size // 2)``, where
    ``c = (n - 1) / 2`` is the input's geometric centre.  So:

    * the output grid is uniform, spacing exactly ``oversamp`` input px;
    * output index ``output_size // 2`` is centred on the input centre, for any
      combination of input size, ``oversamp`` and ``output_size`` parities;
    * flux is conserved up to what falls outside the array (zero-padded).

    Parity is handled by weights rather than by cropping.  A width-``oversamp``
    window centred on a pixel centre covers whole pixels when ``oversamp`` is
    odd, and covers two half-pixels at its ends when ``oversamp`` is even (for
    ``oversamp=10``: weights ``0.5, 1 x 9, 0.5``, summing to 10).  Reshaping
    into aligned blocks, as the previous implementation did, cannot express
    that half-pixel and so had to crop or pad asymmetrically instead.
    """
    a = np.asarray(arr, dtype=np.float64)
    n = a.shape[0]
    half = oversamp / 2.0
    # weight of input pixel j (covering [j-0.5, j+0.5]) inside a window of
    # width `oversamp` centred at 0 -> overlap length, computed once
    off = np.arange(-int(np.ceil(half)), int(np.ceil(half)) + 1)
    w = np.clip(np.minimum(off + 0.5, half) - np.maximum(off - 0.5, -half),
                0.0, None)
    c = (n - 1) / 2.0
    centres = c + oversamp * (np.arange(output_size) - output_size // 2)
    # nearest input index to each window centre, plus the fractional remainder
    base = np.rint(centres).astype(int)
    frac = centres - base
    if np.any(np.abs(frac) > 1e-9):
        # window centres land between input pixels: interpolate the weights
        idx = base[:, None] + off[None, :]
        wgt = np.empty((output_size, off.size), dtype=np.float64)
        for m in range(output_size):
            d = off - frac[m]
            wgt[m] = np.clip(np.minimum(d + 0.5, half)
                             - np.maximum(d - 0.5, -half), 0.0, None)
    else:
        idx = base[:, None] + off[None, :]
        wgt = np.broadcast_to(w, (output_size, off.size))
    ok = (idx >= 0) & (idx < n)
    safe = np.where(ok, idx, 0)
    # separable: apply along axis 0 then axis 1
    rows = np.einsum("mk,mkj->mj", wgt * ok, a[safe, :])
    out = np.einsum("nk,mnk->mn", wgt * ok, rows[:, safe])
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
