"""resample_psf_to_native must not displace the PSF relative to the output centre.

Regression tests for the 2026-07-30 fix.  The previous implementation reshaped
the oversampled array into ``output_size x oversamp`` blocks aligned to index 0,
which forced an asymmetric crop (``start = (101 - 100) // 2 == 0`` dropped only
the LAST row/column) or an end-only pad (``((0, pad), (0, pad))``).  Either one
moves the PSF a fraction of an oversampled pixel away from the output array
centre -- and a forced-photometry caller, which anchors the kernel by its array
centre, reads that as an astrometric offset.
"""
import numpy as np
import pytest

from spherex_retrieval.psf import resample_psf_to_native

OVERSAMP = 10
N_OVER = 101


def _kernels():
    yy, xx = np.indices((N_OVER, N_OVER), dtype=float)
    c = (N_OVER - 1) / 2.0
    sym = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 9.0 ** 2))
    asym = sym + 0.35 * np.exp(
        -(((xx - c) - 14) ** 2 + ((yy - c) - 9) ** 2) / (2 * 11.0 ** 2))
    return sym, asym


def _centroid(a):
    """Flux centroid minus the array's geometric centre, in the array's px."""
    a = np.asarray(a, dtype=float)
    yy, xx = np.indices(a.shape, dtype=float)
    c = (a.shape[0] - 1) / 2.0
    s = a.sum()
    return (a * (xx - c)).sum() / s, (a * (yy - c)).sum() / s


def test_default_output_size_is_odd():
    """Only an odd size can put a PIXEL on the array's geometric centre.

    101 oversampled px is 10.1 native px; the old default floor(101/10) = 10 is
    even, so the PSF centre landed on index 5 while the geometric centre is 4.5
    — half a native pixel apart, by construction.
    """
    out = resample_psf_to_native(_kernels()[0])
    assert out.shape == (11, 11)
    assert out.shape[0] % 2 == 1


def test_symmetric_input_stays_symmetric_and_centred():
    out = resample_psf_to_native(_kernels()[0])
    np.testing.assert_allclose(out, out[::-1, ::-1], atol=1e-12)
    assert np.unravel_index(np.argmax(out), out.shape) == (5, 5)
    cx, cy = _centroid(out)
    assert abs(cx) < 1e-12 and abs(cy) < 1e-12


def test_asymmetric_centroid_is_preserved():
    """An asymmetric kernel's centroid must survive the resampling.

    This is the property the asymmetric crop/pad broke: it shifted the whole
    kernel, so the offset a caller measured was part resampling artefact.
    """
    _, asym = _kernels()
    want = np.array(_centroid(asym)) / OVERSAMP        # native px
    got = np.array(_centroid(resample_psf_to_native(asym, normalize=False)))
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_flux_is_conserved():
    _, asym = _kernels()
    out = resample_psf_to_native(asym, normalize=False)
    assert out.sum() / asym.sum() == pytest.approx(1.0, abs=1e-9)
    assert resample_psf_to_native(asym).sum() == pytest.approx(1.0, abs=1e-12)


def test_even_output_size_warns():
    with pytest.warns(UserWarning, match="even"):
        resample_psf_to_native(_kernels()[0], output_size=10)


@pytest.mark.parametrize("dx", [0.0, 0.25, -0.4])
def test_sub_pixel_shift_lands_where_requested(dx):
    """A requested shift must show up in the output centroid one-for-one."""
    out = resample_psf_to_native(_kernels()[0], sub_pixel_shift=(dx, 0.0))
    assert _centroid(out)[0] == pytest.approx(dx, abs=1e-4)


def test_odd_oversamp_needs_no_half_pixel_weights():
    """With odd oversamp the window covers whole pixels; check it still centres.

    Build a centred 99x99 kernel rather than slicing the 101x101 one: a
    ``[:99, :99]`` crop is itself off-centre, which would test the crop and not
    the resampler.
    """
    n = 99
    yy, xx = np.indices((n, n), dtype=float)
    c = (n - 1) / 2.0
    sym = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 9.0 ** 2))
    out = resample_psf_to_native(sym, oversamp=9)
    assert out.shape[0] % 2 == 1
    np.testing.assert_allclose(out, out[::-1, ::-1], atol=1e-12)
    cx, cy = _centroid(out)
    assert abs(cx) < 1e-12 and abs(cy) < 1e-12
