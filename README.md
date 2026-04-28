# spherex-retrieval

Cutout retrieval for SPHEREx Spectral Image MEFs, packaged for forced-photometry workflows.
One FITS file per overlapping pointing, carrying everything Tractor (or any forward modeller) needs:
science image, variance, flags, zodi, a per-cutout PSF subset with its zone lookup,
and per-pixel wavelength / bandpass maps drawn from the standalone Spectral WCS calibration product.

- **Contents**
- [Install](#install)
- [Quick start](#quick-start)
- [Output](#output)
- [Configuration](#configuration)
- [Forced-photometry recipe](#forced-photometry-recipe)
- [Side data: SAPM (MJy/sr → µJy)](#side-data-sapm-mjysr--%C2%B5jy)
- [Notes](#notes)
  - [Wavelength source](#wavelength-source)
  - [PSF erratum (VERSION ≤ 6.5.5)](#psf-erratum-version--655)
  - [Status codes](#status-codes)
- [References](#references)

---

## Install

```bash
pip install -e .
```

Cloud / S3 access works out of the box (`fsspec`, `s3fs` are declared deps).

---

## Quick start

Programmatic:

```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from spherex_retrieval import retrieve

coord = SkyCoord(ra=258.2084186*u.deg, dec=64.0529535*u.deg, frame="icrs")
size  = 6.15 * 15 * u.arcsec      # 15 native pixels on a side

bundles, out_dir = retrieve(coord, size)   # default: astroquery + IRSA cutout
print(out_dir)
```

CLI:

```bash
spherex-retrieve --ra 258.2084186 --dec 64.0529535 --size 92.25 --out ./cutouts
spherex-retrieve --ra ... --dec ... --size ... --bandpass SPHEREx-D2          # one detector
spherex-retrieve --ra ... --dec ... --size ... --include-sapm                 # add SAPM HDU
```

---

## Output

```
<out_dir>/
├── summary.ecsv                       # one row per overlap, with status enum
├── cutout_0001_<obsid>_D1.fits
├── cutout_0002_<obsid>_D2.fits
└── ...
```

Each per-cutout MEF:

| HDU | name      | content                                                          |
|-----|-----------|------------------------------------------------------------------|
| 0   | PRIMARY   | provenance — `OBSID`, `DETECTOR`, `RA_REQ`, `DEC_REQ`, `STATUS`, `VERSION`, `PSFFIXED`, `OVERSAMP` |
| 1   | IMAGE     | calibrated surface brightness (MJy/sr), cropped                  |
| 2   | FLAGS     | per-pixel bitmap                                                 |
| 3   | VARIANCE  | (MJy/sr)²                                                        |
| 4   | ZODI      | modeled zodiacal background (MJy/sr)                             |
| 5   | PSF       | 101×101×N_zones cube, restricted to overlapping zones            |
| 6   | PSF_ZONES | lookup: `zone_id`, `x`, `y` (orig 0-based), `plane_idx`          |
| opt | CWAVE     | per-pixel central wavelength (µm) — when `include_wavelength`    |
| opt | CBAND     | per-pixel bandwidth (µm) — when `include_wavelength`             |
| opt | SAPM      | per-pixel solid angle (arcsec²) — when `include_sapm`            |

---

## Configuration

| Step       | Default               | Alternates                                                            |
|------------|-----------------------|-----------------------------------------------------------------------|
| Discovery  | `query_backend="astroquery"` (SIA2) | `"pyvo"` (ADQL TAP — no SIA ingestion lag)              |
| Cutout     | `cutout_backend="irsa"` (server-side cutout) | `"fsspec"` byte-range over HTTP or `s3://`     |
| Wavelength | `include_wavelength=True` (CWAVE/CBAND) | `False` to skip                                     |
| PSF        | `subset_psf=True` (overlapping zones only) | `False` to keep all 121 planes                   |
| Bandpass   | all detectors         | `bandpass="SPHEREx-D2"` (filter applied at query time)                |
| SAPM       | `include_sapm=False`  | `True` to fetch + crop Solid Angle Pixel Map (arcsec²)                |
| Survey     | `("spherex_qr2", "spherex_qr2_deep")` | restrict via `collections=(...)`                      |
| Cache      | `~/.cache/spherex-retrieval` (or `$SPHEREX_RETRIEVAL_CACHE`) | `cache_dir=...`                 |
| Concurrency| `max_workers=8`       | tune to taste                                                         |

---

## Forced-photometry recipe

For each source at a known sky position:

```python
from spherex_retrieval import (
    cutout_to_orig, select_zone_for_source,
    resample_psf_to_native, wavelength_at,
)

x_cut, y_cut = bundle.cutout.spatial_wcs.world_to_pixel(source_coord)

# (1) Map cutout pixels back to original detector pixels for PSF zone lookup.
x_orig, y_orig = cutout_to_orig(
    x_cut, y_cut,
    crpix1a=bundle.cutout.image_header["CRPIX1A"],
    crpix2a=bundle.cutout.image_header["CRPIX2A"],
)

# (2) Pick the correct PSF zone for this source.
plane = select_zone_for_source(bundle.psf_subset, x_orig=x_orig, y_orig=y_orig)
psf_oversamp = bundle.psf_subset.cube[plane]            # 101x101, oversampled

# (3) Resample to native pixels at the source sub-pixel phase. Tractor-ready.
# This can be skipped if user wants to use the oversampled PSF for Tractor.
psf_native = resample_psf_to_native(
    psf_oversamp,
    oversamp=bundle.cutout.psf_oversamp,                # 10 in QR-2
    sub_pixel_shift=(x_cut % 1, y_cut % 1),
)

# (4) Wavelength + bandpass at the source position (not the cutout center).
lam, dlam = wavelength_at(bundle.wavelength, x_cut=x_cut, y_cut=y_cut)
```

`resample_psf_to_native` shifts the super-resolved PSF (10× oversampling by default for QR-2)
and pixel-integrates onto the native detector grid, normalised so it can be passed directly
to forward-modelling tools.
Skipping this step makes the effective PSF width and normalisation wrong — see
[spherex_psf.md §9](.claude/spherex_data_desc/spherex_psf.md) in the IRSA tutorials.

---

## Side data: SAPM (MJy/sr → µJy)

Run with `include_sapm=True` to add a `SAPM` HDU to each cutout.  Convert per the IRSA tutorials:

```python
import astropy.units as u
from astropy.io import fits

with fits.open("cutouts/cutout_0001_<obsid>_D2.fits") as h:
    img  = h["IMAGE"].data * u.MJy / u.sr
    sapm = h["SAPM"].data  * u.arcsec**2
    img_uJy = img.to(u.uJy / u.arcsec**2) * sapm
```

Pin a specific cal version (matching legacy results) with `sapm_cal_token="cal-sapm-v2-2025-164"`.

---

## Notes

### Wavelength source

The Explanatory Supplement explicitly flags the L2 `WCS-WAVE` lookup table as **visualization-only**
(~1 nm accuracy via bilinear interpolation) and recommends the standalone Spectral WCS cal product
(`CWAVE` + `CBAND`) for science.  This package uses the latter: the matching cal file is located
via SIA2 (`COLLECTION=spherex_qr2_cal`) and cropped to the same pixel box as the science cutout
using `.section[ylo:..., xlo:...]`, so cloud reads only fetch the relevant pixel slab.

### PSF erratum (VERSION ≤ 6.5.5)

Spectral images with primary `VERSION ≤ 6.5.5` and no `+psffix1` local tag carry an incorrect
`XCTR_i` / `YCTR_i` per-plane mapping in the PSF HDU header.  `retrieve()` detects this from the
PRIMARY header and rewrites the mapping in memory before zone selection; affected cutouts carry
`PSFFIXED = T` in the output primary header.  Reference:
<https://irsa.ipac.caltech.edu/data/SPHEREx/docs/psfhdrerr.html>.

### Status codes

`summary.ecsv` always has one row per overlapping pointing — failures are tagged, never dropped.

| status               | meaning                                                       |
|----------------------|---------------------------------------------------------------|
| `ok`                 | image + PSF + wavelength all retrieved                        |
| `out_of_bounds`      | requested position falls outside the active detector area     |
| `download_failed`    | cutout request errored (network, server, etc.)                |
| `qa_excluded`        | a QA gate dropped this pointing                               |
| `wavelength_missing` | image OK but the cal product could not be located or cropped  |

---

## References

-[IRSA SPHEREx archive](https://irsa.ipac.caltech.edu/Missions/spherex.html)
-[IRSA cutout service](https://irsa.ipac.caltech.edu/ibe/cutouts.html)
- [IRSA Tutorials for SPHEREx](https://caltech-ipac.github.io/irsa-tutorials/spherex/>)
- [SPHEREx Archive at IRSA User Guide](https://caltech-ipac.github.io/spherex-archive-documentation/>)
- [SPHEREx Explanatory Supplement (QR)](https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf)
- [PSF header erratum](https://irsa.ipac.caltech.edu/data/SPHEREx/docs/psfhdrerr.html)
