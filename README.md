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
  - [Shared PSF cube (`psf_source="cal"`)](#shared-psf-cube-psf_sourcecal)
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
| 0   | PRIMARY   | provenance — `OBSID`, `DETECTOR`, `RA_REQ`, `DEC_REQ`, `STATUS`, `VERSION`, `PSFFIXED`, `OVERSAMP`, `PSFSRC`, `PSFKIND`, `PSFNORM` (+ `EPSFCAL`, `DETCOORD`, `ZONENX`, `ZONENY` for R7) |
| 1   | IMAGE     | calibrated surface brightness (MJy/sr), cropped                  |
| 2   | FLAGS     | per-pixel bitmap                                                 |
| 3   | VARIANCE  | (MJy/sr)²                                                        |
| 4   | ZODI      | modeled zodiacal background (MJy/sr)                             |
| 5   | PSF       | N_zones×S×S PSF planes restricted to overlapping zones: 101×101 optical (QR2) or 33×33 effective (R7) |
| 6   | PSF_ZONES | lookup: `zone_id`, `x`, `y` (orig 0-based), `plane_idx` (+ `xwidth`, `ywidth`, `nstar`, `neff` for R7) |
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
| Calibration release | `calibration_release="qr3"` (R7 spectral WCS and SAPM for every image, QR2 included) | `None` for each image's own release (the paper's configuration) |
| Gain correction | `gain_correction=True` (QR2 IMAGE × the R7 `l3_flux_corrections` factor, VARIANCE × factor²; recorded as `FLXCORR`) | `False` to keep the QR2 gains (the paper's configuration) |
| PSF        | `subset_psf=True` (overlapping zones only) | `False` to keep all 121 planes                   |
| PSF product | `psf_source="epsf-cal"` (every image gets the R7 effective PSF: R7 images share their own `epsf` library, QR2 images get the `epsf_release` library instead of their optical cube; one download per detector, cutout download stops before the PSF data) | `"cal"` for the product of the image's own release (the QR2 cube on QR2 images: the paper's configuration); `"l2"` to download it with every cutout; `psf_verify_every=N`, `psf_cal_token=...`, `epsf_release="qr3"` |
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
via SIA2 (`COLLECTION=spherex_<release>_cal`) or the IRSA directory listing and cropped to the
same pixel box as the science cutout using `.section[ylo:..., xlo:...]`, so cloud reads only
fetch the relevant pixel slab.

Which release's calibration is a choice (`calibration_release`, default `"qr3"`): the spectral
WCS and the solid-angle map are properties of the detector in the L2 pixel frame (`DETCOORD =
'sky'` in both releases), so the R7 on-sky calibration is applied to QR2 images as well. Between
`cal-wcs-v4-2025-254` (QR2) and `cal-swcs-v5-2026-191` (R7) the D1 maps are identical and band 5
is shifted by a constant +0.0064 µm (about a fifth of a channel); bands 5 and 6 are the ones the
Supplement says were recalibrated against on-sky data. `calibration_release=None` takes each
image's own release.

### QR2 gains → R7 gains (`gain_correction`)

The absolute gain was re-derived for R7, and the Supplement advises caution when combining QR2
and QR3 data. The SSDC's `l3_flux_corrections_D<n>` products (`qr3/l3_flux_corrections/cal-flxc-v1-2026-191`)
are per-pixel *multiplicative* factors for pre-QR3 data (D1 median 0.970, varying 0.875–0.983
along the dispersion direction; D5 median 1.007). With `gain_correction=True` (default) a QR2
cutout's `IMAGE` is multiplied by the cropped factor and its `VARIANCE` by the factor squared
before writing; `ZODI` (a model in physical units) is left alone; the few dead/hot-pixel factors
(outside 0.5–2) are replaced by their detector-row median. The bundle records `FLXCORR` (cal
token), `FLXCMED` (median factor over the cutout), `FLXCSRC` and `FLXCNREP`. R7 images are never
touched. This is a stop-gap until DR1 reprocesses the QR2 epochs with the R7 gains;
`gain_correction=False` keeps the QR2 gains (the paper's configuration).

### Two PSF kinds: QR2 optical cube and R7 effective PSF (`PSFKIND`)

QR2 files (pipeline 6.x) carry an *optical* PSF: 121 planes on an 11×11 zone lattice, 10×
oversampled, with the detector pixel response deconvolved. Forward models must integrate it
over each native pixel. QR3 and DR1 files (pipeline R7) carry the *effective* PSF instead
(Anderson & King 2000): the `EPSF` binary table, one 33×33 array at 5× per zone of a 21×21
lattice (11×41 on D3), with the pixel response *included*. Forward models must sample it at
the pixel centres (×25 for the fraction per native pixel) and never integrate it again
(doing so widens the PSF by 1/12 px² of variance, ~30 % in N_eff on SPHEREx).

The bundle layout is the same for both. The PRIMARY header says which kind it holds
(`PSFKIND = 'OPTICAL' | 'EPSF'`, `OVERSAMP = 10 | 5`, `PSFNORM = 'hr-sum-1'`: each plane sums
to 1 on its own oversampled grid), and for R7 adds `EPSFCAL` (the calibration source file the
`EPSF` header names), `DETCOORD` (`'sky'`: arrays and zone centres are in the L2 image
orientation for every detector, including the X-flipped MWIR ones — no mirroring), and the
lattice size `ZONENX`/`ZONENY`. `PSF_ZONES` gains the zone widths, star counts and `N_eff`
from the table. The zone lattice is read from the table, never assumed.

Discovery: QR3 images are not yet in IRSA's SIA2/CAOM service (2026-09-18), so
`spherex_qr3` / `spherex_qr3_deep` are resolved through the `spherex.plane` /
`spherex.artifact` TAP tables (footprint `poly`, release from the artifact path); the default
collection list now covers both releases, oldest first. Calibration products come from the
image's own release directory (`qr3/spectral_wcs` is `cal-swcs-v5-...`, `qr3/epsf` is
`cal-epsf-v1-...`, D3 `v2`), resolved per detector.

`psf_source="epsf-cal"` (the default) gives every image the R7 effective PSF. R7 images share
their own per-detector `epsf` library, verified against the L2 file as described below; a QR2
image gets the R7 library of `epsf_release` (default `qr3`) attached in place of its optical
cube, which is never downloaded, and its bundle says `PSFKIND = 'EPSF'`,
`PSFSRC = 'epsf:<file>'`. This is the recommended product for QR2 images too: on A2537 the R7
ePSF fits stars better than the QR2 cube even after the core re-registration (chi2/dof 1.6 vs
2.9, central stacked residual 1 % vs 4 %), and the QR2 cube's registration offset (−0.053,
−0.050 px) is a property of that product, not of the images (the ePSF on the same images shows
≤ 0.016 px). No check against the L2 file is possible for the attachment. Use
`psf_source="cal"` to get the QR2 cube on QR2 images, the paper's configuration of record.

### Shared PSF product (`psf_source="cal"`)

The 121×101×101 PSF cube in an L2 file is a per-detector calibration constant: it is
byte-identical to the `PSF-DATA-CUBE` of the standalone
[`average_psf`](https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/average_psf) cal product
(checked on all six detectors, 1,370 cutouts across pipeline `VERSION` 6.4 – 6.5.7). IRSA's
cutout service passes it through uncropped, so a small cutout is 4.94 MB of PSF out of ~5.1 MB,
re-sent on every request.

With `psf_source="cal"` (and with the default `"epsf-cal"` on R7 images, where the product is
the image's own) the product is fetched once per detector from the cal file and each cutout
download hangs up right after the PSF *header* (~0.12 MB; the service ignores
`Range`, so closing the stream is the only way). The PSF header — the zone table and its
erratum handling — still comes from the L2 file, and the written bundles are identical to
`psf_source="l2"` apart from the `PSFSRC` keyword (`l2` or `cal:<file>`). Partial downloads are
not written to the HTTP cache.

The two published cal versions, `cal-psf-v5-2025-206` and `cal-psf-v5-2026-082`, hold the same
cube; the later one is a header reissue (zone-centre `XCTR_i`/`YCTR_i` X↔Y erratum fixed,
`VERSION`/`DATE` added).

No QR2 L2 keyword names the PSF cal a file was built with, so the identity is sampled rather
than assumed: the first cutout of each detector in a process, and every `psf_verify_every`-th
(200) after it, is downloaded in full and compared with the cal cube. A mismatch emits a
`RuntimeWarning` and switches that detector back to full downloads.

The same mechanism serves the R7 `EPSF` table (3.86 MB of a ~4.4 MB cutout) from the
[`epsf`](https://irsa.ipac.caltech.edu/ibe/data/spherex/qr3/epsf) cal product; there the
`EPSF` header of every cutout names its calibration source file, which the truncated stream
still delivers, so every R7 cutout is also checked by that provenance string against the
library's (after the same first full download). The IRSA cutout service answers HTTP 503 under
four concurrent requests; two are fine.

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
