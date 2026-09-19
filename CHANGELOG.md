# Changelog

All notable changes to spherex-retrieval. This project follows
[Semantic Versioning](https://semver.org/); while the major version is 0 the
public API may still change between minor releases.

## [Unreleased]

Nothing yet.

## [0.3.0] — 2026-09-19

### Added

- **R7 calibration for QR2 images.** `retrieve(..., calibration_release="qr3")`
  (CLI `--calibration-release`) selects the release whose per-detector spectral
  WCS (`CWAVE`/`CBAND`) and solid-angle map are used for *every* image, QR2
  included; the default is the R7 on-sky calibration. The R7 maps are in the
  same L2 pixel frame as the QR2 ones (D1 identical; band 5 shifted by a
  constant +0.0064 µm). `None` takes each image's own release.
- **QR2 gains → R7 gains.** `retrieve(..., gain_correction=True)` (default;
  CLI `--no-gain-correction` to opt out) multiplies a pre-R7 cutout's `IMAGE`
  by the SSDC `l3_flux_corrections` factor and its `VARIANCE` by the square
  (`ZODI` untouched; dead/hot-pixel factors replaced by their detector-row
  median). Recorded in the bundle as `FLXCORR`, `FLXCMED`, `FLXCSRC`,
  `FLXCNREP`. R7 images are never touched. A stop-gap until DR1 reprocesses
  the QR2 epochs.
- `l3_flux_corrections` calibration family (`cal-flxc-...`).

## [0.2.0] — 2026-09-18

### Added

- **R7 effective PSF (QR3 and DR1).** The `EPSF` binary table of effective
  PSFs (33×33 at 5×, 21×21 zones, 11×41 on D3) is read on both backends; the
  per-detector `epsf` library is shared across cutouts like the QR2 cube,
  verified by the `Calibration source file` provenance of every cutout's
  `EPSF` header; the zone lattice is inferred from the table. Bundles keep
  their HDU layout and add `PSFKIND` (`OPTICAL` / `EPSF`), `PSFNORM`,
  `EPSFCAL`, `DETCOORD`, `ZONENX`/`ZONENY`, and zone widths, star counts and
  `N_eff` in `PSF_ZONES`.
- **`psf_source="epsf-cal"`, now the default.** Every image gets the R7
  effective PSF: R7 images share their own library, QR2 images get the
  `epsf_release` library in place of their optical cube (which is never
  downloaded). `"cal"` keeps the product of the image's own release, the
  configuration of the SPHEREx deblending paper.
- **QR3 discovery.** `spherex_qr3` / `spherex_qr3_deep` in the default
  collection list; while IRSA's SIA2/CAOM service does not list QR3, those
  collections are resolved through the `spherex.plane` / `spherex.artifact`
  TAP tables (footprint `poly`, release from the artifact path).
- Calibration tokens resolved per detector (D3 `epsf` is `v2`); per-family
  token prefixes (`qr3` spectral WCS is `cal-swcs`).

### Fixed

- The pyvo/TAP backend's ADQL against the current IRSA schema (`p.obsid`;
  observation id from the file name; S3 URI derived from the IBE path).

## [0.1.0] — 2026-04 … 2026-09

Initial QR2 retrieval: SIA2 / TAP discovery, IRSA cutout service and
fsspec/S3 backends, per-cutout MEF bundles (IMAGE, FLAGS, VARIANCE, ZODI,
PSF, PSF_ZONES, CWAVE, CBAND, SAPM), the QR2 PSF-header erratum fix,
zone-subsetted PSF cubes with a neighbour ring, the correct cutout→detector
pixel origin (`CRPIX1A`/`CRPIX2A`), and the shared per-detector PSF cube
(`psf_source="cal"`) that stops each cutout download before the PSF data.
