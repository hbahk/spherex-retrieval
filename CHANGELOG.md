# Changelog

All notable changes to spherex-retrieval. This project follows
[Semantic Versioning](https://semver.org/); while the major version is 0 the
public API may still change between minor releases.

## [Unreleased]

Nothing yet.

## [0.3.2] — 2026-09-25

### Added

- **Local L2 archive.** `spherex-index build <archive_root> -o index.parquet`
  indexes a local copy of the L2 tree: it joins the archive's ObsCore level-2
  rows to the files on disk, indexes files no table lists from their own
  headers (centre at the SIP position of the middle pixel, corners at the
  outer pixel edges, `MJD-BEG`/`MJD-END`, matching ObsCore), and keeps one
  file per exposure and detector, its latest processing version.
  `spherex-index info` summarises an index. `find_overlapping_many` pairs many
  targets with frames at once (KD-tree cone on frame centres, then the
  `s_region` quadrilateral grown by half the box diagonal, in the gnomonic
  plane), returned file-major.
- **Local backends.** `query_backend="local"` searches an index;
  `cutout_backend="local"` crops the files in place with the cutout service's
  rules; `retrieve()` takes `index=`, `archive_root=` and `cal_roots=`.
  `set_local_cal_roots()` (or `SPHEREX_CAL_ROOTS`) serves a release's
  calibration products (spectral WCS, SAPM, average PSF, ePSF library, flux
  corrections) from a local tree laid out like IRSA's, with no network; a
  configured release with a missing product is an error, not a silent fall
  back to IRSA. On the IBS olaf QR2 archive the local bundles of QSO
  J0233+0653 are bit-identical to IRSA's in every plane (248 of 248).
- **Frame-major extraction for many targets** (`spherex_retrieval.frame`).
  `iter_frame_cutouts` reads each L2 frame once with raw `pread` (walking the
  headers for the offsets; the QR2 archive has five file layouts) and cuts
  every target on it, fetching either the four pixel planes or only the
  detector rows the targets need; the bundles equal the per-target ones bit
  for bit. `retrieve_catalog` pairs a catalogue with frames and reads them on
  a thread pool; `write_catalog_bundles` writes one `retrieve()`-style
  directory per target. For the 1,456 LSST DP1 QSOs (about 400k cutouts) one
  CPU node extracts everything in about 15 minutes.
- **In-memory bundles.** `bundle.bundle_hdulist` / `bundle_bytes` hand a
  bundle on as the bytes `write_bundle` would write, for streaming into
  photometry without touching the disk.

### Fixed

- **QR3 wide and deep returned the same files.** `query_tap` selected the
  release by artifact path only, so `spherex_qr3` and `spherex_qr3_deep` both
  returned every QR3 image and `find_overlapping` stacked the two lists; each
  QR3 exposure was retrieved twice (QSO J0233+0653: 479 rows for 364 files).
  It now joins `spherex.observation` and filters on its collection, and
  `find_overlapping` keeps one row per L2 file (364 rows for 364 files).
- **fsspec cutout window and origin.** The client-side crop took
  `ceil(size / scale)` pixels through `Cutout2D`, while the cutout service
  returns `round(size / scale)` per axis starting at `floor(c + 1 - n/2)`
  (15 vs 16 px for the 92.25 arcsec reference box), and wrote `CRPIX1A/2A`
  with the wrong sign, which mirrored the detector origin and could pick the
  wrong PSF zones. Both now follow the service, including the wavelength
  WCS reference pixels.
- The index builder's header-scan workers start with `spawn` (forking a
  multi-threaded process could deadlock).

## [0.3.1] — 2026-09-19

### Fixed

- **`psf_source="epsf-cal"` with a warm HTTP cache.** A cutout whose full
  download was already cached (from an earlier `"cal"`/`"l2"` retrieval of the
  same box) was returned with the file's own QR2 optical cube, silently
  skipping the R7 library attachment; the bundle then said `PSFKIND =
  'OPTICAL'`. The cached pixels are still reused, but the attaching registry
  now swaps in its library. Verifying registries (`"cal"`) are unchanged.

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
