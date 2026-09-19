"""Command-line entry point: ``spherex-retrieve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

from .core import retrieve
from .query import SUPPORTED_COLLECTIONS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spherex-retrieve",
        description="Download SPHEREx Spectral Image cutouts with side data for forced photometry.",
    )
    p.add_argument("--ra", type=float, required=True, help="Right ascension (deg, ICRS).")
    p.add_argument("--dec", type=float, required=True, help="Declination (deg, ICRS).")
    p.add_argument("--size", type=float, required=True, help="Cutout extent (arcsec, square).")
    p.add_argument("--out", type=Path, default=None, help="Output directory.")
    p.add_argument(
        "--query-backend", choices=["astroquery", "pyvo"], default="astroquery"
    )
    p.add_argument(
        "--cutout-backend", choices=["irsa", "fsspec"], default="irsa"
    )
    p.add_argument(
        "--collection",
        action="append",
        choices=list(SUPPORTED_COLLECTIONS),
        help="May be passed multiple times.  Defaults to every quick release, "
             "wide and deep (QR2: optical PSF cube; QR3: R7 effective PSF).",
    )
    p.add_argument(
        "--bandpass",
        default=None,
        help="Restrict to one detector, e.g. SPHEREx-D2.",
    )
    p.add_argument("--no-wavelength", action="store_true", help="Skip CWAVE/CBAND fetch.")
    p.add_argument(
        "--include-sapm",
        action="store_true",
        help="Also fetch the matching Solid Angle Pixel Map cal product.",
    )
    p.add_argument(
        "--sapm-cal-token",
        default=None,
        help="Pin SAPM cal version (e.g. cal-sapm-v2-2025-164).",
    )
    p.add_argument("--no-psf-subset", action="store_true", help="Keep all 121 PSF planes.")
    p.add_argument("--zone-margin", type=int, default=1,
                        help="widen the retained PSF-zone rectangle by N zones on "
                             "every side so downstream photometry can interpolate "
                             "between zones; 0 reproduces pre-2026-07-28 bundles")
    p.add_argument("--psf-source", choices=["epsf-cal", "cal", "l2"], default="epsf-cal",
                   help="epsf-cal (default): every image gets the R7 effective PSF -- R7 "
                        "images share their own verified epsf library, QR2 images get the "
                        "--epsf-release library attached instead of their optical cube; "
                        "cal: the product of the image's own release (the QR2 cube for QR2 "
                        "images, the paper's configuration), shared per detector with the "
                        "cutout download stopped before the PSF data; l2: download it with "
                        "every cutout")
    p.add_argument("--epsf-release", default="qr3",
                   help="release whose ePSF library --psf-source epsf-cal attaches")
    p.add_argument("--calibration-release", default="qr3",
                   help="release whose spectral WCS (CWAVE/CBAND), SAPM and flux corrections "
                        "are used for every image (default qr3, the R7 on-sky calibration); "
                        "'own' takes each image's own release (the paper's configuration)")
    p.add_argument("--no-gain-correction", action="store_true",
                   help="do not apply the R7 l3_flux_corrections to QR2 images")
    p.add_argument("--psf-verify-every", type=int, default=200,
                   help="with --psf-source cal, download every N-th cutout per detector "
                        "in full and compare its PSF cube with the cal product "
                        "(0 = first cutout only)")
    p.add_argument("--psf-cal-token", default=None,
                   help="Pin PSF cal version (e.g. cal-psf-v5-2026-082 or cal-epsf-v1-2026-191).")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    coord = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg, frame="icrs")
    size = args.size * u.arcsec

    collections = tuple(args.collection) if args.collection else SUPPORTED_COLLECTIONS

    bundles, out_dir = retrieve(
        coord,
        size,
        output_dir=args.out,
        query_backend=args.query_backend,
        cutout_backend=args.cutout_backend,
        collections=collections,
        bandpass=args.bandpass,
        include_wavelength=not args.no_wavelength,
        include_sapm=args.include_sapm,
        sapm_cal_token=args.sapm_cal_token,
        subset_psf=not args.no_psf_subset,
        zone_margin=args.zone_margin,
        psf_source=args.psf_source,
        psf_verify_every=args.psf_verify_every,
        psf_cal_token=args.psf_cal_token,
        epsf_release=args.epsf_release,
        calibration_release=(None if args.calibration_release == "own"
                             else args.calibration_release),
        gain_correction=not args.no_gain_correction,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
    )

    n_ok = sum(b.is_ok for b in bundles)
    print(
        f"retrieved {n_ok}/{len(bundles)} cutouts -> {out_dir}",
        file=sys.stderr,
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
