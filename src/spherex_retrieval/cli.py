"""Command-line entry point: ``spherex-retrieve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

from .core import retrieve


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
        choices=["spherex_qr2", "spherex_qr2_deep"],
        help="May be passed multiple times.  Defaults to both wide+deep.",
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
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    coord = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg, frame="icrs")
    size = args.size * u.arcsec

    collections = tuple(args.collection) if args.collection else (
        "spherex_qr2", "spherex_qr2_deep",
    )

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
