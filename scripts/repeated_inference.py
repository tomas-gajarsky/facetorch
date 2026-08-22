#!/usr/bin/env python3
"""Run repeated inference for one or more explicitly supplied images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from facetorch import FaceAnalyzer, load_config


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--profile", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--face-batch-size", type=int, default=8)
    parser.add_argument("--fix-image-size", action="store_true")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Hydra override applied after the selected packaged profile.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    cfg = load_config(args.profile, overrides=args.config_override)
    analyzer = FaceAnalyzer(cfg.analyzer)
    for _ in range(args.repeats):
        for image in args.images:
            analyzer.run(
                image_source=image,
                face_batch_size=args.face_batch_size,
                fix_img_size=args.fix_image_size,
            )


if __name__ == "__main__":
    main()
