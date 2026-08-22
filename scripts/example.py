#!/usr/bin/env python3
"""Analyze one image with the installed facetorch configuration API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from facetorch import FaceAnalyzer, load_config


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one input image.")
    parser.add_argument("--profile", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    parser.add_argument("--face-batch-size", type=int, default=8)
    parser.add_argument(
        "--include-predictor",
        action="append",
        dest="include_predictors",
        help="Predictor name to run; repeat the option to select more than one.",
    )
    parser.add_argument("--include-tensors", action="store_true")
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
    cfg = load_config(args.profile, overrides=args.config_override)
    analyzer = FaceAnalyzer(cfg.analyzer)
    result = analyzer.run(
        image_source=args.image,
        face_batch_size=args.face_batch_size,
        fix_img_size=args.fix_image_size,
        include_predictors=args.include_predictors,
        include_tensors=args.include_tensors,
        path_output=str(args.output) if args.output is not None else None,
    )
    print(result)


if __name__ == "__main__":
    main()
