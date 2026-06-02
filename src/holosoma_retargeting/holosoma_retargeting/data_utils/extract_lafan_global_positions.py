#!/usr/bin/env python3
"""Extract LAFAN BVH global joint positions for Holosoma retargeting."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bvh_global_positions import extract_bvh_directory


@dataclass
class Config:
    """Configuration for extracting LAFAN BVH global positions."""

    input_dir: str = "./lafan1/lafan"
    output_dir: str = "../demo_data/lafan"
    downsample: int = 1


def main(cfg: Config) -> None:
    extract_bvh_directory(
        input_dir=Path(cfg.input_dir),
        output_dir=Path(cfg.output_dir),
        downsample=cfg.downsample,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
