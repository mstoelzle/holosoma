#!/usr/bin/env python3
"""Extract native 100STYLE BVH joint positions for Holosoma retargeting.

The output mirrors the LAFAN prep step: one `.npy` file per BVH containing
world joint positions in meters, still in the BVH coordinate convention. The
retargeting loader converts those positions from Y-up to Holosoma's Z-up frame.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lafan1 import extract, utils  # type: ignore[import-not-found]

EXPECTED_100STYLE_JOINTS = [
    "Hips",
    "Chest",
    "Chest2",
    "Chest3",
    "Chest4",
    "Neck",
    "Head",
    "RightCollar",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
    "LeftCollar",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "RightToe",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "LeftToe",
]


def _normalize_name(value: str) -> str:
    return Path(value.strip()).stem


def _parse_frame(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    return int(float(value))


def _load_frame_cuts(csv_path: Path | None) -> dict[str, tuple[int, int]]:
    if csv_path is None:
        return {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Frame cuts CSV not found: {csv_path}")

    cuts: dict[str, tuple[int, int]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return cuts

        fields = {name.lower().strip(): name for name in reader.fieldnames}
        if "style_name" in fields:
            style_key = fields["style_name"]
            motion_codes = sorted(
                field[:-6]
                for field in reader.fieldnames
                if field.endswith("_START") and f"{field[:-6]}_STOP" in reader.fieldnames
            )
            for row in reader:
                style_name = row.get(style_key)
                if not style_name:
                    continue
                style_name = _normalize_name(style_name)
                for motion_code in motion_codes:
                    start = _parse_frame(row.get(f"{motion_code}_START"))
                    stop = _parse_frame(row.get(f"{motion_code}_STOP"))
                    if start is None or stop is None:
                        continue
                    cuts[f"{style_name}_{motion_code}"] = (start, stop)
            return cuts

        name_key = next((fields[key] for key in fields if "file" in key or "name" in key or "style" in key), None)
        start_key = next((fields[key] for key in fields if "start" in key or "begin" in key or "first" in key), None)
        end_key = next((fields[key] for key in fields if "end" in key or "stop" in key or "last" in key), None)
        if name_key is None or start_key is None or end_key is None:
            raise ValueError(
                "Frame cuts CSV must contain either 100STYLE STYLE_NAME/*_START/*_STOP columns "
                "or filename/name, start/begin, and end/stop columns. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            if not row.get(name_key):
                continue
            start = _parse_frame(row.get(start_key))
            end = _parse_frame(row.get(end_key))
            if start is None or end is None:
                continue
            cuts[_normalize_name(row[name_key])] = (start, end)
    return cuts


def extract_global_positions(bvh_file_path: Path) -> tuple[np.ndarray, list[str]]:
    anim = extract.read_bvh(str(bvh_file_path))
    _, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return global_positions / 100.0, anim.bones


@dataclass
class Config:
    """Configuration for extracting 100STYLE BVH global positions."""

    input_dir: str
    output_dir: str = "../demo_data/100style"
    frame_cuts_csv: str | None = None
    downsample: int = 1
    strict_joints: bool = True


def main(cfg: Config) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    if cfg.downsample < 1:
        raise ValueError("downsample must be >= 1")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frame_cuts = _load_frame_cuts(Path(cfg.frame_cuts_csv) if cfg.frame_cuts_csv else None)
    bvh_files = sorted(input_dir.rglob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No .bvh files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for bvh_path in bvh_files:
        positions, joint_names = extract_global_positions(bvh_path)
        if cfg.strict_joints and joint_names != EXPECTED_100STYLE_JOINTS:
            raise ValueError(
                f"Unexpected joint order in {bvh_path}: {joint_names}. "
                f"Expected: {EXPECTED_100STYLE_JOINTS}"
            )

        stem = bvh_path.stem
        if stem in frame_cuts:
            start, end = frame_cuts[stem]
            positions = positions[start:end]

        positions = positions[:: cfg.downsample]
        rel_stem = bvh_path.relative_to(input_dir).with_suffix("").as_posix().replace("/", "__")
        output_path = output_dir / f"{rel_stem}.npy"
        np.save(str(output_path), positions)
        print(f"Saved {output_path} | frames={positions.shape[0]} joints={positions.shape[1]}")


if __name__ == "__main__":
    main(tyro.cli(Config))
