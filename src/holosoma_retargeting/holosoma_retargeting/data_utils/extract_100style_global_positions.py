#!/usr/bin/env python3
"""Extract native 100STYLE BVH joint positions for Holosoma retargeting.

The output mirrors the LAFAN prep step: one `.npy` file per BVH containing
world joint positions in meters, still in the BVH coordinate convention. The
retargeting loader converts those positions from Y-up to Holosoma's Z-up frame.
"""

from __future__ import annotations

import sys
import csv
from dataclasses import dataclass
from pathlib import Path

import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bvh_global_positions import extract_bvh_directory

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


@dataclass
class Config:
    """Configuration for extracting 100STYLE BVH global positions."""

    input_dir: str
    output_dir: str = "../demo_data/100style"
    frame_cuts_csv: str | None = None
    downsample: int = 1
    strict_joints: bool = True


def main(cfg: Config) -> None:
    frame_cuts = _load_frame_cuts(Path(cfg.frame_cuts_csv) if cfg.frame_cuts_csv else None)

    def frame_window_for_path(bvh_path: Path) -> tuple[int, int] | None:
        return frame_cuts.get(bvh_path.stem)

    extract_bvh_directory(
        input_dir=Path(cfg.input_dir),
        output_dir=Path(cfg.output_dir),
        recursive=True,
        preserve_subdirs=True,
        downsample=cfg.downsample,
        expected_joint_names=EXPECTED_100STYLE_JOINTS if cfg.strict_joints else None,
        frame_window_for_path=frame_window_for_path,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
