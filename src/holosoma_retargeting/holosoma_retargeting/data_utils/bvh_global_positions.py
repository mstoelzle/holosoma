"""Shared BVH-to-global-position extraction utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

DATA_UTILS_DIR = Path(__file__).resolve().parent
if str(DATA_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_UTILS_DIR))

from lafan1 import extract, utils  # type: ignore[import-not-found]

FrameWindow = tuple[int, int]
FrameWindowResolver = Callable[[Path], FrameWindow | None]


@dataclass(frozen=True)
class BvhGlobalPositions:
    """World-space BVH joint positions and skeleton metadata."""

    positions: np.ndarray
    joint_names: list[str]
    parents: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.positions.shape[1])


def extract_global_positions(bvh_file_path: Path | str) -> BvhGlobalPositions:
    """Read a BVH file and compute global joint positions in meters."""
    anim = extract.read_bvh(str(bvh_file_path))
    _, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return BvhGlobalPositions(
        positions=global_positions / 100.0,
        joint_names=list(anim.bones),
        parents=anim.parents,
    )


def validate_joint_order(joint_names: list[str], expected_joint_names: list[str] | None, bvh_path: Path) -> None:
    if expected_joint_names is None or joint_names == expected_joint_names:
        return
    raise ValueError(f"Unexpected joint order in {bvh_path}: {joint_names}. Expected: {expected_joint_names}")


def apply_frame_window(positions: np.ndarray, frame_window: FrameWindow | None) -> np.ndarray:
    if frame_window is None:
        return positions
    start, end = frame_window
    return positions[start:end]


def output_stem_for_bvh(bvh_path: Path, input_dir: Path, *, preserve_subdirs: bool = False) -> str:
    if preserve_subdirs:
        return bvh_path.relative_to(input_dir).with_suffix("").as_posix().replace("/", "__")
    return bvh_path.stem


def iter_bvh_files(input_dir: Path, *, recursive: bool = False) -> list[Path]:
    pattern = "**/*.bvh" if recursive else "*.bvh"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def extract_bvh_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    preserve_subdirs: bool = False,
    downsample: int = 1,
    expected_joint_names: list[str] | None = None,
    frame_window_for_path: FrameWindowResolver | None = None,
) -> list[Path]:
    """Extract all BVH files in a directory to `.npy` global-position files."""
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    bvh_files = iter_bvh_files(input_dir, recursive=recursive)
    if not bvh_files:
        raise FileNotFoundError(f"No .bvh files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for bvh_path in bvh_files:
        result = extract_global_positions(bvh_path)
        validate_joint_order(result.joint_names, expected_joint_names, bvh_path)

        frame_window = frame_window_for_path(bvh_path) if frame_window_for_path is not None else None
        positions = apply_frame_window(result.positions, frame_window)
        positions = positions[::downsample]

        output_path = output_dir / f"{output_stem_for_bvh(bvh_path, input_dir, preserve_subdirs=preserve_subdirs)}.npy"
        np.save(str(output_path), positions)
        output_paths.append(output_path)
        print(f"Saved {output_path} | frames={positions.shape[0]} joints={positions.shape[1]}")

    return output_paths
