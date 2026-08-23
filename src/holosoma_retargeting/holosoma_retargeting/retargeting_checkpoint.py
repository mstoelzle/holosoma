"""Atomic persistence and validation for resumable retargeting runs."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from holosoma_retargeting.xsens.tennis_racket import (
    TennisRacketMotion,
    tennis_racket_motion_from_npz,
)

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetargetingCheckpoint:
    """Validated state required to continue at the first unfinished frame."""

    completed_frames: int
    qpos: np.ndarray
    cost: float
    orientation_errors_rad: np.ndarray
    axis_errors_deg: np.ndarray
    racket_motion: TennisRacketMotion | None
    racket_active: bool
    racket_reentry_streak: int
    racket_previous_branch: int | None


def checkpoint_path_for_result(result_path: str | Path) -> Path:
    """Return the sidecar checkpoint path for a final result path."""

    path = Path(result_path)
    if path.suffix == ".npz":
        return path.with_name(f"{path.stem}.checkpoint.npz")
    return path.with_name(f"{path.name}.checkpoint.npz")


def atomic_savez(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically replace ``path`` with a NumPy archive built from ``payload``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def checkpoint_payload(
    *,
    total_frames: int,
    qpos: np.ndarray,
    cost: float,
    orientation_errors_rad: np.ndarray,
    axis_errors_deg: np.ndarray,
    racket_motion: TennisRacketMotion | None,
    racket_active: bool = False,
    racket_reentry_streak: int = 0,
    racket_previous_branch: int | None = None,
) -> dict[str, np.ndarray]:
    """Build the lightweight archive payload for an accepted motion prefix."""

    qpos = np.asarray(qpos, dtype=float)
    payload = {
        "checkpoint_schema_version": np.asarray(CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
        "checkpoint_completed_frames": np.asarray(qpos.shape[0], dtype=np.int64),
        "checkpoint_total_frames": np.asarray(total_frames, dtype=np.int64),
        "qpos": qpos,
        "cost": np.asarray(cost, dtype=float),
        "orientation_errors_rad": np.asarray(orientation_errors_rad, dtype=float),
        "axis_errors_deg": np.asarray(axis_errors_deg, dtype=float),
        "checkpoint_has_racket": np.asarray(racket_motion is not None, dtype=np.bool_),
        "checkpoint_racket_active": np.asarray(racket_active, dtype=np.bool_),
        "checkpoint_racket_reentry_streak": np.asarray(racket_reentry_streak, dtype=np.int64),
        "checkpoint_racket_previous_branch": np.asarray(
            -1 if racket_previous_branch is None else racket_previous_branch,
            dtype=np.int8,
        ),
    }
    if racket_motion is not None:
        payload.update(racket_motion.as_npz_payload())
    return payload


def _scalar(data: Mapping[str, np.ndarray], key: str) -> object:
    if key not in data:
        raise ValueError(f"Checkpoint is missing required field '{key}'")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Checkpoint field '{key}' must be scalar")
    return value.item()


def _validate_errors(
    values: np.ndarray,
    *,
    name: str,
    completed_frames: int,
    expected_width: int,
    has_orientation_targets: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    expected_shape = (completed_frames, expected_width) if has_orientation_targets else (0,)
    if values.shape != expected_shape:
        raise ValueError(f"Checkpoint {name} must have shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Checkpoint {name} must contain only finite values")
    return values


def load_retargeting_checkpoint(
    path: str | Path,
    *,
    total_frames: int,
    nq: int,
    has_orientation_targets: bool,
    orientation_error_count: int,
    axis_error_count: int,
    racket_tracking_mode: str | None,
) -> RetargetingCheckpoint:
    """Load a checkpoint and reject incompatible or malformed recovery state."""

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load retargeting checkpoint '{path}': {exc}") from exc

    schema_version = int(_scalar(data, "checkpoint_schema_version"))
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema version {schema_version}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    saved_total_frames = int(_scalar(data, "checkpoint_total_frames"))
    if saved_total_frames != total_frames:
        raise ValueError(f"Checkpoint frame count {saved_total_frames} does not match current motion {total_frames}")
    completed_frames = int(_scalar(data, "checkpoint_completed_frames"))
    if not 1 <= completed_frames <= total_frames:
        raise ValueError(f"Checkpoint completed frame count must be in [1, {total_frames}], got {completed_frames}")

    if "qpos" not in data:
        raise ValueError("Checkpoint is missing required field 'qpos'")
    qpos = np.asarray(data["qpos"], dtype=float)
    if qpos.shape != (completed_frames, nq):
        raise ValueError(f"Checkpoint qpos must have shape ({completed_frames}, {nq}), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("Checkpoint qpos must contain only finite values")

    cost = float(_scalar(data, "cost"))
    if not np.isfinite(cost):
        raise ValueError("Checkpoint cost must be finite")
    if "orientation_errors_rad" not in data or "axis_errors_deg" not in data:
        raise ValueError("Checkpoint is missing orientation diagnostic fields")
    orientation_errors = _validate_errors(
        data["orientation_errors_rad"],
        name="orientation_errors_rad",
        completed_frames=completed_frames,
        expected_width=orientation_error_count,
        has_orientation_targets=has_orientation_targets,
    )
    axis_errors = _validate_errors(
        data["axis_errors_deg"],
        name="axis_errors_deg",
        completed_frames=completed_frames,
        expected_width=axis_error_count,
        has_orientation_targets=has_orientation_targets,
    )

    has_racket = bool(_scalar(data, "checkpoint_has_racket"))
    expects_racket = racket_tracking_mode is not None
    if has_racket != expects_racket:
        raise ValueError("Checkpoint tennis-racket state does not match the current retargeting run")

    racket_motion = tennis_racket_motion_from_npz(data) if has_racket else None
    racket_active = bool(_scalar(data, "checkpoint_racket_active"))
    racket_reentry_streak = int(_scalar(data, "checkpoint_racket_reentry_streak"))
    racket_previous_branch_value = int(_scalar(data, "checkpoint_racket_previous_branch"))
    if racket_reentry_streak < 0:
        raise ValueError("Checkpoint racket reentry streak must be nonnegative")
    if racket_previous_branch_value not in (-1, 0, 1):
        raise ValueError("Checkpoint previous racket branch must be -1, 0, or 1")
    racket_previous_branch = None if racket_previous_branch_value == -1 else racket_previous_branch_value
    if racket_motion is not None:
        if racket_motion.position_m.shape[0] != completed_frames:
            raise ValueError("Checkpoint tennis-racket history is not aligned with qpos")
        if racket_motion.tracking_mode != racket_tracking_mode:
            raise ValueError("Checkpoint tennis-racket tracking mode does not match the current retargeting run")
    elif racket_active or racket_reentry_streak or racket_previous_branch is not None:
        raise ValueError("Checkpoint contains racket filter state without racket motion")

    return RetargetingCheckpoint(
        completed_frames=completed_frames,
        qpos=qpos,
        cost=cost,
        orientation_errors_rad=orientation_errors,
        axis_errors_deg=axis_errors,
        racket_motion=racket_motion,
        racket_active=racket_active,
        racket_reentry_streak=racket_reentry_streak,
        racket_previous_branch=racket_previous_branch,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "RetargetingCheckpoint",
    "atomic_savez",
    "checkpoint_path_for_result",
    "checkpoint_payload",
    "load_retargeting_checkpoint",
]
