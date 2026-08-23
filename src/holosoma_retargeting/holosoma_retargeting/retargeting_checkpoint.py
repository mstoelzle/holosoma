"""Atomic persistence for resumable retargeting runs."""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holosoma_retargeting.xsens.tennis_racket import (
    TennisRacketMotion,
    tennis_racket_motion_from_npz,
)

CHECKPOINT_SCHEMA_VERSION = 1
_RacketFilterState = tuple[bool, int, int | None]


@dataclass(frozen=True)
class RetargetingCheckpoint:
    """Validated state required to continue at the first unfinished frame."""

    qpos: np.ndarray
    cost: float
    orientation_errors_rad: np.ndarray
    axis_errors_deg: np.ndarray
    racket_motion: TennisRacketMotion | None = None
    racket_filter_state: _RacketFilterState | None = None

    @property
    def completed_frames(self) -> int:
        return int(self.qpos.shape[0])


def checkpoint_path_for_result(result_path: str | Path) -> Path:
    """Return the sidecar checkpoint path for a final result path."""

    path = Path(result_path)
    suffix = ".checkpoint.npz"
    return path.with_name(f"{path.stem}{suffix}" if path.suffix == ".npz" else f"{path.name}{suffix}")


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


@contextmanager
def checkpoint_on_error(save: Callable[[], None]) -> Iterator[None]:
    """Best-effort checkpoint before propagating an interrupted frame loop."""

    try:
        yield
    except BaseException as error:
        try:
            save()
        except Exception as checkpoint_error:
            warnings.warn(f"Failed to save retargeting checkpoint: {checkpoint_error}", stacklevel=2)
            if hasattr(error, "add_note"):
                error.add_note(f"Additionally failed to save checkpoint: {checkpoint_error}")
        raise


def checkpoint_payload(
    *,
    total_frames: int,
    qpos: np.ndarray,
    cost: float,
    orientation_errors_rad: np.ndarray,
    axis_errors_deg: np.ndarray,
    racket_motion: TennisRacketMotion | None = None,
    racket_filter_state: _RacketFilterState | None = None,
) -> dict[str, np.ndarray]:
    """Build the lightweight archive payload for an accepted motion prefix."""

    if (racket_motion is None) != (racket_filter_state is None):
        raise ValueError("Racket motion and filter state must be checkpointed together")
    payload = {
        "checkpoint_schema_version": np.asarray(CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
        "checkpoint_total_frames": np.asarray(total_frames, dtype=np.int64),
        "qpos": np.asarray(qpos, dtype=float),
        "cost": np.asarray(cost, dtype=float),
        "orientation_errors_rad": np.asarray(orientation_errors_rad, dtype=float),
        "axis_errors_deg": np.asarray(axis_errors_deg, dtype=float),
    }
    if racket_motion is not None and racket_filter_state is not None:
        active, reentry_streak, previous_branch = racket_filter_state
        payload.update(racket_motion.as_npz_payload())
        payload.update(
            checkpoint_racket_active=np.asarray(active, dtype=np.bool_),
            checkpoint_racket_reentry_streak=np.asarray(reentry_streak, dtype=np.int64),
            checkpoint_racket_previous_branch=np.asarray(
                -1 if previous_branch is None else previous_branch,
                dtype=np.int8,
            ),
        )
    return payload


def _scalar(data: Mapping[str, np.ndarray], key: str) -> object:
    try:
        value = np.asarray(data[key])
    except KeyError as exc:
        raise ValueError(f"Checkpoint is missing required field '{key}'") from exc
    if value.shape != ():
        raise ValueError(f"Checkpoint field '{key}' must be scalar")
    return value.item()


def load_retargeting_checkpoint(
    path: str | Path,
    *,
    total_frames: int,
    nq: int,
    orientation_shape: tuple[int, int] | None,
    racket_tracking_mode: str | None,
) -> RetargetingCheckpoint:
    """Load a checkpoint and reject incompatible recovery state."""

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load retargeting checkpoint '{path}': {exc}") from exc

    version = int(_scalar(data, "checkpoint_schema_version"))
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema version {version}; expected {CHECKPOINT_SCHEMA_VERSION}")
    saved_total = int(_scalar(data, "checkpoint_total_frames"))
    if saved_total != total_frames:
        raise ValueError(f"Checkpoint frame count {saved_total} does not match current motion {total_frames}")

    try:
        qpos = np.asarray(data["qpos"], dtype=float)
        orientation_errors = np.asarray(data["orientation_errors_rad"], dtype=float)
        axis_errors = np.asarray(data["axis_errors_deg"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Checkpoint is missing required field '{exc.args[0]}'") from exc
    completed_frames = qpos.shape[0] if qpos.ndim == 2 else 0
    if qpos.shape != (completed_frames, nq) or not 1 <= completed_frames <= total_frames:
        raise ValueError(f"Checkpoint qpos has incompatible shape {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("Checkpoint qpos must contain only finite values")

    expected_orientation_shape = (0,) if orientation_shape is None else (completed_frames, orientation_shape[0])
    expected_axis_shape = (0,) if orientation_shape is None else (completed_frames, orientation_shape[1])
    if orientation_errors.shape != expected_orientation_shape or axis_errors.shape != expected_axis_shape:
        raise ValueError("Checkpoint orientation diagnostics are not aligned with qpos")
    if not np.isfinite(orientation_errors).all() or not np.isfinite(axis_errors).all():
        raise ValueError("Checkpoint orientation diagnostics must contain only finite values")

    cost = float(_scalar(data, "cost"))
    if not np.isfinite(cost):
        raise ValueError("Checkpoint cost must be finite")

    racket_motion = tennis_racket_motion_from_npz(data)
    if (racket_motion is None) != (racket_tracking_mode is None):
        raise ValueError("Checkpoint tennis-racket state does not match the current retargeting run")
    racket_filter_state = None
    if racket_motion is not None:
        if racket_motion.position_m.shape[0] != completed_frames or racket_motion.tracking_mode != racket_tracking_mode:
            raise ValueError("Checkpoint tennis-racket history is not aligned with qpos")
        previous_branch = int(_scalar(data, "checkpoint_racket_previous_branch"))
        racket_filter_state = (
            bool(_scalar(data, "checkpoint_racket_active")),
            int(_scalar(data, "checkpoint_racket_reentry_streak")),
            None if previous_branch == -1 else previous_branch,
        )
        if racket_filter_state[1] < 0 or racket_filter_state[2] not in (None, 0, 1):
            raise ValueError("Checkpoint tennis-racket filter state is invalid")

    return RetargetingCheckpoint(
        qpos=qpos,
        cost=cost,
        orientation_errors_rad=orientation_errors,
        axis_errors_deg=axis_errors,
        racket_motion=racket_motion,
        racket_filter_state=racket_filter_state,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "RetargetingCheckpoint",
    "atomic_savez",
    "checkpoint_on_error",
    "checkpoint_path_for_result",
    "checkpoint_payload",
    "load_retargeting_checkpoint",
]
