#!/usr/bin/env python3
"""Play G1 qpos and global Xsens HDF5 motion in one Viser scene."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig, XsensViserConfig  # noqa: E402
from holosoma_retargeting.data_utils.xsens_hdf5 import (  # noqa: E402
    XsensHdf5Motion,
    load_xsens_hdf5_motion,
)
from holosoma_retargeting.src.recording_utils import (  # noqa: E402
    build_record_frame_indices,
    record_viser_sequence,
)
from holosoma_retargeting.src.viser_utils import (  # noqa: E402
    QposViserApplier,
    create_motion_control_sliders,
    create_timed_motion_control_sliders,
    sample_qpos_at_time,
)
from holosoma_retargeting.src.xsens_viser import (  # noqa: E402
    XsensKinematicPositionProjector,
    XsensMotionSampler,
    XsensUsdActor,
    build_subject_xsens_reference_model,
    load_xsens_usd_model,
    reference_grounding_offset_m,
    resolve_g1_xsens_usd,
    resolve_package_path,
    resolve_subject_xsens_usd,
    validate_g1_xsens_usd,
    validate_subject_xsens_usd,
)


def load_npz(npz_path: str) -> tuple[np.ndarray, int]:
    data = np.load(resolve_package_path(npz_path), allow_pickle=True)
    qpos = np.asarray(data["qpos"])
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def _nominal_fps(times_s: np.ndarray, fallback: float) -> float:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size <= 1:
        return float(fallback)
    intervals = np.diff(times)
    intervals = intervals[intervals > 0.0]
    return float(1.0 / np.median(intervals)) if intervals.size else float(fallback)


ActorMode = Literal["robot", "xsens", "g1_xsens"]
ACTOR_MODE_ORDER: tuple[ActorMode, ...] = ("robot", "xsens", "g1_xsens")


def resolve_actor_modes(actor_modes: Sequence[str]) -> tuple[ActorMode, ...]:
    """Expand aliases, validate actor modes, and return a stable unique ordering."""

    requested = tuple(actor_modes)
    if not requested:
        raise ValueError("actor_modes must contain at least one actor")
    allowed = {*ACTOR_MODE_ORDER, "all"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown actor modes: {unknown}. Expected values from {sorted(allowed)}")
    if "all" in requested:
        return ACTOR_MODE_ORDER
    return tuple(mode for mode in ACTOR_MODE_ORDER if mode in requested)


def resolve_xsens_actor_offsets(
    actor_modes: Sequence[str],
    g1_xsens_composition_offset_m: Sequence[float],
) -> dict[ActorMode, np.ndarray]:
    """Return visual root offsets while preserving each actor's local HDF5 poses."""

    active_modes = resolve_actor_modes(actor_modes)
    configured_offset = np.asarray(g1_xsens_composition_offset_m, dtype=float)
    if configured_offset.shape != (3,) or not np.isfinite(configured_offset).all():
        raise ValueError("g1_xsens_composition_offset_m must contain three finite xyz values")
    offsets = {
        mode: np.zeros(3, dtype=float)
        for mode in active_modes
        if mode in {"xsens", "g1_xsens"}
    }
    if "xsens" in offsets and "g1_xsens" in offsets:
        offsets["g1_xsens"] = configured_offset
    return offsets


@dataclass(frozen=True)
class GroundPlaneBounds:
    """Dataset-derived horizontal bounds for the Viser ground grid."""

    center_xy: np.ndarray
    width: float
    height: float


def compute_ground_plane_bounds(
    *,
    qpos: np.ndarray | None = None,
    xsens_positions_m: np.ndarray | None = None,
    robot_dof: int = 0,
    contains_object_in_qpos: bool = False,
    padding_m: float = 1.0,
    minimum_width_m: float | None = None,
    minimum_height_m: float | None = None,
) -> GroundPlaneBounds:
    """Compute a centered grid that covers every displayed motion position."""

    if padding_m < 0.0:
        raise ValueError("padding_m must be non-negative")
    if minimum_width_m is not None and minimum_width_m <= 0.0:
        raise ValueError("minimum_width_m must be positive when provided")
    if minimum_height_m is not None and minimum_height_m <= 0.0:
        raise ValueError("minimum_height_m must be positive when provided")

    point_sets: list[np.ndarray] = []
    if qpos is not None:
        robot_motion = np.asarray(qpos, dtype=float)
        if robot_motion.ndim != 2 or robot_motion.shape[0] == 0 or robot_motion.shape[1] < 3:
            raise ValueError("qpos must have shape [frames, values] and contain base xyz")
        point_sets.append(robot_motion[:, 0:2])
        if contains_object_in_qpos and robot_motion.shape[1] >= 7 + robot_dof + 7:
            point_sets.append(robot_motion[:, -7:-5])

    if xsens_positions_m is not None:
        xsens_positions = np.asarray(xsens_positions_m, dtype=float)
        if xsens_positions.ndim != 3 or xsens_positions.shape[-1] != 3 or xsens_positions.shape[0] == 0:
            raise ValueError("xsens_positions_m must have shape [frames, segments, 3]")
        point_sets.append(xsens_positions[..., 0:2].reshape(-1, 2))

    if not point_sets:
        raise ValueError("At least one motion dataset is required to calibrate the ground plane")
    points = np.concatenate(point_sets, axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError("Motion datasets contain no finite horizontal positions")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = maximum - minimum
    width = float(span[0] + 2.0 * padding_m)
    height = float(span[1] + 2.0 * padding_m)
    if minimum_width_m is not None:
        width = max(width, float(minimum_width_m))
    if minimum_height_m is not None:
        height = max(height, float(minimum_height_m))
    return GroundPlaneBounds(center_xy=center, width=width, height=height)


def make_player(
    config: ViserConfig,
    qpos: np.ndarray | None = None,
    fps: int | None = None,
    xsens_motion: XsensHdf5Motion | None = None,
) -> viser.ViserServer:
    """Create a Viser player containing any configured combination of actors."""

    xsens_config = config if isinstance(config, XsensViserConfig) else None
    actor_modes = resolve_actor_modes(xsens_config.actor_modes if xsens_config is not None else ("robot",))
    show_robot = "robot" in actor_modes
    show_xsens = "xsens" in actor_modes or "g1_xsens" in actor_modes
    xsens_actor_offsets = resolve_xsens_actor_offsets(
        actor_modes,
        xsens_config.g1_xsens_composition_offset_m if xsens_config is not None else (0.0, 0.0, 0.0),
    )
    if show_robot and qpos is None:
        raise ValueError(f"actor_modes={actor_modes} requires qpos motion for the robot actor")
    if show_xsens and (
        xsens_config is None or xsens_motion is None or xsens_config.xsens_hdf5 is None
    ):
        raise ValueError(f"actor_modes={actor_modes} requires an XsensViserConfig with --xsens-hdf5")

    server = viser.ViserServer()

    vr: ViserUrdf | None = None
    vo: ViserUrdf | None = None
    robot_root = None
    object_root = None
    robot_dof = 0
    robot_applier: QposViserApplier | None = None
    actual_robot_fps = float(fps if fps is not None else config.fps)

    if show_robot:
        robot_root = server.scene.add_frame("/robot", show_axes=False)
        robot_urdf = yourdfpy.URDF.load(
            resolve_package_path(config.robot_urdf),
            load_meshes=True,
            build_scene_graph=True,
        )
        vr = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
        vr.show_visual = config.show_meshes
        robot_dof = len(vr.get_actuated_joint_limits())

        if config.object_urdf:
            object_root = server.scene.add_frame("/object", show_axes=False)
            object_urdf = yourdfpy.URDF.load(
                resolve_package_path(config.object_urdf),
                load_meshes=True,
                build_scene_graph=True,
            )
            vo = ViserUrdf(server, urdf_or_path=object_urdf, root_node_name="/object")
            vo.show_visual = config.show_meshes

        assert qpos is not None and robot_root is not None
        robot_applier = QposViserApplier(
            viser_robot=vr,
            robot_base_frame=robot_root,
            robot_dof=robot_dof,
            viser_object=vo if config.assume_object_in_qpos else None,
            object_base_frame=object_root if config.assume_object_in_qpos else None,
            contains_object_in_qpos=config.assume_object_in_qpos,
        )

    xsens_actors: dict[ActorMode, XsensUsdActor] = {}
    xsens_sampler: XsensMotionSampler | None = None
    g1_xsens_projector: XsensKinematicPositionProjector | None = None
    subject_reference_model = None
    g1_grounding_offset_m = 0.0
    if show_xsens:
        assert xsens_config is not None and xsens_motion is not None and xsens_config.xsens_hdf5 is not None
        xsens_sampler = XsensMotionSampler(xsens_motion)
        if "xsens" in actor_modes:
            subject_model_path = resolve_subject_xsens_usd(xsens_config.xsens_hdf5, xsens_config.xsens_usd)
            subject_model = load_xsens_usd_model(subject_model_path)
            validate_subject_xsens_usd(subject_model, xsens_config.xsens_hdf5)
            subject_reference_model = subject_model
            xsens_actors["xsens"] = XsensUsdActor(
                server,
                subject_model_path,
                model=subject_model,
                root_node_name="/xsens",
                show_meshes=xsens_config.show_xsens_meshes,
                show_landmarks=xsens_config.show_xsens_landmarks,
            )
        if "g1_xsens" in actor_modes:
            g1_model_path = resolve_g1_xsens_usd(xsens_config.g1_xsens_usd)
            g1_model = load_xsens_usd_model(g1_model_path)
            validate_g1_xsens_usd(g1_model)
            if subject_reference_model is None:
                subject_reference_model = build_subject_xsens_reference_model(xsens_config.xsens_hdf5)
            g1_grounding_offset_m = reference_grounding_offset_m(subject_reference_model, g1_model)
            xsens_actor_offsets["g1_xsens"][2] += g1_grounding_offset_m
            xsens_actors["g1_xsens"] = XsensUsdActor(
                server,
                g1_model_path,
                model=g1_model,
                root_node_name="/g1_xsens",
                show_meshes=xsens_config.show_xsens_meshes,
                show_landmarks=xsens_config.show_xsens_landmarks,
            )
            g1_xsens_projector = XsensKinematicPositionProjector(
                g1_model,
                xsens_sampler.segment_names,
            )
        for mode, actor in xsens_actors.items():
            actor.root.position = xsens_actor_offsets[mode]

    contains_object = bool(qpos is not None and robot_applier is not None and robot_applier.has_object_input(qpos))
    xsens_ground_positions = None
    if show_xsens and xsens_motion is not None:
        xsens_ground_positions = np.concatenate(
            [
                xsens_motion.positions_m + xsens_actor_offsets[mode][None, None, :]
                for mode in xsens_actors
            ],
            axis=1,
        )
    ground = compute_ground_plane_bounds(
        qpos=qpos if show_robot else None,
        xsens_positions_m=xsens_ground_positions,
        robot_dof=robot_dof,
        contains_object_in_qpos=contains_object,
        padding_m=config.grid_padding,
        minimum_width_m=config.grid_width,
        minimum_height_m=config.grid_height,
    )
    server.scene.add_grid(
        "/grid",
        width=ground.width,
        height=ground.height,
        position=(float(ground.center_xy[0]), float(ground.center_xy[1]), 0.0),
    )
    print(
        f"[viser_player] ground center=({ground.center_xy[0]:.3f}, {ground.center_xy[1]:.3f}) | "
        f"extent=({ground.width:.3f}, {ground.height:.3f}) m"
    )
    for mode, offset in xsens_actor_offsets.items():
        print(f"[viser_player] {mode} root offset=({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}) m")
    if "g1_xsens" in xsens_actors:
        print(f"[viser_player] g1_xsens grounding correction={g1_grounding_offset_m:.3f} m")

    with server.gui.add_folder("Display"):
        robot_meshes_cb = (
            server.gui.add_checkbox("Show robot meshes", initial_value=config.show_meshes)
            if vr is not None
            else None
        )
        xsens_display_controls = {}
        if xsens_config is not None:
            for mode, actor in xsens_actors.items():
                label = "Xsens" if mode == "xsens" else "G1-proportioned Xsens"
                meshes_cb = server.gui.add_checkbox(
                    f"Show {label} meshes", initial_value=xsens_config.show_xsens_meshes
                )
                landmarks_cb = server.gui.add_checkbox(
                    f"Show {label} landmarks", initial_value=xsens_config.show_xsens_landmarks
                )
                xsens_display_controls[mode] = (actor, meshes_cb, landmarks_cb)

    if robot_meshes_cb is not None:

        @robot_meshes_cb.on_update
        def _(_evt) -> None:
            assert vr is not None
            vr.show_visual = bool(robot_meshes_cb.value)
            if vo is not None:
                vo.show_visual = bool(robot_meshes_cb.value)

    for actor, meshes_cb, landmarks_cb in xsens_display_controls.values():

        @meshes_cb.on_update
        def _(_evt, actor=actor, checkbox=meshes_cb) -> None:
            actor.set_mesh_visibility(bool(checkbox.value))

        @landmarks_cb.on_update
        def _(_evt, actor=actor, checkbox=landmarks_cb) -> None:
            actor.set_landmark_visibility(bool(checkbox.value))

    if not show_xsens:
        assert qpos is not None and vr is not None and robot_root is not None and robot_applier is not None
        print(f"[viser_player] mode=robot | {qpos.shape[0]} frames | robot_dof={robot_dof}")
        if config.record_video:
            frame_indices = build_record_frame_indices(
                n_frames=int(qpos.shape[0]),
                start_frame=config.record_start_frame,
                end_frame=config.record_end_frame,
                stride=config.record_stride,
            )
            record_viser_sequence(
                server=server,
                apply_frame=lambda frame_idx: robot_applier.apply_frame(qpos, frame_idx),
                frame_indices=frame_indices,
                output_path=config.record_path,
                width=config.record_width,
                height=config.record_height,
                fps=float(config.record_fps if config.record_fps is not None else actual_robot_fps),
                connect_timeout=config.record_connect_timeout,
                start_delay=config.record_start_delay,
                settle_time=config.record_settle_time,
                warmup_renders=config.record_warmup_renders,
                transport_format=config.record_transport_format,
            )
            if config.record_exit_after:
                raise SystemExit(0)

        create_motion_control_sliders(
            server=server,
            viser_robot=vr,
            robot_base_frame=robot_root,
            motion_sequence=qpos,
            robot_dof=robot_dof,
            viser_object=vo if config.assume_object_in_qpos else None,
            object_base_frame=object_root if config.assume_object_in_qpos else None,
            contains_object_in_qpos=config.assume_object_in_qpos,
            initial_fps=round(actual_robot_fps),
            initial_interp_mult=config.visual_fps_multiplier,
            loop=config.loop,
            frame_times_s=np.arange(qpos.shape[0], dtype=float) / actual_robot_fps,
        )
        return server

    assert xsens_sampler is not None and xsens_actors
    master_times = xsens_sampler.times_s.copy()
    if show_robot:
        assert qpos is not None and robot_applier is not None
        robot_duration_s = max(0.0, (int(qpos.shape[0]) - 1) / actual_robot_fps)
        common_duration_s = min(xsens_sampler.duration_s, robot_duration_s)
        master_times = master_times[master_times <= common_duration_s + 1e-12]
        if master_times.size == 0:
            master_times = np.array([0.0])

    def _apply_time(time_s: float) -> None:
        xsens_pose = xsens_sampler.sample(time_s)
        for mode, actor in xsens_actors.items():
            actor_positions = xsens_pose.positions_m
            if mode == "g1_xsens":
                assert g1_xsens_projector is not None
                actor_positions = g1_xsens_projector.project(
                    xsens_pose.positions_m,
                    xsens_pose.quaternions_wxyz,
                )
            actor.apply_pose(
                xsens_sampler.segment_names,
                actor_positions,
                xsens_pose.quaternions_wxyz,
            )
        if show_robot:
            assert qpos is not None and robot_applier is not None
            has_object = robot_applier.has_object_input(qpos)
            sampled_qpos = sample_qpos_at_time(
                qpos,
                time_s,
                fps=actual_robot_fps,
                robot_dof=robot_dof,
                has_object_input=has_object,
            )
            robot_applier.apply_qpos(sampled_qpos, has_object_input=has_object)

    playback_fps = _nominal_fps(master_times, config.fps)
    print(
        f"[viser_player] actors={','.join(actor_modes)} | {master_times.size} master frames | "
        f"duration={master_times[-1]:.3f}s | nominal_fps={playback_fps:.3f}"
    )
    if config.record_video:
        frame_indices = build_record_frame_indices(
            n_frames=int(master_times.size),
            start_frame=config.record_start_frame,
            end_frame=config.record_end_frame,
            stride=config.record_stride,
        )
        record_viser_sequence(
            server=server,
            apply_frame=lambda frame_idx: _apply_time(float(master_times[frame_idx])),
            frame_indices=frame_indices,
            output_path=config.record_path,
            width=config.record_width,
            height=config.record_height,
            fps=float(config.record_fps if config.record_fps is not None else playback_fps),
            connect_timeout=config.record_connect_timeout,
            start_delay=config.record_start_delay,
            settle_time=config.record_settle_time,
            warmup_renders=config.record_warmup_renders,
            transport_format=config.record_transport_format,
        )
        if config.record_exit_after:
            raise SystemExit(0)

    create_timed_motion_control_sliders(
        server,
        master_times,
        _apply_time,
        initial_fps=playback_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
    )
    return server


def main(cfg: XsensViserConfig) -> None:
    actor_modes = resolve_actor_modes(cfg.actor_modes)
    qpos: np.ndarray | None = None
    fps: int | None = None
    xsens_motion: XsensHdf5Motion | None = None
    if "robot" in actor_modes:
        qpos, fps = load_npz(cfg.qpos_npz)
    if "xsens" in actor_modes or "g1_xsens" in actor_modes:
        if cfg.xsens_hdf5 is None:
            raise ValueError(f"actor_modes={actor_modes} requires --xsens-hdf5")
        xsens_motion = load_xsens_hdf5_motion(
            resolve_package_path(cfg.xsens_hdf5),
            target_fps=cfg.xsens_target_fps,
            include_tracked_props=True,
        )
    make_player(cfg, qpos=qpos, fps=fps, xsens_motion=xsens_motion)
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(XsensViserConfig))
