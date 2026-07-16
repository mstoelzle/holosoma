#!/usr/bin/env python3
"""Render the human-subject Xsens avatar, G1 Xsens avatar, and G1 side-by-side in Viser.

The source avatar is loaded from ``--calibrated-xsens-usd-path`` when supplied.
Otherwise the script exports a calibrated USD from ``--hdf5-path`` first.  The
physical G1 T-pose is solved from the same recording so all three columns use
the Xsens T-pose convention.

Example:
    python examples/xsens_tennis/compare_xsens_g1_tpose.py \
        --hdf5-path demo_data/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02.hdf5
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.kinematics import KinematicTree  # noqa: E402
from holosoma_retargeting.usd import open_usd_stage, read_kinematic_tree_from_stage  # noqa: E402
from holosoma_retargeting.xsens.g1_kinematic_reduction import (  # noqa: E402
    G1XsensReductionConfig,
    build_g1_proportioned_xsens_tree,
    extract_g1_anthropometry,
)
from holosoma_retargeting.xsens.tpose_calibration import (  # noqa: E402
    XsensTposeCalibrationConfig,
    solve_xsens_tpose_calibration,
)
from holosoma_retargeting.xsens.usd_conversion import convert_xsens_hdf5_to_usd  # noqa: E402


@dataclass(frozen=True)
class XsensG1TposeComparisonConfig:
    """Inputs and display options for the three-model comparison."""

    hdf5_path: Path = Path("demo_data/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02.hdf5")
    """Xsens recording providing the calibrated avatar and T-pose targets."""

    calibrated_xsens_usd_path: Path | None = None
    """Existing human-subject Xsens avatar USD; generated from hdf5_path when omitted."""

    generated_usd_dir: Path = Path("demo_results/g1/models/xsens")
    """Destination for the generated human-subject Xsens avatar USD."""

    g1_urdf_path: Path = Path("models/g1/g1_29dof.urdf")
    """G1 model rendered by Viser and used by the T-pose calibration."""

    preserve_joint_offsets: bool = False
    """Preserve compound G1 offsets in the G1-proportioned Xsens avatar."""

    include_tennis_racket: bool = False
    """Include the calibrated recording's tracked tennis-racket segment."""

    spacing_m: float = 2.0
    """Center-to-center lateral spacing between columns."""

    port: int = 8080
    """Viser server port."""

    tpose_max_nfev: int = 400
    """Maximum evaluations for the physical G1 T-pose calibration."""

    default_human_height: float = 1.78
    """Human height used by the existing Xsens-to-G1 T-pose calibration."""


@dataclass(frozen=True)
class ComparisonAssets:
    calibrated_xsens_model: KinematicTree
    g1_xsens_model: KinematicTree
    calibrated_xsens_usd_path: Path
    g1_urdf_path: Path
    g1_qpos: np.ndarray
    g1_solver_success: bool
    g1_solver_cost: float


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_existing_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    package_relative = _package_root() / path
    if package_relative.is_file():
        return package_relative.resolve()
    raise FileNotFoundError(path)


def _resolve_g1_xml(urdf_path: Path) -> Path:
    sibling_xml = urdf_path.with_suffix(".xml")
    return sibling_xml if sibling_xml.is_file() else urdf_path


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def tree_vertical_bounds(model: KinematicTree) -> tuple[float, float]:
    """Return reference-pose bounds including local render meshes."""

    z_values: list[float] = []
    for body in model.bodies:
        pose = body.reference_pose
        rotation = _quat_matrix(pose.rotation_wxyz)
        if body.meshes:
            for mesh in body.meshes:
                world_vertices = np.asarray(mesh.vertices_m, dtype=float) @ rotation.T + pose.translation_m
                z_values.extend((float(world_vertices[:, 2].min()), float(world_vertices[:, 2].max())))
        else:
            z_values.append(float(pose.translation_m[2]))
    if not z_values:
        raise ValueError("Cannot compute bounds for an empty kinematic tree")
    return min(z_values), max(z_values)


def side_by_side_offsets(spacing_m: float) -> tuple[float, float, float]:
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    return (-float(spacing_m), 0.0, float(spacing_m))


def _calibrated_usd_path(config: XsensG1TposeComparisonConfig, hdf5_path: Path) -> Path:
    if config.calibrated_xsens_usd_path is not None:
        return _resolve_existing_path(config.calibrated_xsens_usd_path)
    output_dir = config.generated_usd_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hdf5_path.stem}_xsens_model.usda"
    convert_xsens_hdf5_to_usd(
        hdf5_path,
        output_path,
        include_visuals=True,
        include_landmarks=False,
        include_tennis_racket=config.include_tennis_racket,
    )
    return output_path.resolve()


def prepare_comparison_assets(config: XsensG1TposeComparisonConfig) -> ComparisonAssets:
    """Load/generate all three comparison assets without starting Viser."""

    hdf5_path = _resolve_existing_path(config.hdf5_path)
    g1_urdf_path = _resolve_existing_path(config.g1_urdf_path)
    calibrated_usd_path = _calibrated_usd_path(config, hdf5_path)
    calibrated_xsens_model = read_kinematic_tree_from_stage(open_usd_stage(calibrated_usd_path))

    anthropometry = extract_g1_anthropometry(_resolve_g1_xml(g1_urdf_path))
    g1_xsens_model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(
            preserve_joint_offsets=config.preserve_joint_offsets,
            include_visuals=True,
        ),
    )

    calibration = solve_xsens_tpose_calibration(
        hdf5_path,
        config=XsensTposeCalibrationConfig(
            robot_type="g1",
            variant="Tpose",
            robot_urdf_file=str(g1_urdf_path),
            default_human_height=config.default_human_height,
            max_nfev=config.tpose_max_nfev,
            verbose=0,
        ),
    )
    return ComparisonAssets(
        calibrated_xsens_model=calibrated_xsens_model,
        g1_xsens_model=g1_xsens_model,
        calibrated_xsens_usd_path=calibrated_usd_path,
        g1_urdf_path=g1_urdf_path,
        g1_qpos=np.asarray(calibration.qpos[0], dtype=float),
        g1_solver_success=calibration.solver_success,
        g1_solver_cost=calibration.solver_cost,
    )


def _add_kinematic_tree(
    server: viser.ViserServer,
    *,
    root_path: str,
    model: KinematicTree,
    lateral_offset_m: float,
    include_tennis_racket: bool = False,
) -> tuple[viser.FrameHandle, list[viser.FrameHandle], float]:
    minimum_z, maximum_z = tree_vertical_bounds(model)
    root = server.scene.add_frame(
        root_path,
        show_axes=False,
        position=np.array([0.0, lateral_offset_m, -minimum_z]),
    )
    axes: list[viser.FrameHandle] = []
    for body in model.bodies:
        if body.name == "TennisRacket" and not include_tennis_racket:
            continue
        body_path = f"{root_path}/bodies/{body.name}"
        server.scene.add_frame(
            body_path,
            show_axes=False,
            position=body.reference_pose.translation_m,
            wxyz=body.reference_pose.rotation_wxyz,
        )
        axes.append(
            server.scene.add_frame(
                f"{body_path}/axes",
                show_axes=True,
                axes_length=0.07,
                axes_radius=0.0025,
                visible=False,
            )
        )
        for mesh in body.meshes:
            server.scene.add_mesh_simple(
                f"{body_path}/{mesh.name}",
                vertices=np.asarray(mesh.vertices_m, dtype=float),
                faces=np.asarray(mesh.faces, dtype=np.int64),
                color=mesh.color_rgb,
                material="toon5",
                flat_shading=False,
                side="double",
            )
    return root, axes, maximum_z - minimum_z


def make_comparison_viewer(
    config: XsensG1TposeComparisonConfig,
    assets: ComparisonAssets | None = None,
) -> viser.ViserServer:
    assets = assets or prepare_comparison_assets(config)
    calibrated_offset, generated_offset, robot_offset = side_by_side_offsets(config.spacing_m)

    server = viser.ViserServer(port=config.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=5.0,
        height=3.0 * config.spacing_m + 2.0,
        position=(0.0, 0.0, 0.0),
    )

    calibrated_root, calibrated_axes, calibrated_height = _add_kinematic_tree(
        server,
        root_path="/comparison/calibrated_xsens",
        model=assets.calibrated_xsens_model,
        lateral_offset_m=calibrated_offset,
        include_tennis_racket=config.include_tennis_racket,
    )
    generated_root, generated_axes, generated_height = _add_kinematic_tree(
        server,
        root_path="/comparison/g1_xsens",
        model=assets.g1_xsens_model,
        lateral_offset_m=generated_offset,
    )

    g1_qpos = assets.g1_qpos
    robot_root = server.scene.add_frame(
        "/comparison/actual_g1",
        show_axes=False,
        position=np.array([0.0, robot_offset, g1_qpos[2]]),
        wxyz=g1_qpos[3:7],
    )
    robot_axes = server.scene.add_frame(
        "/comparison/actual_g1/axes",
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.003,
        visible=False,
    )
    robot_urdf = yourdfpy.URDF.load(
        str(assets.g1_urdf_path),
        load_meshes=True,
        build_scene_graph=True,
    )
    viser_robot = ViserUrdf(
        server,
        urdf_or_path=robot_urdf,
        root_node_name="/comparison/actual_g1",
    )
    robot_dof = len(viser_robot.get_actuated_joint_limits())
    viser_robot.update_cfg(g1_qpos[7 : 7 + robot_dof])

    label_height = max(calibrated_height, generated_height, 1.5) + 0.12
    label_specs = (
        ("calibrated_xsens", "Human-subject Xsens avatar", calibrated_offset),
        (
            "g1_xsens",
            f"G1 Xsens avatar (offsets {'on' if config.preserve_joint_offsets else 'off'})",
            generated_offset,
        ),
        ("actual_g1", "Actual G1", robot_offset),
    )
    for name, text, lateral_offset in label_specs:
        server.scene.add_label(
            f"/labels/{name}",
            text,
            position=(0.0, lateral_offset, label_height),
            anchor="bottom-center",
            font_size_mode="scene",
            font_scene_height=0.075,
        )

    with server.gui.add_folder("Models"):
        show_calibrated = server.gui.add_checkbox("Human-subject Xsens avatar", initial_value=True)
        show_generated = server.gui.add_checkbox("G1 Xsens avatar", initial_value=True)
        show_robot = server.gui.add_checkbox("Actual G1", initial_value=True)
        show_axes = server.gui.add_checkbox("Show frames", initial_value=False)

    @show_calibrated.on_update
    def _(_) -> None:
        calibrated_root.visible = bool(show_calibrated.value)

    @show_generated.on_update
    def _(_) -> None:
        generated_root.visible = bool(show_generated.value)

    @show_robot.on_update
    def _(_) -> None:
        viser_robot.show_visual = bool(show_robot.value)
        robot_root.visible = bool(show_robot.value)

    @show_axes.on_update
    def _(_) -> None:
        visible = bool(show_axes.value)
        for handle in (*calibrated_axes, *generated_axes, robot_axes):
            handle.visible = visible

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (5.2, 0.0, 1.55)
        client.camera.look_at = (0.0, 0.0, 0.78)
        client.camera.up = (0.0, 0.0, 1.0)

    print(f"[compare_xsens_g1_tpose] Human-subject Xsens avatar USD: {assets.calibrated_xsens_usd_path}")
    print(
        f"[compare_xsens_g1_tpose] G1 T-pose solver_success={assets.g1_solver_success} cost={assets.g1_solver_cost:.4f}"
    )
    print("[compare_xsens_g1_tpose] Column order: human-subject Xsens avatar | G1 Xsens avatar | actual G1")
    return server


def main(config: XsensG1TposeComparisonConfig) -> None:
    assets = prepare_comparison_assets(config)
    make_comparison_viewer(config, assets)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(XsensG1TposeComparisonConfig))
