"""XSens naming and grounding adapters for generic morphology transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XsensHdf5Motion,
    XsensHdf5Tpose,
    load_xsens_hdf5_calibration,
    load_xsens_hdf5_tpose,
)
from holosoma_retargeting.kinematics import (
    GroundingSurface,
    KinematicMorphologyAdapter,
    KinematicMotion,
    KinematicPose,
    KinematicTree,
    LowestSurfaceGrounding,
    with_body_attachments,
)
from holosoma_retargeting.xsens.g1_kinematic_reduction import (
    G1XsensReductionConfig,
    build_g1_proportioned_xsens_tree,
    extract_g1_anthropometry,
)
from holosoma_retargeting.xsens.geometry_attachments import build_xsens_avatar_mesh_attachments
from holosoma_retargeting.xsens.kinematic_model import build_xsens_kinematic_tree, normalize_xsens_name

XsensGroundingMode = Literal["none", "match_lowest_soles"]
_SOLE_SOURCE_BODY_NAMES = ("LeftFoot", "LeftToe", "RightFoot", "RightToe")


@dataclass(frozen=True)
class PreparedG1XsensMorphology:
    """Reusable subject/target models and adapter for G1-proportioned Xsens poses."""

    source_model: KinematicTree
    target_model: KinematicTree
    adapter: KinematicMorphologyAdapter


def _body_to_source_mapping(
    model: KinematicTree,
    source_body_names: tuple[str, ...],
    *,
    model_label: str,
) -> dict[str, str]:
    normalized_sources: dict[str, str] = {}
    for name in source_body_names:
        normalized = normalize_xsens_name(name)
        if normalized in normalized_sources:
            raise ValueError(f"XSens source body names contain a duplicate semantic name '{name}'")
        normalized_sources[normalized] = name

    model_source_names = tuple(str(body.metadata.get("xsens:sourceSegmentName", body.name)) for body in model.bodies)
    normalized_model_order = tuple(normalize_xsens_name(name) for name in model_source_names)
    normalized_source_order = tuple(normalize_xsens_name(name) for name in source_body_names)
    if normalized_model_order != normalized_source_order:
        raise ValueError(
            f"{model_label} XSens semantic body order does not match the source motion: "
            f"model={model_source_names}, source={source_body_names}"
        )

    return {
        body.name: normalized_sources[normalize_xsens_name(source_name)]
        for body, source_name in zip(model.bodies, model_source_names, strict=True)
    }


def _outsole_surfaces(model: KinematicTree) -> tuple[GroundingSurface, ...]:
    source_to_body = {
        normalize_xsens_name(str(body.metadata.get("xsens:sourceSegmentName", body.name))): body
        for body in model.bodies
    }
    surfaces: list[GroundingSurface] = []
    for source_name in _SOLE_SOURCE_BODY_NAMES:
        body = source_to_body.get(normalize_xsens_name(source_name))
        if body is None:
            raise KeyError(f"XSens grounding model is missing body '{source_name}'")
        mesh_names = tuple(mesh.name for mesh in body.meshes if "outsole" in mesh.name.lower())
        if not mesh_names:
            raise ValueError(f"XSens grounding body '{body.name}' has no outsole mesh")
        surfaces.append(GroundingSurface(body.name, mesh_names))
    return tuple(surfaces)


def build_xsens_morphology_adapter(
    target_model: KinematicTree,
    source_segment_names: Sequence[str],
    *,
    source_model: KinematicTree | None = None,
    grounding: XsensGroundingMode = "none",
) -> KinematicMorphologyAdapter:
    """Build a generic adapter using XSens semantic body metadata and order."""

    source_names = tuple(source_segment_names)
    target_mapping = _body_to_source_mapping(target_model, source_names, model_label="Target")
    grounding_policy = None
    if grounding == "match_lowest_soles":
        if source_model is None:
            raise ValueError("XSens lowest-sole grounding requires a source model")
        source_mapping = _body_to_source_mapping(source_model, source_names, model_label="Source")
        grounding_policy = LowestSurfaceGrounding(
            source_model,
            target_model,
            source_names,
            source_body_to_pose_body=source_mapping,
            target_body_to_pose_body=target_mapping,
            source_surfaces=_outsole_surfaces(source_model),
            target_surfaces=_outsole_surfaces(target_model),
        )
    elif grounding != "none":
        raise ValueError(f"Unknown XSens grounding mode '{grounding}'")

    return KinematicMorphologyAdapter(
        target_model,
        source_names,
        target_body_to_source_body=target_mapping,
        grounding=grounding_policy,
    )


def build_subject_xsens_reference_model(
    hdf5_path: str | Path,
    *,
    include_tennis_racket: bool = True,
) -> KinematicTree:
    """Build the calibrated subject model and visuals needed for grounding."""

    calibration = load_xsens_hdf5_calibration(Path(hdf5_path).expanduser())
    model = build_xsens_kinematic_tree(
        calibration,
        include_tennis_racket=include_tennis_racket,
    )
    body_names = {body.name for body in model.bodies}
    meshes = {
        name: attachments
        for name, attachments in build_xsens_avatar_mesh_attachments(calibration).items()
        if name in body_names
    }
    return with_body_attachments(model, meshes=meshes)


def prepare_g1_xsens_morphology(
    source_segment_names: Sequence[str],
    *,
    hdf5_path: str | Path,
    g1_model_path: str | Path | None = None,
    grounding: XsensGroundingMode = "match_lowest_soles",
    preserve_joint_offsets: bool = False,
) -> PreparedG1XsensMorphology:
    """Build the direct subject-to-G1 Xsens morphology path used by retargeting.

    The resulting adapter performs no IK or optimization: it preserves every
    source segment orientation (and thus every relative joint rotation) while
    reconstructing body origins from the G1-sized tree's joint anchors.
    """

    source_names = tuple(source_segment_names)
    source_model = build_subject_xsens_reference_model(
        hdf5_path,
        include_tennis_racket=False,
    )
    target_model = build_g1_proportioned_xsens_tree(
        extract_g1_anthropometry(g1_model_path),
        G1XsensReductionConfig(
            preserve_joint_offsets=preserve_joint_offsets,
            include_visuals=True,
            include_tennis_racket=False,
        ),
    )
    adapter = build_xsens_morphology_adapter(
        target_model,
        source_names,
        source_model=source_model if grounding == "match_lowest_soles" else None,
        grounding=grounding,
    )
    return PreparedG1XsensMorphology(source_model, target_model, adapter)


def adapt_xsens_motion_to_g1(
    motion: XsensHdf5Motion,
    *,
    hdf5_path: str | Path,
    g1_model_path: str | Path | None = None,
    grounding: XsensGroundingMode = "match_lowest_soles",
    preserve_joint_offsets: bool = False,
) -> KinematicMotion:
    """Reconstruct one body-only Xsens motion using G1-derived proportions."""

    prepared = prepare_g1_xsens_morphology(
        motion.segment_names,
        hdf5_path=hdf5_path,
        g1_model_path=g1_model_path,
        grounding=grounding,
        preserve_joint_offsets=preserve_joint_offsets,
    )
    return prepared.adapter.adapt_motion(
        KinematicMotion(
            tuple(motion.segment_names),
            motion.positions_m,
            motion.quaternions_wijk,
            motion.times_s,
        )
    )


def adapt_xsens_tpose_to_g1(
    *,
    hdf5_path: str | Path,
    g1_model_path: str | Path | None = None,
    grounding: XsensGroundingMode = "match_lowest_soles",
    preserve_joint_offsets: bool = False,
    variant: str = "Tpose",
) -> XsensHdf5Tpose:
    """Reconstruct a recorded Xsens T-pose with G1-derived proportions.

    Only positions are morphology-adapted. The recorded global segment
    orientations remain byte-for-byte identical so the sensor-frame side of
    orientation-offset calibration is unchanged.
    """

    tpose = load_xsens_hdf5_tpose(hdf5_path, variant=variant)
    prepared = prepare_g1_xsens_morphology(
        tpose.segment_names,
        hdf5_path=hdf5_path,
        g1_model_path=g1_model_path,
        grounding=grounding,
        preserve_joint_offsets=preserve_joint_offsets,
    )
    adapted = prepared.adapter.adapt_pose(
        KinematicPose(
            tuple(tpose.segment_names),
            tpose.positions_m,
            tpose.quaternions_wijk,
        )
    )
    return XsensHdf5Tpose(
        positions_m=adapted.positions_m,
        quaternions_wijk=adapted.orientations_wxyz,
        variant=f"G1Proportioned{variant}",
        segment_names=list(tpose.segment_names),
        source_indices=list(tpose.source_indices),
    )


def xsens_body_to_source_mapping(
    model: KinematicTree,
    source_segment_names: Sequence[str],
) -> Mapping[str, str]:
    """Return the validated model-body to source-segment mapping."""

    return _body_to_source_mapping(model, tuple(source_segment_names), model_label="Model")


__all__ = [
    "PreparedG1XsensMorphology",
    "XsensGroundingMode",
    "adapt_xsens_motion_to_g1",
    "adapt_xsens_tpose_to_g1",
    "build_subject_xsens_reference_model",
    "build_xsens_morphology_adapter",
    "prepare_g1_xsens_morphology",
    "xsens_body_to_source_mapping",
]
