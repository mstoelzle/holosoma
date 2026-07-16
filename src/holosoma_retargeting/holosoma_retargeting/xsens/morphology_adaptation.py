"""XSens naming and grounding adapters for generic morphology transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from holosoma_retargeting.kinematics import (
    GroundingSurface,
    KinematicMorphologyAdapter,
    KinematicTree,
    LowestSurfaceGrounding,
)
from holosoma_retargeting.xsens.kinematic_model import normalize_xsens_name

XsensGroundingMode = Literal["none", "match_lowest_soles"]
_SOLE_SOURCE_BODY_NAMES = ("LeftFoot", "LeftToe", "RightFoot", "RightToe")


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


def xsens_body_to_source_mapping(
    model: KinematicTree,
    source_segment_names: Sequence[str],
) -> Mapping[str, str]:
    """Return the validated model-body to source-segment mapping."""

    return _body_to_source_mapping(model, tuple(source_segment_names), model_label="Model")


__all__ = ["XsensGroundingMode", "build_xsens_morphology_adapter", "xsens_body_to_source_mapping"]
