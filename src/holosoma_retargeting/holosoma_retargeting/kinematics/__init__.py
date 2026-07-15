"""Backend-independent rigid-body kinematic model definitions."""

from .model import (
    KinematicTree,
    MeshAttachment,
    PointSetAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
    ValidationReport,
    compute_joint_positions,
    compute_reference_joint_positions,
    validate_kinematic_tree,
    with_body_attachments,
)

__all__ = [
    "KinematicTree",
    "MeshAttachment",
    "PointSetAttachment",
    "RigidBodyDefinition",
    "SphericalJointDefinition",
    "Transform",
    "ValidationReport",
    "compute_joint_positions",
    "compute_reference_joint_positions",
    "validate_kinematic_tree",
    "with_body_attachments",
]
