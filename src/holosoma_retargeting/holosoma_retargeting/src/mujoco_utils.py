from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

Pair = Tuple[str, str]


@dataclass(frozen=True)
class MujocoFramePoseSet:
    """World poses of named MuJoCo bodies or geometries."""

    names: tuple[str, ...]
    positions_m: np.ndarray
    quaternions_wxyz: np.ndarray


def mujoco_frame_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a named MuJoCo body/geometry world position and rotation matrix."""

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id >= 0:
        return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, frame_name)
    if geom_id >= 0:
        return data.geom_xpos[geom_id].copy(), data.geom_xmat[geom_id].reshape(3, 3).copy()
    raise KeyError(f"No MuJoCo body or geom named '{frame_name}'")


def evaluate_mujoco_frame_poses(
    model_path: str | Path,
    qpos: np.ndarray,
    frame_names: Sequence[str],
) -> MujocoFramePoseSet:
    """Evaluate named frame world poses at one configuration of any MuJoCo model."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    values = np.asarray(qpos, dtype=float).reshape(-1)
    if values.shape != (model.nq,):
        raise ValueError(f"Expected qpos with shape ({model.nq},), got {values.shape}")

    data = mujoco.MjData(model)
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
    positions = []
    quaternions = []
    for frame_name in frame_names:
        position, rotation = mujoco_frame_pose(model, data, frame_name)
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        positions.append(position)
        quaternions.append(quaternion_xyzw[[3, 0, 1, 2]])
    return MujocoFramePoseSet(
        names=tuple(frame_names),
        positions_m=np.asarray(positions, dtype=float).reshape(-1, 3),
        quaternions_wxyz=np.asarray(quaternions, dtype=float).reshape(-1, 4),
    )


def replace_named_joint_qpos(
    model_path: str | Path,
    qpos: np.ndarray,
    joint_values: Mapping[str, float],
) -> np.ndarray:
    """Return a qpos copy with selected scalar joints replaced by name."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    result = np.asarray(qpos, dtype=float).reshape(-1).copy()
    if result.shape != (model.nq,):
        raise ValueError(f"Expected qpos with shape ({model.nq},), got {result.shape}")
    for joint_name, value in joint_values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"No MuJoCo joint named '{joint_name}'")
        if int(model.jnt_type[joint_id]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint '{joint_name}' is not scalar")
        result[int(model.jnt_qposadr[joint_id])] = float(value)
    return result


def _mesh_local_vf(model, geom_id):
    """Return local vertices and faces for a MuJoCo mesh geom."""
    mesh_id = int(model.geom_dataid[geom_id])  # Note: sometime geom does not have mesh, mesh_id will be -1

    v0, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    f0, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])

    V = model.mesh_vert[v0 : v0 + nv].astype(np.float64, copy=True)

    F = model.mesh_face[f0 : f0 + nf].astype(np.int32, copy=True)

    return V, F


def _to_world(v_local, data, geom_id):
    """Transform local vertices to world using geom pose."""
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]

    return v_local @ R.T + t


def _world_mesh_from_geom(model, data, geom_id, geom_name):
    V_local, F = _mesh_local_vf(model, geom_id)

    V_world = _to_world(V_local, data, geom_id)

    return V_world, F
