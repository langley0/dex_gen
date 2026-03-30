from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from pathlib import Path

import mujoco
import numpy as np

from grasp_gen.grasp_equilibrium import mesh_scale_np
from grasp_gen.grasp_optimizer_io import GraspRunArtifact
from grasp_gen.hand import Hand
from grasp_gen.prop import Prop
from grasp_gen.prop_assets import prop_from_metadata

from .scene import build_physics_scene
from .types import MotionSpec, SamplingEvalState, default_motion_specs


FAILURE_PENALTY = 1000.0
QUAT_EPS = 1.0e-8


@dataclass(frozen=True)
class PhysicsEvalConfig:
    translation_delta: float = 0.03
    rotation_delta_deg: float = 25.0
    settle_time: float = 0.20
    motion_time: float = 0.60
    timestep: float = 0.005
    lost_contact_steps: int = 10
    max_translation_scale: float = 0.75
    max_translation_min: float = 0.05
    max_rotation_deg: float = 75.0
    squeeze_max_delta: float = 0.10
    squeeze_steps: int = 5
    object_density: float = 400.0


@dataclass(frozen=True)
class SourceGrasp:
    source_path: Path
    state_name: str
    sample_index: int
    hand_side: str
    prop_meta: dict[str, object]
    hand_pose: np.ndarray
    contact_indices: np.ndarray


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < QUAT_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _unit_quat(lhs)
    w2, x2, y2, z2 = _unit_quat(rhs)
    return _unit_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def _quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, _unit_quat(quat))
    return matrix.reshape(3, 3)


def _matrix_to_quat_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return _unit_quat(quat)


def _ortho6d_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    ortho6d = np.asarray(ortho6d, dtype=np.float64).reshape(6)
    first = ortho6d[:3]
    first /= max(float(np.linalg.norm(first)), QUAT_EPS)
    second = ortho6d[3:6] - first * float(np.dot(first, ortho6d[3:6]))
    second /= max(float(np.linalg.norm(second)), QUAT_EPS)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), QUAT_EPS)
    half = 0.5 * float(angle)
    return _unit_quat(
        np.array(
            [np.cos(half), *(np.sin(half) * axis)],
            dtype=np.float64,
        )
    )


def _quat_angle_rad(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = _quat_mul(_quat_conjugate(lhs), rhs)
    w = float(np.clip(abs(delta[0]), 0.0, 1.0))
    return 2.0 * float(np.arccos(w))


def _transform_relative(parent_pos: np.ndarray, parent_quat: np.ndarray, child_pos: np.ndarray, child_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent_rot = _quat_to_matrix_np(parent_quat)
    rel_pos = parent_rot.T @ (np.asarray(child_pos, dtype=np.float64) - np.asarray(parent_pos, dtype=np.float64))
    rel_rot = parent_rot.T @ _quat_to_matrix_np(child_quat)
    return rel_pos.astype(np.float32), _matrix_to_quat_np(rel_rot).astype(np.float32)


def _object_motion_score(translation: float, rotation_rad: float, object_scale: float) -> float:
    return float(translation / max(object_scale, 1.0e-6) + rotation_rad)


def _motion_pose(
    base_root_pos: np.ndarray,
    base_root_quat: np.ndarray,
    motion: MotionSpec,
    alpha: float,
    translation_delta: float,
    rotation_delta_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if motion.kind == "translation":
        delta = float(motion.direction) * float(translation_delta) * np.asarray(motion.axis, dtype=np.float64)
        return base_root_pos + float(alpha) * delta, base_root_quat.copy()

    delta_quat = _axis_angle_quat(np.asarray(motion.axis, dtype=np.float64), float(motion.direction) * float(alpha) * rotation_delta_rad)
    return base_root_pos.copy(), _quat_mul(base_root_quat, delta_quat)


def _squeeze_schedule(max_delta: float, steps: int) -> np.ndarray:
    if steps <= 1 or max_delta <= 0.0:
        return np.asarray([0.0], dtype=np.float32)
    return np.linspace(0.0, float(max_delta), int(steps), dtype=np.float32)


def select_source_grasp(artifact: GraspRunArtifact, *, state_name: str, index: int) -> SourceGrasp:
    if state_name not in {"best", "current"}:
        raise ValueError(f"Unsupported state_name: {state_name!r}")

    if state_name == "best":
        hand_pose = np.asarray(artifact.state.best_hand_pose, dtype=np.float32)
        contact_indices = np.asarray(artifact.state.best_contact_indices, dtype=np.int32)
        total_energy = np.asarray(artifact.state.best_energy.total, dtype=np.float32)
    else:
        hand_pose = np.asarray(artifact.state.hand_pose, dtype=np.float32)
        contact_indices = np.asarray(artifact.state.contact_indices, dtype=np.int32)
        total_energy = np.asarray(artifact.state.energy.total, dtype=np.float32)

    sample_index = int(np.argmin(total_energy)) if index < 0 else int(index)
    if not 0 <= sample_index < hand_pose.shape[0]:
        raise ValueError(f"sample index out of range: {sample_index}")

    return SourceGrasp(
        source_path=artifact.path,
        state_name=state_name,
        sample_index=sample_index,
        hand_side=str(artifact.metadata["hand"]["side"]),
        prop_meta=dict(artifact.metadata["prop"]),
        hand_pose=np.asarray(hand_pose[sample_index], dtype=np.float32),
        contact_indices=np.asarray(contact_indices[sample_index], dtype=np.int32),
    )


def _evaluate_motion(
    scene,
    motion: MotionSpec,
    *,
    qpos_target: np.ndarray,
    object_scale: float,
    config: PhysicsEvalConfig,
) -> tuple[float, float, float, float, float, int, int, bool, bool, bool, int]:
    settle_steps = max(int(round(config.settle_time / config.timestep)), 1)
    motion_steps = max(int(round(config.motion_time / config.timestep)), 1)
    max_translation_fail = max(float(config.max_translation_min), float(config.max_translation_scale) * float(object_scale))
    max_rotation_fail_rad = np.deg2rad(float(config.max_rotation_deg))
    rotation_delta_rad = np.deg2rad(float(config.rotation_delta_deg))

    scene.reset(qpos_target=qpos_target)
    lost_streak = 0
    min_contacts = 1_000_000
    final_contacts = 0
    for _ in range(settle_steps):
        scene.step(root_pos=scene.base_root_pos, root_quat=scene.base_root_quat, qpos_target=qpos_target)
        contact_count, _, _, _ = scene.contact_counts()
        min_contacts = min(min_contacts, contact_count)
        final_contacts = contact_count
        if contact_count <= 0:
            lost_streak += 1
        else:
            lost_streak = 0
        if lost_streak >= config.lost_contact_steps:
            return FAILURE_PENALTY, np.inf, np.inf, np.inf, np.inf, min_contacts, final_contacts, True, True, True, 0

    reference_pos, reference_quat = scene.object_pose()
    max_translation = 0.0
    max_rotation_rad = 0.0
    final_translation = 0.0
    final_rotation_rad = 0.0
    early_stop = False
    lost = False
    fail = False
    steps_run = 0

    for step_index in range(motion_steps):
        if motion_steps == 1:
            alpha = 1.0
        else:
            phase = float(step_index) / float(motion_steps - 1)
            alpha = 0.5 - 0.5 * cos(pi * phase)
        root_pos, root_quat = _motion_pose(
            scene.base_root_pos,
            scene.base_root_quat,
            motion,
            alpha,
            config.translation_delta,
            rotation_delta_rad,
        )
        scene.step(root_pos=root_pos, root_quat=root_quat, qpos_target=qpos_target)
        object_pos, object_quat = scene.object_pose()
        final_translation = float(np.linalg.norm(object_pos - reference_pos))
        final_rotation_rad = _quat_angle_rad(reference_quat, object_quat)
        max_translation = max(max_translation, final_translation)
        max_rotation_rad = max(max_rotation_rad, final_rotation_rad)

        contact_count, _, _, _ = scene.contact_counts()
        min_contacts = min(min_contacts, contact_count)
        final_contacts = contact_count
        if contact_count <= 0:
            lost_streak += 1
        else:
            lost_streak = 0
        steps_run = step_index + 1

        if lost_streak >= config.lost_contact_steps:
            lost = True
            early_stop = True
            break
        if max_translation > max_translation_fail or max_rotation_rad > max_rotation_fail_rad:
            fail = True
            early_stop = True
            break

    score = _object_motion_score(max_translation, max_rotation_rad, object_scale)
    if lost or fail:
        score += FAILURE_PENALTY
    return (
        score,
        max_translation,
        max_rotation_rad,
        final_translation,
        final_rotation_rad,
        min_contacts,
        final_contacts,
        lost,
        fail,
        early_stop,
        steps_run,
    )


def evaluate_source_grasp(
    source: SourceGrasp,
    *,
    config: PhysicsEvalConfig,
    motions: tuple[MotionSpec, ...] | None = None,
) -> tuple[dict[str, object], SamplingEvalState]:
    hand = Hand(source.hand_side)
    prop = prop_from_metadata(source.prop_meta)
    object_scale = float(mesh_scale_np(prop.vertices))

    hand_pose = np.asarray(source.hand_pose, dtype=np.float64).reshape(-1)
    base_root_pos = hand_pose[:3].copy()
    base_root_quat = _matrix_to_quat_np(_ortho6d_to_matrix_np(hand_pose[3:9]))
    base_qpos = hand_pose[9:].copy()

    motions = default_motion_specs() if motions is None else motions
    squeeze_deltas = _squeeze_schedule(config.squeeze_max_delta, config.squeeze_steps)
    qpos_lower = np.asarray(hand.model.jnt_range[:, 0], dtype=np.float64)
    qpos_upper = np.asarray(hand.model.jnt_range[:, 1], dtype=np.float64)
    qpos_targets = np.clip(base_qpos[None, :] + squeeze_deltas[:, None], qpos_lower[None, :], qpos_upper[None, :])

    prop_pos = np.asarray(prop.pos, dtype=np.float64)
    prop_quat = _unit_quat(np.asarray(prop.quat, dtype=np.float64))
    object_relative_pos, object_relative_quat = _transform_relative(base_root_pos, base_root_quat, prop_pos, prop_quat)

    attempt_count = len(squeeze_deltas)
    motion_count = len(motions)
    motion_scores = np.full((attempt_count, motion_count), np.nan, dtype=np.float32)
    motion_max_translation = np.full((attempt_count, motion_count), np.nan, dtype=np.float32)
    motion_max_rotation_rad = np.full((attempt_count, motion_count), np.nan, dtype=np.float32)
    motion_final_translation = np.full((attempt_count, motion_count), np.nan, dtype=np.float32)
    motion_final_rotation_rad = np.full((attempt_count, motion_count), np.nan, dtype=np.float32)
    motion_contact_min = np.full((attempt_count, motion_count), -1, dtype=np.int32)
    motion_contact_final = np.full((attempt_count, motion_count), -1, dtype=np.int32)
    motion_lost = np.zeros((attempt_count, motion_count), dtype=np.bool_)
    motion_fail = np.zeros((attempt_count, motion_count), dtype=np.bool_)
    motion_early_stop = np.zeros((attempt_count, motion_count), dtype=np.bool_)
    motion_steps = np.zeros((attempt_count, motion_count), dtype=np.int32)
    initial_contact_count = np.zeros((attempt_count,), dtype=np.int32)
    initial_penetration_count = np.zeros((attempt_count,), dtype=np.int32)
    initial_depth_sum = np.zeros((attempt_count,), dtype=np.float32)
    initial_max_depth = np.zeros((attempt_count,), dtype=np.float32)
    initial_overlap = np.zeros((attempt_count,), dtype=np.bool_)
    overall_scores = np.zeros((attempt_count,), dtype=np.float32)
    success = np.zeros((attempt_count,), dtype=np.bool_)

    for attempt_index, qpos_target in enumerate(qpos_targets):
        scene = build_physics_scene(
            hand,
            prop,
            source.prop_meta,
            root_pos=base_root_pos,
            root_quat=base_root_quat,
            qpos_target=qpos_target,
            timestep=config.timestep,
            density=config.object_density,
        )
        scene.reset(qpos_target=qpos_target)
        init_contact_count, init_penetration_count, init_depth_sum, init_max_depth = scene.contact_counts()
        initial_contact_count[attempt_index] = int(init_contact_count)
        initial_penetration_count[attempt_index] = int(init_penetration_count)
        initial_depth_sum[attempt_index] = float(init_depth_sum)
        initial_max_depth[attempt_index] = float(init_max_depth)
        if init_penetration_count > 0 or init_depth_sum > 0.0:
            initial_overlap[attempt_index] = True
            motion_fail[attempt_index, :] = True
            motion_early_stop[attempt_index, :] = True
            overall_scores[attempt_index] = float(motion_count * FAILURE_PENALTY)
            continue
        accumulated_score = 0.0
        attempt_success = True
        for motion_index, motion in enumerate(motions):
            (
                score,
                max_translation,
                max_rotation,
                final_translation,
                final_rotation,
                min_contacts,
                final_contacts,
                lost,
                fail,
                early_stop,
                steps_run,
            ) = _evaluate_motion(scene, motion, qpos_target=qpos_target, object_scale=object_scale, config=config)
            motion_scores[attempt_index, motion_index] = float(score)
            motion_max_translation[attempt_index, motion_index] = float(max_translation)
            motion_max_rotation_rad[attempt_index, motion_index] = float(max_rotation)
            motion_final_translation[attempt_index, motion_index] = float(final_translation)
            motion_final_rotation_rad[attempt_index, motion_index] = float(final_rotation)
            motion_contact_min[attempt_index, motion_index] = int(min_contacts if min_contacts != 1_000_000 else 0)
            motion_contact_final[attempt_index, motion_index] = int(final_contacts)
            motion_lost[attempt_index, motion_index] = bool(lost)
            motion_fail[attempt_index, motion_index] = bool(fail)
            motion_early_stop[attempt_index, motion_index] = bool(early_stop)
            motion_steps[attempt_index, motion_index] = int(steps_run)
            accumulated_score += float(score)
            if lost or fail:
                attempt_success = False
                break
        completed = motion_index + 1
        if attempt_success and completed == motion_count:
            success[attempt_index] = True
            overall_scores[attempt_index] = float(accumulated_score / motion_count)
        else:
            remaining = motion_count - completed
            overall_scores[attempt_index] = float(accumulated_score + remaining * FAILURE_PENALTY)

    if np.any(success):
        candidate = np.where(success, overall_scores, np.inf)
        chosen_attempt_index = int(np.argmin(candidate))
    else:
        chosen_attempt_index = int(np.argmin(overall_scores))

    metadata: dict[str, object] = {
        "source": {
            "result_path": str(source.source_path),
            "state": source.state_name,
            "sample_index": int(source.sample_index),
        },
        "hand": {"side": source.hand_side},
        "prop": dict(source.prop_meta),
        "eval": {
            "translation_delta": float(config.translation_delta),
            "rotation_delta_deg": float(config.rotation_delta_deg),
            "settle_time": float(config.settle_time),
            "motion_time": float(config.motion_time),
            "timestep": float(config.timestep),
            "lost_contact_steps": int(config.lost_contact_steps),
            "max_translation_scale": float(config.max_translation_scale),
            "max_translation_min": float(config.max_translation_min),
            "max_rotation_deg": float(config.max_rotation_deg),
            "squeeze_max_delta": float(config.squeeze_max_delta),
            "squeeze_steps": int(config.squeeze_steps),
            "object_density": float(config.object_density),
            "object_scale": float(object_scale),
        },
        "motions": [
            {
                "name": motion.name,
                "kind": motion.kind,
                "axis_name": motion.axis_name,
                "axis": np.asarray(motion.axis, dtype=float).tolist(),
                "direction": int(motion.direction),
            }
            for motion in motions
        ],
        "result": {
            "chosen_attempt_index": int(chosen_attempt_index),
            "chosen_squeeze_delta": float(squeeze_deltas[chosen_attempt_index]),
            "success_count": int(np.count_nonzero(success)),
            "best_overall_score": float(overall_scores[chosen_attempt_index]),
            "initial_overlap_count": int(np.count_nonzero(initial_overlap)),
        },
    }
    state = SamplingEvalState(
        base_hand_pose=np.asarray(source.hand_pose, dtype=np.float32),
        base_contact_indices=np.asarray(source.contact_indices, dtype=np.int32),
        object_relative_pos=np.asarray(object_relative_pos, dtype=np.float32),
        object_relative_quat=np.asarray(object_relative_quat, dtype=np.float32),
        squeeze_deltas=np.asarray(squeeze_deltas, dtype=np.float32),
        qpos_targets=np.asarray(qpos_targets, dtype=np.float32),
        initial_contact_count=initial_contact_count,
        initial_penetration_count=initial_penetration_count,
        initial_depth_sum=initial_depth_sum,
        initial_max_depth=initial_max_depth,
        initial_overlap=initial_overlap,
        motion_scores=motion_scores,
        motion_max_translation=motion_max_translation,
        motion_max_rotation_rad=motion_max_rotation_rad,
        motion_final_translation=motion_final_translation,
        motion_final_rotation_rad=motion_final_rotation_rad,
        motion_contact_min=motion_contact_min,
        motion_contact_final=motion_contact_final,
        motion_lost=motion_lost,
        motion_fail=motion_fail,
        motion_early_stop=motion_early_stop,
        motion_steps=motion_steps,
        overall_scores=overall_scores,
        success=success,
        chosen_attempt_index=chosen_attempt_index,
    )
    return metadata, state
