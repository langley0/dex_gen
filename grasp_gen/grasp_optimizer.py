from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .grasp_optimizer_state import GraspBatchEnergy, GraspBatchSnapshot, GraspBatchState


TEMPERATURE_EPS = 1.0e-8
BEST_ENERGY_EPS = 1.0e-12


class GraspEnergyModelLike(Protocol):
    pose_dim: int
    point_count: int

    def project(self, hand_pose: jax.Array) -> jax.Array: ...

    def energy(self, hand_pose: jax.Array, contact_indices: jax.Array) -> GraspBatchEnergy: ...


@dataclass(frozen=True)
class GraspBatchOptimizerConfig:
    switch_possibility: float = 0.5
    starting_temperature: float = 18.0
    temperature_decay: float = 0.95
    annealing_period: int = 30
    step_size: float = 0.005
    step_size_period: int = 50
    mu: float = 0.98
    rms_epsilon: float = 1.0e-6


def _seed_to_key(seed: int) -> jax.Array:
    return jax.random.key(np.uint32(int(seed) % (2**32)))


def _scheduled_value(base: jax.Array, decay: jax.Array, period: int, step_index: jax.Array) -> jax.Array:
    power = jnp.floor_divide(step_index, period).astype(jnp.float32)
    return base * jnp.power(decay, power)


class GraspBatchOptimizer:
    def __init__(
        self,
        energy_model: GraspEnergyModelLike,
        *,
        contact_count: int,
        config: GraspBatchOptimizerConfig | None = None,
    ):
        self.energy_model = energy_model
        self.contact_count = int(contact_count)
        self.config = GraspBatchOptimizerConfig() if config is None else config
        self.pose_dim = int(energy_model.pose_dim)
        self.point_count = int(energy_model.point_count)
        self._project = jax.jit(energy_model.project)
        self._energy = jax.jit(energy_model.energy)

        self._validate_init()
        self._energy_only, self._energy_and_grad = self._make_energy_functions()
        self._step_fn = self._make_step_function()
        self._run_many_fn = self._make_run_many_function()

    def init_state(
        self,
        initial_hand_pose: np.ndarray | jax.Array,
        *,
        seed: int = 0,
        initial_contact_indices: np.ndarray | jax.Array | None = None,
    ) -> GraspBatchState:
        hand_pose = jnp.asarray(initial_hand_pose, dtype=jnp.float32)
        if hand_pose.ndim != 2 or hand_pose.shape[1] != self.pose_dim:
            raise ValueError(f"initial_hand_pose must have shape (batch, {self.pose_dim}), got {tuple(hand_pose.shape)}")

        batch_size = int(hand_pose.shape[0])
        rng_key = _seed_to_key(seed)
        rng_key, contact_key = jax.random.split(rng_key)
        contact_indices = self._init_contact_indices(batch_size, contact_key, initial_contact_indices)

        energy, projected_hand_pose = self._energy_only(hand_pose, contact_indices)
        return GraspBatchState(
            hand_pose=projected_hand_pose,
            contact_indices=contact_indices,
            energy=energy,
            best_hand_pose=projected_hand_pose,
            best_contact_indices=contact_indices,
            best_energy=energy,
            ema_grad=jnp.zeros((self.pose_dim,), dtype=jnp.float32),
            accepted_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            rejected_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            step_index=jnp.asarray(0, dtype=jnp.int32),
            rng_key=rng_key,
        )

    def step(self, state: GraspBatchState) -> tuple[GraspBatchState, GraspBatchSnapshot]:
        return self._step_fn(state)

    def run(
        self,
        state: GraspBatchState,
        steps: int,
    ) -> tuple[GraspBatchState, list[GraspBatchSnapshot]]:
        if steps < 0:
            raise ValueError("steps must be non-negative.")

        snapshots: list[GraspBatchSnapshot] = []
        next_state = state
        for _ in range(int(steps)):
            next_state, snapshot = self.step(next_state)
            snapshots.append(snapshot)
        return next_state, snapshots

    def run_many(self, state: GraspBatchState, steps: int) -> GraspBatchState:
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if steps == 0:
            return state
        return self._run_many_fn(state, steps=int(steps))

    def temperature(self, step_index: int) -> float:
        value = _scheduled_value(
            jnp.asarray(self.config.starting_temperature, dtype=jnp.float32),
            jnp.asarray(self.config.temperature_decay, dtype=jnp.float32),
            int(self.config.annealing_period),
            jnp.asarray(step_index, dtype=jnp.int32),
        )
        return float(value)

    def step_size(self, step_index: int) -> float:
        value = _scheduled_value(
            jnp.asarray(self.config.step_size, dtype=jnp.float32),
            jnp.asarray(self.config.temperature_decay, dtype=jnp.float32),
            int(self.config.step_size_period),
            jnp.asarray(step_index, dtype=jnp.int32),
        )
        return float(value)

    def _validate_init(self) -> None:
        if self.pose_dim <= 0:
            raise ValueError("energy_model.pose_dim must be positive.")
        if self.point_count <= 0:
            raise ValueError("energy_model.point_count must be positive.")
        if self.contact_count <= 0:
            raise ValueError("contact_count must be positive.")
        if self.config.annealing_period <= 0:
            raise ValueError("config.annealing_period must be positive.")
        if self.config.step_size_period <= 0:
            raise ValueError("config.step_size_period must be positive.")
        if self.config.step_size <= 0.0:
            raise ValueError("config.step_size must be positive.")
        if self.config.starting_temperature <= 0.0:
            raise ValueError("config.starting_temperature must be positive.")
        if not 0.0 < self.config.temperature_decay <= 1.0:
            raise ValueError("config.temperature_decay must be in (0, 1].")
        if not 0.0 <= self.config.switch_possibility <= 1.0:
            raise ValueError("config.switch_possibility must be in [0, 1].")
        if not 0.0 <= self.config.mu < 1.0:
            raise ValueError("config.mu must be in [0, 1).")
        if self.config.rms_epsilon <= 0.0:
            raise ValueError("config.rms_epsilon must be positive.")

    def _init_contact_indices(
        self,
        batch_size: int,
        rng_key: jax.Array,
        initial_contact_indices: np.ndarray | jax.Array | None,
    ) -> jax.Array:
        if initial_contact_indices is None:
            return jax.random.randint(
                rng_key,
                shape=(batch_size, self.contact_count),
                minval=0,
                maxval=self.point_count,
            )

        contact_indices = jnp.asarray(initial_contact_indices, dtype=jnp.int32)
        if contact_indices.shape != (batch_size, self.contact_count):
            raise ValueError(
                f"initial_contact_indices must have shape ({batch_size}, {self.contact_count}), "
                f"got {tuple(contact_indices.shape)}"
            )
        return jnp.clip(contact_indices, 0, self.point_count - 1)

    def _make_energy_functions(self):
        def energy_only(hand_pose: jax.Array, contact_indices: jax.Array) -> tuple[GraspBatchEnergy, jax.Array]:
            projected_hand_pose = self._project(hand_pose)
            energy = self._energy(projected_hand_pose, contact_indices)
            return energy, projected_hand_pose

        def total_energy_and_aux(hand_pose: jax.Array, contact_indices: jax.Array):
            energy, projected_hand_pose = energy_only(hand_pose, contact_indices)
            return jnp.sum(energy.total), (energy, projected_hand_pose)

        return (
            jax.jit(energy_only),
            jax.jit(jax.value_and_grad(total_energy_and_aux, argnums=0, has_aux=True)),
        )

    def _make_step_function(self):
        step_size_base = jnp.asarray(self.config.step_size, dtype=jnp.float32)
        temperature_base = jnp.asarray(self.config.starting_temperature, dtype=jnp.float32)
        decay = jnp.asarray(self.config.temperature_decay, dtype=jnp.float32)
        mu = jnp.asarray(self.config.mu, dtype=jnp.float32)
        switch_possibility = jnp.asarray(self.config.switch_possibility, dtype=jnp.float32)
        rms_epsilon = jnp.asarray(self.config.rms_epsilon, dtype=jnp.float32)
        annealing_period = int(self.config.annealing_period)
        step_size_period = int(self.config.step_size_period)

        @jax.jit
        def step_fn(state: GraspBatchState) -> tuple[GraspBatchState, GraspBatchSnapshot]:
            step_index = state.step_index
            batch_size = state.hand_pose.shape[0]

            step_size = _scheduled_value(step_size_base, decay, step_size_period, step_index)
            temperature = _scheduled_value(temperature_base, decay, annealing_period, step_index)

            (_, (current_energy, current_hand_pose)), gradient = self._energy_and_grad(
                state.hand_pose,
                state.contact_indices,
            )

            mean_square_grad = jnp.mean(jnp.square(gradient), axis=0)
            ema_grad = mu * state.ema_grad + (1.0 - mu) * mean_square_grad
            proposed_hand_pose = current_hand_pose - step_size * gradient / (jnp.sqrt(ema_grad)[None, :] + rms_epsilon)

            next_rng_key, switch_key, index_key, accept_key = jax.random.split(state.rng_key, 4)
            switch_mask = jax.random.uniform(switch_key, shape=(batch_size, self.contact_count)) < switch_possibility
            sampled_contact_indices = jax.random.randint(
                index_key,
                shape=(batch_size, self.contact_count),
                minval=0,
                maxval=self.point_count,
            )
            proposed_contact_indices = jnp.where(switch_mask, sampled_contact_indices, state.contact_indices)
            proposed_energy, proposed_hand_pose = self._energy_only(proposed_hand_pose, proposed_contact_indices)

            alpha = jax.random.uniform(accept_key, shape=(batch_size,))
            accept = alpha < jnp.exp(
                (current_energy.total - proposed_energy.total) / jnp.maximum(temperature, TEMPERATURE_EPS)
            )
            accept_mask = accept[:, None]

            next_hand_pose = jnp.where(accept_mask, proposed_hand_pose, current_hand_pose)
            next_contact_indices = jnp.where(accept_mask, proposed_contact_indices, state.contact_indices)
            next_energy = GraspBatchEnergy(
                total=jnp.where(accept, proposed_energy.total, current_energy.total),
                distance=jnp.where(accept, proposed_energy.distance, current_energy.distance),
                penetration=jnp.where(accept, proposed_energy.penetration, current_energy.penetration),
                penetration_depth=jnp.where(accept, proposed_energy.penetration_depth, current_energy.penetration_depth),
                selected_penetration=jnp.where(
                    accept,
                    proposed_energy.selected_penetration,
                    current_energy.selected_penetration,
                ),
            )

            is_new_best = next_energy.total < (state.best_energy.total - BEST_ENERGY_EPS)
            best_hand_pose = jnp.where(is_new_best[:, None], next_hand_pose, state.best_hand_pose)
            best_contact_indices = jnp.where(is_new_best[:, None], next_contact_indices, state.best_contact_indices)
            best_energy = GraspBatchEnergy(
                total=jnp.where(is_new_best, next_energy.total, state.best_energy.total),
                distance=jnp.where(is_new_best, next_energy.distance, state.best_energy.distance),
                penetration=jnp.where(is_new_best, next_energy.penetration, state.best_energy.penetration),
                penetration_depth=jnp.where(is_new_best, next_energy.penetration_depth, state.best_energy.penetration_depth),
                selected_penetration=jnp.where(
                    is_new_best,
                    next_energy.selected_penetration,
                    state.best_energy.selected_penetration,
                ),
            )

            next_state = GraspBatchState(
                hand_pose=next_hand_pose,
                contact_indices=next_contact_indices,
                energy=next_energy,
                best_hand_pose=best_hand_pose,
                best_contact_indices=best_contact_indices,
                best_energy=best_energy,
                ema_grad=ema_grad,
                accepted_steps=state.accepted_steps + accept.astype(jnp.int32),
                rejected_steps=state.rejected_steps + (~accept).astype(jnp.int32),
                step_index=step_index + jnp.asarray(1, dtype=jnp.int32),
                rng_key=next_rng_key,
            )
            snapshot = GraspBatchSnapshot(
                step_index=next_state.step_index,
                hand_pose=next_hand_pose,
                contact_indices=next_contact_indices,
                energy=next_energy,
                accepted=accept,
                temperature=temperature,
                step_size=step_size,
            )
            return next_state, snapshot

        return step_fn

    def _make_run_many_function(self):
        @partial(jax.jit, static_argnames=("steps",))
        def run_many_fn(state: GraspBatchState, *, steps: int) -> GraspBatchState:
            def step_body(_: int, current: GraspBatchState) -> GraspBatchState:
                return self._step_fn(current)[0]

            return jax.lax.fori_loop(0, steps, step_body, state)

        return run_many_fn
