from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterator, Mapping, TypeVar

import jax

from .grasp_optimizer_state import GraspBatchEnergy, GraspBatchState


T = TypeVar("T")


@dataclass
class ProfileSection:
    total_s: float = 0.0
    count: int = 0

    @property
    def avg_s(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_s / self.count


class RunProfiler:
    def __init__(self) -> None:
        self._sections: dict[str, ProfileSection] = {}

    @contextmanager
    def section(self, name: str, *, sync: Callable[[], None] | None = None) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            if sync is not None:
                sync()
            self.add(name, perf_counter() - start)

    def add(self, name: str, duration_s: float) -> None:
        section = self._sections.setdefault(name, ProfileSection())
        section.total_s += float(duration_s)
        section.count += 1

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_s": section.total_s,
                "count": section.count,
                "avg_s": section.avg_s,
            }
            for name, section in self._sections.items()
        }


def profile_call(
    profiler: RunProfiler,
    name: str,
    fn: Callable[[], T],
    *,
    sync: Callable[[T], object] | None = None,
) -> T:
    start = perf_counter()
    value = fn()
    if sync is not None:
        sync(value)
    profiler.add(name, perf_counter() - start)
    return value


def block_grasp_energy(energy: GraspBatchEnergy) -> GraspBatchEnergy:
    jax.block_until_ready(energy.total)
    jax.block_until_ready(energy.distance)
    jax.block_until_ready(energy.penetration)
    jax.block_until_ready(energy.penetration_depth)
    jax.block_until_ready(energy.selected_penetration)
    return energy


def block_grasp_state(state: GraspBatchState) -> GraspBatchState:
    jax.block_until_ready(state.hand_pose)
    jax.block_until_ready(state.contact_indices)
    block_grasp_energy(state.energy)
    jax.block_until_ready(state.best_hand_pose)
    jax.block_until_ready(state.best_contact_indices)
    block_grasp_energy(state.best_energy)
    jax.block_until_ready(state.ema_grad)
    jax.block_until_ready(state.accepted_steps)
    jax.block_until_ready(state.rejected_steps)
    jax.block_until_ready(state.step_index)
    jax.block_until_ready(jax.random.key_data(state.rng_key))
    return state


def find_bottleneck(
    summary: Mapping[str, Mapping[str, float | int]],
) -> tuple[str, Mapping[str, float | int]] | None:
    if not summary:
        return None
    name = max(summary, key=lambda key: float(summary[key]["total_s"]))
    return name, summary[name]


def format_profile_summary(summary: Mapping[str, Mapping[str, float | int]]) -> str:
    if not summary:
        return "(no profiling data)"

    lines: list[str] = []
    for name, stats in sorted(summary.items(), key=lambda item: float(item[1]["total_s"]), reverse=True):
        total_s = float(stats["total_s"])
        count = int(stats["count"])
        avg_ms = 1.0e3 * float(stats["avg_s"])
        lines.append(f"{name:>24} : total={total_s:7.3f}s count={count:4d} avg={avg_ms:8.3f}ms")
    return "\n".join(lines)
