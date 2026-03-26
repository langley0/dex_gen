#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from view_grasp_candidate_env import (
    DEFAULT_CANDIDATE_JSON,
    PROJECT_ROOT,
    _candidate_hand_side,
    _candidate_object_body_name,
    _configure_candidate_camera,
    _contact_labels,
    _find_candidate,
    _load_candidate_payload,
    build_candidate_env_model,
    save_snapshot,
)


def _json_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _candidate_summary_lines(candidate: dict, rank: int) -> list[str]:
    contact_count = int(candidate.get("contact_count", len(candidate["contacts"])))
    return [
        f"{rank:02d}. score={candidate['score']:.6f} "
        f"k={contact_count} "
        f"fingers={','.join(candidate['fingers'])}",
        f"    e_dis={candidate['e_dis']:.6f} "
        f"e_tq={candidate['e_tq']:.6f} "
        f"e_align={candidate.get('e_align', 0.0):.6f} "
        f"e_palm={candidate.get('e_palm', 0.0):.6f} "
        f"e_qpos={candidate.get('e_qpos', 0.0):.6f} "
        f"e_pen={candidate['e_pen']:.1f}",
        f"    contacts={_contact_labels(candidate)}",
    ]


def _save_comparison_strip(
    initial_snapshot: Path,
    final_snapshot: Path,
    output_path: Path,
) -> None:
    initial_image = Image.open(initial_snapshot).convert("RGB")
    final_image = Image.open(final_snapshot).convert("RGB")
    strip = Image.new(
        "RGB",
        (initial_image.width + final_image.width, max(initial_image.height, final_image.height)),
        color=(245, 245, 245),
    )
    strip.paste(initial_image, (0, 0))
    strip.paste(final_image, (initial_image.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review generated grasp results with a top-candidate summary and initial/final snapshots."
    )
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--rank", type=int, default=1, help="1-based candidate rank to inspect.")
    parser.add_argument("--top", type=int, default=5, help="How many top candidates to print.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/review"),
        help="Directory for review snapshots.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the final candidate in the MuJoCo viewer after saving snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_json = _json_path(args.candidate_json)
    output_dir = _json_path(args.output_dir)

    payload = _load_candidate_payload(candidate_json)
    side = _candidate_hand_side(payload)
    object_body_name = _candidate_object_body_name(payload)
    candidates = payload["candidates"]
    candidate = _find_candidate(payload, rank=args.rank)

    print(f"Candidate JSON : {candidate_json}")
    print(f"Hand side      : {side}")
    print(f"Saved count    : {len(candidates)}")
    print("")
    print("Top candidates")
    for entry in candidates[: max(args.top, 0)]:
        for line in _candidate_summary_lines(entry, int(entry["rank"])):
            print(line)

    rank_prefix = f"rank{candidate['rank']:02d}"
    initial_snapshot = output_dir / f"{rank_prefix}_step0.png"
    final_snapshot = output_dir / f"{rank_prefix}_final.png"
    comparison_strip = output_dir / f"{rank_prefix}_compare.png"

    initial_model, initial_data, _, initial_object_center, _ = build_candidate_env_model(
        side=side,
        object_body_name=object_body_name,
        candidate=candidate,
        align_mode="mean_translation",
        trace_step=0,
    )
    save_snapshot(initial_model, initial_data, initial_object_center, initial_snapshot)

    final_model, final_data, _, final_object_center, _ = build_candidate_env_model(
        side=side,
        object_body_name=object_body_name,
        candidate=candidate,
        align_mode="mean_translation",
        trace_step=None,
    )
    save_snapshot(final_model, final_data, final_object_center, final_snapshot)
    _save_comparison_strip(initial_snapshot, final_snapshot, comparison_strip)

    print("")
    print(f"Reviewed rank  : {candidate['rank']}")
    print(f"Final score    : {candidate['score']:.6f}")
    print(
        "Loss terms     : "
        f"e_dis={candidate['e_dis']:.6f}, "
        f"e_tq={candidate['e_tq']:.6f}, "
        f"e_align={candidate.get('e_align', 0.0):.6f}, "
        f"e_palm={candidate.get('e_palm', 0.0):.6f}, "
        f"e_qpos={candidate.get('e_qpos', 0.0):.6f}, "
        f"e_pen={candidate['e_pen']:.1f}"
    )
    print(f"Contacts       : {_contact_labels(candidate)}")
    print(f"Initial image  : {initial_snapshot}")
    print(f"Final image    : {final_snapshot}")
    print(f"Compare image  : {comparison_strip}")

    if args.viewer:
        import mujoco.viewer

        print("Close the viewer window to exit.")
        with mujoco.viewer.launch_passive(final_model, final_data) as viewer:
            _configure_candidate_camera(viewer.cam, final_object_center)
            while viewer.is_running():
                viewer.sync()


if __name__ == "__main__":
    main()
