# grasp_refine TODO

## High Priority

- [ ] Change single-sample `best_hand_pose` selection to prioritize actual
      overlap quality before surrogate total energy.
- [ ] Add an overlap-release mode where `surface_pull` is disabled or heavily
      down-weighted until actual penetration is cleared.
- [ ] Wire `external_threshold` into the single-sample energy term as a real
      deadband / activation threshold.
- [ ] Add a surface-band stopping term so overlap is pushed out only to the
      surface neighborhood instead of overshooting or re-pulling too hard.

## Paper Alignment

- [ ] Revisit the exact DGA SPF / ERF / SRF mapping and document each equation
      next to the implementation.
- [ ] Decide whether external repulsion should stay strictly `max`-based like
      the paper or allow `topk` / `mean-positive` variants for overlap cleanup.
- [ ] Clarify whether `contact_target_local` should remain unused, become a soft
      anchor term, or be removed from the API.

## Evaluation

- [ ] Track both surrogate improvement and actual MuJoCo overlap improvement in
      one summary view.
- [ ] Save the step that minimizes actual penetration, not only the step that
      minimizes surrogate total.
- [ ] Add side-by-side viewer overlays for `initial`, `best-surrogate`, and
      `best-actual`.
- [ ] Add a simple sweep script for `surface_pull_weight`,
      `external_repulsion_weight`, and `surface_pull_threshold`.

## Batch Path

- [ ] Add batch-level selection for the best actual-overlap-fixed sample.
- [ ] Revisit the sample-wise batch evaluation path for GPU efficiency.
- [ ] Export batch diagnostics in a format that is easy to compare across runs.
