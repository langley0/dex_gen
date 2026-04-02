## Multi-Object Generation Stage 1

This stage is the recommended first multi-object dataset for `grasp_refine`.

### Object set

- `cylinder`
- `cube_s006`
- `cube_s008`
- `drill`
- `decor01`

### Why this set

- `cylinder`: simple primitive, already used in the current baseline
- `cube_s006` / `cube_s008`: similar topology with different scale, useful for checking object-key and size sensitivity
- `drill`: asymmetric mesh object
- `decor01`: second mesh object with a different local shape distribution

This gives:

- primitive + mesh coverage
- symmetric + asymmetric coverage
- repeated category with scale variation

### Preparation script

Use:

```bash
./.venv/bin/python scripts/prepare_multi_object_grasp_data.py --object-set stage1
```

This writes a manifest under:

```text
outputs/grasp_multi_object_stage1/manifest.json
```

The manifest contains:

- per-object optimizer artifact paths
- exact optimizer commands
- the combined `run_grasp_refine_prepare_dga_dataset.py` command

### Recommended execution order

1. Run optimizer artifacts for all objects.
2. Build one combined DGA-style prepared dataset.
3. Train `grasp_refine` with object-level validation enabled.
4. Evaluate best checkpoint sample quality before changing model size or loss weights.

### Notes

- The script uses explicit output paths, so multiple cube sizes do not collide.
- The current stage keeps the object set intentionally small to make the first multi-object experiment easier to interpret.
