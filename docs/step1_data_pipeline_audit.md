# Step 1 Audit: Data Pipeline

This document audits only the data stage.

Scope:

- what one training sample means in DGA
- what one training sample means in `grasp_refine`
- whether the condition seen by the model is semantically the same

This is intentionally narrow. We do not discuss model architecture, loss design,
or sampling here except when they depend directly on data semantics.

## 1. Question for This Step

Before asking whether the model matches DGA, answer this first:

> Is `grasp_refine` training on the same kind of `(target pose, object condition)`
> pair that DGA trains on?

If the answer is no, later model comparisons are hard to interpret.

## 2. DGA Data Pipeline, Precisely

Reference:

- `third_party/DexGrasp-Anything/datasets/multidex_shadowhand_ur.py`
- `third_party/DexGrasp-Anything/configs/task/grasp_gen_ur.yaml`
- `third_party/DexGrasp-Anything/datasets/misc.py`

### 2.1 Sample source

DGA `MultiDexShadowHandUR` loads from two native dataset assets:

- pose/metadata tensor file: `shadowhand_downsample.pt` or `shadowhand.pt`
- object point clouds with normals: `object_pcds_nors.pkl`

Evidence:

- dataset asset paths: `multidex_shadowhand_ur.py:90-92`
- native dataset load: `multidex_shadowhand_ur.py:112-115`

### 2.2 Train/test split semantics

DGA uses a fixed object-level split.

- Train objects are listed explicitly.
- Test objects are listed explicitly.
- A sample belongs to train or test based on object identity, not random record split.

Evidence:

- train split list: `multidex_shadowhand_ur.py:19-31`
- test split list: `multidex_shadowhand_ur.py:32-34`
- split selection by phase: `multidex_shadowhand_ur.py:69-76`
- sample filtering by object name: `multidex_shadowhand_ur.py:142-146`

Meaning:

- Generalization is measured across unseen objects.
- No random holdout of same-object samples is involved here.

### 2.3 What the target `x` is

For each metadata record `mdata`:

- translation is `mdata_qpos[:3]`
- joint angles are `mdata_qpos[9:]`
- root orientation is omitted from `x`

Then DGA builds:

- `x = [normalized_translation(3), normalized_joint_angles(24)]`

Evidence:

- qpos slicing: `multidex_shadowhand_ur.py:133-141`
- constructed target stored as `qpos`: `multidex_shadowhand_ur.py:141-146`
- returned to model as `data['x']`: `multidex_shadowhand_ur.py:199-205`

Important consequence:

- DGA data stage does not ask the model to predict root orientation.
- So the learning target is already reduced before the network sees it.

### 2.4 How object condition is built

For each sample:

1. object identity is `frame['object_name']`
2. object rotation matrix is `frame['object_rot_mat']`
3. canonical object point cloud is loaded from `self.scene_pcds[scene_id]`
4. both points and normals are rotated by `scene_rot_mat`
5. `num_points=2048` points are re-sampled

Evidence:

- object id and rotation lookup: `multidex_shadowhand_ur.py:178-181`
- normal rotation: `multidex_shadowhand_ur.py:182`
- point rotation: `multidex_shadowhand_ur.py:183`
- point resampling: `multidex_shadowhand_ur.py:186-191`
- point count config: `grasp_gen_ur.yaml:17-35`

Important consequence:

- DGA condition is not just "object geometry".
- It is "object geometry in the sample's rotated pose frame".

This is a very strong semantic detail.

### 2.5 Randomness policy

DGA uses different point sampling behavior by phase:

- train: random permutation every call
- test: fixed seed `0` before resampling

Evidence:

- fixed seed in non-train phase: `multidex_shadowhand_ur.py:186-189`

Meaning:

- train sees point-resampling augmentation
- test is stabilized for evaluation repeatability

### 2.6 Feature tensor shape seen by the model

Config says:

- `use_normal: true`
- `use_color: false`
- `num_points: 2048`

So the scene feature per point is normal-only feature with xyz position.

Evidence:

- dataset config: `grasp_gen_ur.yaml:21-33`
- feature build from normals: `multidex_shadowhand_ur.py:212-215`

With PointTransformer, the dataloader collate:

- stacks tensors
- flattens batched points into `(B*N, C)`
- builds `offset` for batch boundaries

Evidence:

- point-transformer collate: `datasets/misc.py:15-46`

## 3. `grasp_refine` Data Pipeline, Precisely

Reference:

- `grasp_refine/io.py`
- `grasp_refine/dataset.py`
- `grasp_refine/object_mesh.py`
- `grasp_refine/types.py`
- `grasp_refine/normalization.py`

### 3.1 Sample source

`grasp_refine` does not train on DGA native dataset records.

Instead it trains on local optimizer/refine artifact `.npz` files and converts
them into `GraspRecord`s.

Evidence:

- artifact discovery: `grasp_refine/io.py:34-41`
- sample state extraction: `grasp_refine/io.py:44-58`
- one record per saved pose in artifact: `grasp_refine/io.py:73-90`

Meaning:

- The data distribution is the distribution produced by the local optimizer
  pipeline, not the original DGA dataset distribution.

### 3.2 Train/val split semantics

`grasp_refine` uses random sample-level split controlled by `train_fraction`.

Evidence:

- split config: `grasp_refine/types.py:12-23`
- random shuffled index split: `grasp_refine/dataset.py:82-108`

Meaning:

- Same object kind can appear in both train and val.
- This is not semantically equivalent to DGA object-level generalization split.

### 3.3 What the target `pose` is

`grasp_refine` record pose is loaded directly from artifact `hand_pose` or
`best_hand_pose`.

The downstream normalizer assumes:

- `pose[:3]` = translation
- `pose[3:9]` = root rotation in 6D
- `pose[9:]` = Inspire joints

Evidence:

- raw record pose load: `grasp_refine/io.py:48-55`, `grasp_refine/io.py:84-86`
- pose normalization layout: `grasp_refine/normalization.py:21-37`
- pose dimension from Inspire hand spec: `grasp_refine/inspire_hand.py:23-25`

Meaning:

- `grasp_refine` target variable is full root pose plus joints.
- This is not the same target as DGA `x`.

### 3.4 How object condition is built

For each object kind:

1. mesh is loaded from object metadata
2. surface points and normals are sampled from the mesh
3. a stable seed is used per object kind
4. sampled cloud is cached and reused for every sample of that object kind

Evidence:

- cached per-object cloud: `grasp_refine/dataset.py:52-63`
- stable per-kind seed: `grasp_refine/dataset.py:57-60`
- mesh sampling implementation: `grasp_refine/object_mesh.py:132-175`

Meaning:

- `grasp_refine` condition is object geometry in a canonical mesh frame.
- It is not re-oriented per sample.
- It is deterministic for an object kind unless config changes.

### 3.5 Missing per-sample object rotation

This is the biggest data-stage mismatch found in this step.

DGA explicitly rotates object points and normals by the sample's rotation matrix:

- `multidex_shadowhand_ur.py:180-183`

`grasp_refine` currently does not include any per-sample object rotation in the
dataset payload:

- returned dataset fields are only `pose`, `pose_raw`, `object_points`,
  `object_normals`, `object_index`, `contact_indices`, `energy`
- no object transform or rotation field is present

Evidence:

- dataset sample payload: `grasp_refine/dataset.py:39-50`
- mesh sampler returns canonical mesh points only: `grasp_refine/object_mesh.py:151-175`

Meaning:

- In DGA, two samples of the same object with different object rotations produce
  different point-cloud conditions.
- In `grasp_refine`, two samples of the same object kind produce the same object
  point cloud condition.

This is not a minor implementation detail. It changes the meaning of the
conditioning variable.

### 3.6 Randomness policy

`grasp_refine` point cloud sampling is effectively deterministic per object kind:

- seed is stable per object kind
- sampled cloud is cached

Evidence:

- stable seed: `grasp_refine/dataset.py:57-60`
- cache reuse: `grasp_refine/dataset.py:52-63`

Meaning:

- There is no train-time point resampling augmentation like DGA.
- The model sees one fixed cloud per object kind.

## 4. Direct Comparison

### 4.1 Sample source

- DGA: native grasp dataset records
- `grasp_refine`: local optimizer artifact poses
- Verdict: `mismatch`

### 4.2 Split semantics

- DGA: fixed object-level split
- `grasp_refine`: random sample-level split
- Verdict: `mismatch`

### 4.3 Target variable

- DGA: translation + joint angles only
- `grasp_refine`: translation + root rotation 6D + joints
- Verdict: `mismatch`

### 4.4 Object condition semantics

- DGA: object cloud rotated into per-sample object frame
- `grasp_refine`: canonical mesh cloud shared across same object kind
- Verdict: `mismatch`

### 4.5 Point cloud randomness

- DGA train: random resampling
- `grasp_refine`: deterministic cached sampling
- Verdict: `mismatch`

### 4.6 Point count and feature type

- DGA: 2048 points, xyz + normals, no color by default
- `grasp_refine`: 2048 points by default, xyz + normals
- Verdict: `match`

## 5. Step 1 Conclusion

At the data stage alone, `grasp_refine` is not yet solving the same supervised
learning problem as DGA.

The two most important findings are:

1. `grasp_refine` trains on a different target variable.
2. `grasp_refine` conditions on a different object representation, because it
   loses DGA's per-sample object rotation.

So before comparing model blocks or diffusion code, we should treat the current
port as a semantic redesign rather than a faithful data-level port.

## 6. What Must Be Verified Next

The next step should stay narrow and build directly on this result.

Recommended Step 2:

- audit pose representation and normalization in detail

Specific questions for Step 2:

- Is the DGA omission of root rotation essential or incidental?
- Can we define a parity target representation for Inspire hand?
- Are translation normalization ranges in `grasp_refine` comparable to DGA's
  fixed global bounds, or are they dataset-dependent in a way that changes the task?

## 7. Step 1 Priority Fix Candidates

These are not implementation tasks yet. They are the first candidates to test
once we move from analysis to intervention.

1. Introduce object-level split semantics into `grasp_refine`.
2. Add per-sample object transform conditioning or rotate sampled mesh points into
   the sample's object frame.
3. Decide whether parity mode should remove root rotation from the training target.
