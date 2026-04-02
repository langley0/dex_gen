# DGA to `grasp_refine` Audit Plan

This document fixes the reference implementation first, then defines how to audit
`grasp_refine` against that reference without mixing multiple changes at once.

The baseline is the bundled `third_party/DexGrasp-Anything` code, not our memory
of the paper or README.

## 1. Source of Truth

Use these files as the implementation baseline:

- Dataset: `third_party/DexGrasp-Anything/datasets/multidex_shadowhand_ur.py`
- Training entry: `third_party/DexGrasp-Anything/train_ddm.py`
- Diffusion and loss: `third_party/DexGrasp-Anything/models/dm/ddpm.py`
- Model: `third_party/DexGrasp-Anything/models/model/unet.py`
- Task config: `third_party/DexGrasp-Anything/configs/task/grasp_gen_ur.yaml`
- Model config: `third_party/DexGrasp-Anything/configs/model/unet_grasp.yaml`
- Diffusion config: `third_party/DexGrasp-Anything/configs/diffuser/ddpm.yaml`

Audit the port against these files:

- Dataset: `grasp_refine/io.py`, `grasp_refine/dataset.py`
- Pose spec and normalization: `grasp_refine/inspire_hand.py`, `grasp_refine/normalization.py`
- Model: `grasp_refine/model_factory.py`, `grasp_refine/model_dga.py`, `grasp_refine/model.py`
- Diffusion and loss: `grasp_refine/diffusion.py`, `grasp_refine/losses.py`
- Training loop: `grasp_refine/trainer.py`
- Sampling and inference: `grasp_refine/sampling.py`, `grasp_refine/inference.py`

## 2. Baseline Pipeline Summary

### 2.1 Dataset

DGA `MultiDexShadowHandUR` uses:

- Object-level fixed train/test split, not a random sample split.
- Pose target `x = [global_trans(3), joint_angle(24)]`.
- Joint angles are taken from `mdata_qpos[9:]`; translation is `mdata_qpos[:3]`.
- Translation and joint angle are normalized to `[-1, 1]`.
- Object condition is a point cloud plus normals.
- Point cloud is rotated by the object rotation matrix from the dataset metadata.
- In training, object points are randomly resampled every call.
- In test, point resampling is fixed by `np.random.seed(0)`.

Code evidence:

- Fixed object split: `multidex_shadowhand_ur.py:19-49`
- Pose construction and normalization: `multidex_shadowhand_ur.py:133-146`
- Point cloud rotation and resampling: `multidex_shadowhand_ur.py:178-191`
- Data payload: `multidex_shadowhand_ur.py:204-220`

### 2.2 Pose Representation

DGA training does not model hand root orientation as part of `x`.

- Task dimension is hard-coded as `3 + 24 = 27`.
- During auxiliary physics losses, the code inserts a fixed identity 6D rotation.

Code evidence:

- Model dimension: `utils/misc.py:18-21`
- Identity 6D root rotation injected for hand model losses: `models/dm/ddpm.py:132-149`

This means the reference problem is not "full pose diffusion". It is closer to
"translation plus hand joint diffusion under object conditioning".

### 2.3 Model

DGA model structure is:

1. Scene encoder extracts scene tokens from object points and normals.
2. UNet-like denoiser consumes noisy `x_t`, timestep embedding, and scene tokens.
3. The denoiser predicts noise.

Code evidence:

- Scene encoder and conditioning path: `models/model/unet.py:32-49`, `models/model/unet.py:133-165`
- Time embedding and denoiser blocks: `models/model/unet.py:51-88`
- Forward pass: `models/model/unet.py:90-131`

### 2.4 Diffusion and Loss

DGA diffusion training:

- Uses linear beta schedule with `beta_start=1e-4`, `beta_end=1e-2`, `steps=100`.
- Uses `rand_t_type='half'`.
- Predicts noise with L1 or L2 loss.
- Adds three geometry/physics-inspired terms: `ERF`, `SPF`, `SRF`.

Code evidence:

- Schedule and loss config: `configs/diffuser/ddpm.yaml:1-16`
- Timestep sampling and noise objective: `models/dm/ddpm.py:101-149`

### 2.5 Training Loop

DGA training loop:

- Uses PyTorch DDP entrypoint.
- Uses Adam.
- Trains on dataloader batches, not pre-materialized full-device arrays.
- Logs and checkpoints by epoch.

Code evidence:

- Dataset/dataloader creation: `train_ddm.py:130-147`
- Optimizer and DDP: `train_ddm.py:149-173`
- Train step loop: `train_ddm.py:178-215`

### 2.6 Sampling and Inference

DGA sampling:

- Starts from Gaussian noise.
- Runs ancestral reverse diffusion.
- Can optionally use DPM-Solver++.
- Can optionally use optimizer-guided correction during sampling.
- Reuses precomputed condition tokens during reverse steps.

Code evidence:

- Reverse sampling core: `models/dm/ddpm.py:200-311`
- Final sample wrapper: `models/dm/ddpm.py:313-340`
- Inference entry: `sample.py:31-89`

## 3. Current `grasp_refine` Mapping

### 3.1 Dataset

`grasp_refine` currently uses optimizer artifact `.npz` files as the training set.

- Records come from `best_hand_pose` or `hand_pose`.
- Object points are re-sampled from mesh files using a stable seed per object kind.
- Train/val split is random sample split controlled by `train_fraction`.

Code evidence:

- Artifact loader: `grasp_refine/io.py:35-89`
- Dataset sample payload: `grasp_refine/dataset.py:39-63`
- Random sample split: `grasp_refine/dataset.py:82-108`

### 3.2 Pose Representation

`grasp_refine` uses:

- `pose = [root_pos(3), root_rot_6d(6), inspire_joints(N)]`
- Translation is normalized from dataset min/max plus padding.
- Rotation `3:9` is passed through unchanged.
- Joint values are normalized to `[-1, 1]` using Inspire joint limits from MuJoCo.

Code evidence:

- Inspire pose dimension: `grasp_refine/inspire_hand.py:17-25`
- MuJoCo joint bounds extraction: `grasp_refine/inspire_hand.py:28-50`
- Normalization logic: `grasp_refine/normalization.py:17-37`
- Translation bounds built from observed data, not fixed constants: `grasp_refine/dataset.py:70-79`

### 3.3 Model

`grasp_refine` has two architectures:

- `mlp`
- `dga_transformer`

The `dga_transformer` is only DGA-inspired, not an implementation-level clone of
the DGA UNet.

Code evidence:

- Architecture switch: `grasp_refine/model_factory.py:10-46`
- Transformer scene token pooling and cross-attention denoiser:
  `grasp_refine/model_dga.py:136-250`

### 3.4 Diffusion and Loss

`grasp_refine` keeps the same high-level DDPM training objective shape:

- same beta endpoints
- same default step count
- same `rand_t_type`
- same L1/L2 noise loss option

But it replaces DGA auxiliary terms with:

- `joint_limit_loss`
- `root_distance_loss`

Code evidence:

- Diffusion schedule and timestep sampling: `grasp_refine/diffusion.py:24-59`
- Training loss composition: `grasp_refine/diffusion.py:70-122`

### 3.5 Training Loop

`grasp_refine` training differs structurally:

- JAX instead of PyTorch
- full dataset materialized into arrays
- full epoch executed with `jax.lax.scan`
- AdamW instead of Adam
- optional validation split in the same training script

Code evidence:

- Bundle construction: `grasp_refine/trainer.py:52-87`
- Batch materialization and padding: `grasp_refine/trainer.py:111-156`
- JIT epoch runners: `grasp_refine/trainer.py:167-251`
- Training loop: `grasp_refine/trainer.py:282-340`

### 3.6 Sampling and Inference

`grasp_refine` inference:

- samples only from object-conditioned noise
- uses one custom reverse update
- does not implement DPM-Solver++
- does not implement optimizer-guided sampling
- evaluates generated poses only with scalar post-hoc heuristics

Code evidence:

- Reverse update loop: `grasp_refine/sampling.py:30-70`
- Inference entry and post-hoc metrics: `grasp_refine/inference.py:84-168`

## 4. High-Risk Differences Already Visible

These should be treated as likely root causes before investigating smaller issues.

### Risk A: Different training data definition

DGA trains on its native dataset records, while `grasp_refine` trains on optimizer
artifacts. This is not just a loader difference. It changes the distribution of
target poses, object metadata, and possibly the "quality prior" that the model sees.

### Risk B: Different split semantics

DGA uses fixed object-level train/test split. `grasp_refine` uses random sample split.
This can hide generalization failure and make comparisons to DGA misleading.

### Risk C: Different pose problem

DGA learns `3 + 24`, with no modeled root orientation in `x`.
`grasp_refine` learns `3 + 6 + Inspire joints`.
This is the largest semantic deviation in the port.

### Risk D: Different hand and joint manifold

DGA auxiliary losses use the ShadowHand kinematic model.
`grasp_refine` uses Inspire joint limits from MuJoCo.
Even if the diffusion code matched, the feasible hand manifold is different.

### Risk E: Different auxiliary losses

DGA adds `ERF + SPF + SRF`.
`grasp_refine` replaces them with `joint_limit + root_distance`.
This changes what "good sample" means during optimization.

### Risk F: Different reverse process

DGA supports ancestral DDPM, optional DPM-Solver++, and optimizer-guided correction.
`grasp_refine` currently uses a much simpler custom reverse update.

## 5. Audit Order

Do not inspect everything at once. Use this order.

### Step 1. Lock the exact DGA target problem

Deliverable:

- one-page table describing DGA `x`, condition, normalization, model input/output,
  loss terms, and reverse sampler

Exit criteria:

- no ambiguity remains about whether DGA models root rotation
- no ambiguity remains about whether DGA uses fixed object split or random split

### Step 2. Check whether `grasp_refine` solves the same problem

Questions:

- Is the target variable the same?
- Is the object condition the same?
- Is the train/test split semantics the same?
- Is the same hand family being modeled?

Deliverable:

- "same / different / partially same" matrix

Exit criteria:

- if the answer is "different problem", stop calling it a direct port and define
  the new target explicitly

### Step 3. Align data before touching the model

Required checks:

- same sample source
- same object split semantics
- same point count
- same point sampling randomness policy
- same normalization ranges

Reason:

If this stage is not aligned, later model changes will be hard to interpret.

### Step 4. Align pose representation

Required checks:

- whether root rotation should be modeled at all
- whether DGA should be emulated as `3 + joints` first
- whether Inspire needs a reduced target representation for parity experiments

Reason:

This is the highest-impact semantic mismatch.

### Step 5. Align sampler before deep model changes

Required checks:

- reverse equation parity
- ancestral noise injection parity
- whether condition tokens are reused the same way
- whether DPM-Solver++ parity matters for the comparison being made

### Step 6. Align auxiliary objectives

Required checks:

- whether to reproduce DGA `ERF/SPF/SRF`
- whether current `joint_limit/root_distance` can only be used as ablations,
  not as the reference objective

### Step 7. Only then compare architectures

Questions:

- Is the current `dga_transformer` close enough to a DGA ablation, or do we need
  a closer UNet port first?
- Which architectural gaps still matter after data and sampler are aligned?

## 6. Concrete Work Items

Use the following checklist in order.

- [ ] Write the DGA baseline table from code, not from README text.
- [ ] Write the `grasp_refine` baseline table from code.
- [ ] Mark each row as `match`, `partial`, or `mismatch`.
- [ ] Freeze the top 3 mismatches that are most likely to explain output drift.
- [ ] Design one-variable experiments for those top 3 mismatches.
- [ ] Run the first parity experiment on data/split, not on model architecture.

## 7. Recommended First Experiments

Do these in order.

1. Data split parity experiment
   - Replace random `train_fraction` split with object-level split semantics.

2. Pose target parity experiment
   - Train a parity variant that removes root rotation from the modeled target,
     matching DGA's `3 + joint` style first.

3. Sampler parity experiment
   - Replace the current reverse update with a faithful ancestral DDPM update
     before touching the network architecture.

4. Loss parity experiment
   - Add a DGA-like auxiliary-loss path or, at minimum, isolate current losses as
     an ablation rather than the reference.

## 8. Current Working Hypothesis

The largest output gap is unlikely to come from a small bug in `model_factory.py`.
It is more likely caused by a stack of semantic mismatches:

1. different training data source
2. different target pose representation
3. different auxiliary objective
4. different reverse sampling process

The model architecture may still matter, but it should not be the first suspect.
