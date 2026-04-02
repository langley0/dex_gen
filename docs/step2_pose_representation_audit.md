# Step 2 Audit: Pose Representation and Normalization

This step compares the current rebuilt `grasp_refine` data representation against
the deprecated implementation.

Scope:

- what pose the model would learn
- whether root rotation is part of the target
- how normalization is defined

## 1. What was the next checklist item?

From the original audit order, the next item after Step 1 data pipeline is:

- Step 2. Pose representation and normalization

The key question is:

> Does the learning target use the DGA-style reduced pose, or does it still use
> the old full-pose representation?

## 2. Deprecated implementation

Relevant files:

- `grasp_refine/.deprecated/legacy_v1/types.py`
- `grasp_refine/.deprecated/legacy_v1/inspire_hand.py`
- `grasp_refine/.deprecated/legacy_v1/normalization.py`
- `grasp_refine/.deprecated/legacy_v1/dataset.py`

### 2.1 Pose definition

Deprecated `grasp_refine` uses full pose:

- translation: `pose[:3]`
- root rotation 6D: `pose[3:9]`
- Inspire joints: `pose[9:]`

Evidence:

- pose dimension from hand spec: `legacy_v1/inspire_hand.py`
- `pose_dim = 3 + 6 + joints`: `legacy_v1/inspire_hand.py`
- normalizer layout: `legacy_v1/normalization.py`

### 2.2 Normalization rule

Deprecated normalization behavior:

- translation is normalized to `[-1, 1]`
- root rotation 6D is passed through unchanged
- joints are normalized to `[-1, 1]`
- translation bounds come from dataset min/max plus padding
- joint bounds come from MuJoCo Inspire limits

Evidence:

- translation/joint normalization and rotation passthrough:
  `legacy_v1/normalization.py`
- translation bounds from observed records:
  `legacy_v1/dataset.py`
- padding lives in config:
  `legacy_v1/types.py`

### 2.3 Meaning

The deprecated model is asked to learn:

- root translation
- root rotation
- hand joints

So it is not DGA-style.

## 3. Current rebuilt implementation

Relevant files:

- `grasp_refine/types.py`
- `grasp_refine/dataset.py`

### 3.1 Pose definition

The rebuilt data layer now converts source full pose into DGA-style pose:

- keep `full_pose[:3]`
- drop `full_pose[3:9]`
- keep `full_pose[9:]`

So the new target is:

- `pose = [translation(3), joints(N)]`

Evidence:

- conversion function `_dga_pose_from_full_pose`: `grasp_refine/dataset.py`
- converted record stores both `pose` and `pose_full`: `grasp_refine/dataset.py`

### 3.2 Coordinate handling

The rebuilt data layer does not simply throw away root rotation.

In `hand_aligned_object` mode:

- object cloud is first placed in world frame using object `pos/quat`
- then rotated into a frame where hand root rotation becomes identity

Meaning:

- hand root rotation is removed from the learning target
- but its geometric effect is transferred into the conditioned object cloud

Evidence:

- `_world_object_cloud`: `grasp_refine/dataset.py`
- `_transform_object_cloud`: `grasp_refine/dataset.py`
- config flag `coordinate_mode`: `grasp_refine/types.py`

### 3.3 Normalization status

The rebuilt implementation currently has:

- no new normalization module yet
- no new hand-spec module yet
- no learned-pose scaling config yet

This means Step 2 is only partially completed:

- pose representation has changed
- normalization has not yet been rebuilt

## 4. Direct comparison

### 4.1 Pose target

- deprecated: full pose `3 + 6 + 12 = 21`
- rebuilt: DGA-style pose `3 + 12 = 15`
- verdict: different

### 4.2 Root rotation handling

- deprecated: root rotation is explicit model target
- rebuilt: root rotation is removed from target and pushed into object conditioning
- verdict: different

### 4.3 Translation normalization

- deprecated: implemented, dataset min/max plus padding
- rebuilt: not implemented yet
- verdict: missing in rebuilt

### 4.4 Joint normalization

- deprecated: implemented using Inspire joint limits
- rebuilt: not implemented yet
- verdict: missing in rebuilt

### 4.5 Data config surface

- deprecated `DatasetConfig` includes `train_fraction`, `seed`, `normalizer_padding`
- rebuilt `DatasetConfig` currently only covers source loading and object coordinate mode
- verdict: simplified in rebuilt

## 5. Conclusion

Yes, there is already a major difference between the rebuilt code and deprecated code
at the Step 2 level.

The rebuilt version has already moved to the DGA-style pose target, which is good
for parity with DGA.

But normalization has not been recreated yet, so the Step 2 work is not complete.

## 6. Immediate next task

The next concrete implementation task should be:

- add a new DGA-style normalizer for `pose = [translation, joints]`

That normalizer should decide:

- how translation bounds are set
- whether bounds are fixed or dataset-derived
- how joint limits are supplied for the reduced pose
