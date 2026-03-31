# DexGrasp Anything Initial Grasp Notes

이 문서는 DexGrasp Anything(DGA) 공개 코드 기준으로
"초기 파지(initial grasp)"가 실제로 무엇을 뜻하는지 정리한 메모다.

핵심은 DGA 안에서 "초기 파지"라는 말을 하나의 의미로 쓰기 어렵다는 점이다.
코드를 보면 적어도 아래 세 층을 구분해야 한다.

1. 학습 데이터에 들어 있는 정답 grasp pose
2. 추론 시 diffusion이 시작하는 초기 상태
3. 평가 시 Isaac Gym에 넣는 초기 grasp pose

## 한 줄 결론

DGA는 추론 때 사람이 작성한 초기 hand pose에서 시작하지 않는다.

현재 공개 코드 기준으로:

- 학습 시에는 dataset에 저장된 grasp pose를 정답으로 사용하고
- 추론 시에는 그 pose에서 시작하는 것이 아니라 Gaussian noise에서 시작하며
- physics-guided sampling은 "초기 파지 생성기"가 아니라 denoising 경로를
  보정하는 단계다

즉 DGA의 생성 파이프라인은
"초기 파지를 먼저 만들고 refine"가 아니라
"object-conditioned diffusion이 노이즈에서 grasp를 생성하고,
필요하면 그 중간 경로를 물리 항으로 보정"하는 구조에 가깝다.

## 1. 학습용 grasp pose는 dataset이 제공한다

각 dataset loader를 보면 `metadata`에서 이미 grasp pose를 읽어온다.

대표적으로 아래 파일들이 같은 패턴을 가진다.

- `datasets/real_dex.py`
- `datasets/DexGraspNet.py`
- `datasets/Unidexgrasp.py`
- `datasets/Grasp_anyting.py`
- `datasets/DexGRAB.py`
- `datasets/multidex_shadowhand_ur.py`

일반적인 흐름은 다음과 같다.

1. `mdata['translations']`를 읽는다
2. `mdata['joint_positions']`를 읽는다
3. 둘을 이어 붙여 `qpos` 또는 `x`로 사용한다
4. 필요하면 translation과 joint angle을 각각 normalize한다

예를 들어 `real_dex.py`에서는:

- `global_trans = mdata['translations']`
- `joint_angle = mdata['joint_positions']`
- `mdata_qpos = torch.cat([global_trans, joint_angle], dim=0)`

로 구성한다.

여기서 중요한 점은 모델 입력/출력 차원이다.

`utils/misc.py`의 `compute_model_dim(cfg)`는 `grasp_gen_ur`에 대해
`3 + 24 = 27`을 반환한다.

즉 DGA가 직접 모델링하는 grasp 벡터는 현재 공개 코드 기준으로:

- 전역 translation 3차원
- 손 joint 24차원

의 27차원이다.

## 2. 회전은 grasp 벡터에 직접 포함되지 않는다

dataset loader를 보면 hand/global rotation을 그대로 `x`에 넣지 않는다.

대신 보통:

- `hand_rot_mat = mdata['rotations']`
- `object_rot_mat = hand_rot_mat.T`

형태로 별도 저장해 둔다.

예를 들어 `real_dex.py`, `DexGraspNet.py`, `Unidexgrasp.py`,
`Grasp_anyting.py` 모두 비슷하게
`'object_rot_mat': hand_rot_mat.T`를 frame에 저장한다.

즉 학습 표현은 "회전까지 포함된 완전한 손 pose"라기보다,

- object-aligned frame에서의 hand translation
- hand joint angles

를 diffusion target으로 쓰고,
회전은 object point cloud 정렬과 최종 출력 복원 단계에서 따로 다루는 설계다.

## 3. 추론 시 시작점은 hand-crafted grasp가 아니라 Gaussian noise다

실제 reverse diffusion 시작은 `models/dm/ddpm.py`의
`p_sample_loop(...)`에 있다.

여기서 샘플링은:

- `x_t = torch.randn_like(data['x'], device=self.device)`

로 시작한다.

즉 초기 상태는:

- dataset에서 읽은 어떤 특정 grasp pose가 아니고
- open-hand template도 아니며
- 완전히 Gaussian noise인 27차원 벡터다

이 사실은 `models/visualizer.py`에서도 다시 확인된다.

visualizer는 샘플링용 data를 만들 때:

- `data['x'] = torch.randn(self.ksample, 27, device=device)`

를 직접 넣는다.

즉 공개 코드에서 추론 파이프라인은
"object point cloud + normals"를 condition으로 받아
"random noise -> grasp pose"를 복원하는 전형적인 conditional diffusion 구조다.

## 4. Physics-Guided Sampling은 시작 pose를 정하는 단계가 아니다

README나 코드 설명만 보면 물리 항이 생성 자체를 담당하는 것처럼 보일 수 있지만,
공개 구현을 보면 역할은 더 제한적이다.

`models/base.py`에서 optimizer가 설정되면 diffusion sampler에 연결되고,
실제 개입 지점은 `models/dm/ddpm.py`의 sampling loop다.

현재 구조는:

1. diffusion이 노이즈에서 샘플을 복원하고
2. 각 timestep 또는 지정된 interval에서
3. `optimizer.gradient(...)`가 현재 샘플을 보정한다

즉 physics-guided sampling의 역할은:

- 초기 파지 자체를 만드는 것보다는
- denoising 중간 경로를 물리 항으로 수정하는 것

이다.

`scripts/grasp_gen_ur/sample.sh`도 이 점을 잘 보여 준다.

- `OPT` 없이 실행하면 일반 샘플링
- `OPT`를 주면 `optimizer=grasp_with_object`가 켜진다

즉 물리 가이던스는 샘플링의 옵션이지,
별도의 초기 파지 생성 모듈이 아니다.

## 5. 학습 시 모델이 배우는 것은 "noise -> dataset grasp" 복원이다

`models/dm/ddpm.py`의 `forward(...)`를 보면 학습은 표준 diffusion 방식이다.

1. dataset의 `data['x']`를 정답 grasp로 사용한다
2. 여기에 timestep별 Gaussian noise를 섞어 `x_t`를 만든다
3. 모델이 noise를 예측한다
4. 예측한 `pred_x0`에 대해 추가 물리 항을 계산한다

즉 학습 시점의 grasp "초기값"은
추론용 의미의 initial grasp가 아니라
이미 존재하는 데이터셋의 정답 grasp example이다.

여기서 물리 항은:

- `ERF_loss`
- `SPF_loss`
- `SRF_loss`

로 loss에 더해진다.

이건 "초기 파지를 작성한다"기보다
"정답 grasp manifold와 물리적으로 더 그럴듯한 grasp manifold를 함께 학습한다"
에 가깝다.

## 6. 최종 출력은 27D 예측값을 다시 full qpos 형식으로 복원한다

샘플링이 끝난 뒤 `models/visualizer.py`는 출력을 후처리한다.

순서는 대략 아래와 같다.

1. joint angle과 translation을 denormalize한다
2. identity 6D rotation을 만든다
3. object rotation과 결합해 world 쪽 회전을 복원한다
4. translation도 같은 회전으로 world frame에 맞춘다
5. 최종적으로 `[translation(3), rotation_6d(6), joint(24)]`를 이어 붙인다

즉 모델이 직접 예측한 것은 27차원이지만,
최종 `sample_qpos`로 저장될 때는 보통:

- `3 + 6 + 24 = 33`차원 형태

로 확장된다.

`scripts/grasp_gen_ur/test.py`와 `envs/tasks/grasp_test_force_shadowhand.py`는
바로 이 최종 `sample_qpos`를 평가 입력으로 사용한다.

## 7. 평가 시의 "initial grasp"는 생성 결과 자체다

평가 단계로 넘어가면 "초기 파지"라는 말이 또 다른 뜻을 가진다.

`scripts/grasp_gen_ur/test.py`는:

- `res_diffuser.pkl` 안의 `sample_qpos`를 읽고
- 이를 `init_opt_q=q_generated`로 Isaac Gym 환경에 넘긴다

즉 evaluation 관점에서의 initial grasp는:

- diffusion이 생성한 최종 grasp pose

다.

그 뒤 `envs/tasks/grasp_test_force_shadowhand.py` 안에서
`_set_normal_force_pose()`가 한 번 더 실행되어,

- object surface normal 방향으로 약한 closure를 주는 local adjustment

를 수행한 뒤 안정성 테스트를 한다.

그래서 평가 코드 기준으로는
"초기 파지 -> 평가 전 closure 보정 -> push stability test"
의 흐름이 존재한다.

하지만 이것은 생성기 본체의 initial grasp 작성 방식과는 다른 층이다.

## 8. DGA에서 "초기 파지를 어떻게 작성하나?"에 대한 가장 정확한 답

코드 기준으로 가장 정확히 답하면 아래와 같다.

### 질문 1. 학습 데이터의 grasp는 어떻게 준비되나?

- 각 dataset `.pt` 파일 안에 들어 있는 `translations`와
  `joint_positions`를 읽어온다
- 회전은 별도 `object_rot_mat`로 처리한다
- 즉 DGA가 학습하는 grasp target은 dataset이 이미 제공한다

### 질문 2. 추론 시작 pose는 어떻게 정하나?

- hand-crafted initialization을 쓰지 않는다
- 27차원 Gaussian noise에서 시작한다

### 질문 3. physics-guided sampling은 언제 개입하나?

- 시작점을 만드는 단계가 아니라
- reverse diffusion 중간 보정 단계에서 개입한다

### 질문 4. 평가할 때 simulator에 넣는 초기 pose는 무엇인가?

- diffusion 최종 출력 `sample_qpos`
- 이후 normal-force closure를 한 번 더 적용한다

## 9. 우리 쪽 구현과 비교할 때 읽어야 하는 포인트

현재 `grasp_refine`는 upstream grasp 결과를 입력으로 받는 post-process 단계다.
반면 DGA 원본 공개 코드는:

- object-conditioned generation
- noise start
- reverse diffusion
- optional physics-guided correction

을 하나의 샘플링 루프 안에서 수행한다.

따라서 둘의 가장 큰 차이는:

- DGA는 처음부터 grasp를 생성한다
- 현재 `grasp_refine`는 이미 생성된 grasp를 refine한다

는 점이다.

이 차이 때문에 DGA의 "initial grasp"를 그대로 현재 refine 쪽에 대응시키려면,
아래 셋을 분리해서 이해해야 한다.

- dataset grasp target
- diffusion start noise
- simulator evaluation seed pose

## 짧은 결론

DGA 공개 구현에는
"물체를 보고 초기 hand pose를 하나 휴리스틱하게 써 내려가는 단계"가 없다.

현재 코드 기준으로 초기 파지의 실체는 다음과 같다.

- 학습 기준: dataset에 저장된 grasp pose
- 추론 시작: Gaussian noise
- 평가 시작: diffusion이 생성한 `sample_qpos`

따라서 DGA를 "초기 파지 생성기 + refine기"로 보는 것보다는,
"noise에서 grasp를 직접 생성하는 conditional diffusion 모델이며,
물리 항은 그 학습과 샘플링을 보조한다"로 이해하는 편이 코드와 더 잘 맞는다.
