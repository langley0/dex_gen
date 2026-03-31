# grasp_refine 구현 메모

이 문서는 현재 `grasp_refine`에서 사용 중인 `surface_pull`,
`external_repulsion`, `self_repulsion`의 구현 상태를 코드만 보고 정리한
문서다.

범위:

- 의도한 논문 설계가 아니라 현재 로컬 코드에 실제로 들어 있는 동작만 다룬다.
- 핵심 기준 파일은 아래와 같다.
  - `scripts/_grasp_refine_bridge.py`
  - `grasp_refine/refine.py`
  - `grasp_gen/hand_contacts.py`
  - `grasp_gen/grasp_energy.py`
  - `grasp_gen/grasp_optimizer.py`

## 한 줄 요약

현재 refine 단계는 "어떤 세그먼트를 쓸지"를 새로 고르지 않는다.

세그먼트 선택은 이미 upstream에서 끝나 있고, refine는 그 결과로 들어온
`contact_indices`를 고정한 채 손 자세만 움직인다. 다만 세 항은 같은 세그먼트
집합을 보지 않는다.

- `surface_pull`
  - 고정된 `contact_indices`가 가리키는 세그먼트만 직접 본다.
- `external_repulsion`
  - 손 전체 dense cloud를 보지만, 현재 구현은 `max(...)`라서 매 step마다
    가장 나쁜 침투 지점이 있는 세그먼트가 사실상 gradient를 지배한다.
- `self_repulsion`
  - 선택된 contact point 자체가 아니라, contact-enabled body마다 대표점
    하나만 써서 body 단위로 작동한다.

## 세그먼트 집합이 어디서 오나

`grasp_gen/hand.py`의 `SEGS`가 contact-enabled finger segment의 기준이다.

- `thumb_1`
- `thumb_0`
- `index_1`
- `index_0`
- `middle_1`
- `middle_0`
- `ring_1`
- `ring_0`
- `pinky_1`
- `pinky_0`

`grasp_gen/hand_contacts.py`의 역할 매핑은 아래와 같다.

- `segment == "1"` -> `proximal`
- thumb의 `segment == "0"` -> `distal`
- 나머지 손가락의 `segment == "0"` -> `intermediate`

중요한 점:

- MuJoCo asset에는 `collision_hand_right_palm_0` 같은 palm collision geom도
  있다.
- 하지만 contact sampling은 `SEGS`의 10개 finger segment만 순회한다.
- 그래서 palm은 selected contact pool에는 들어가지 않지만,
  `external_repulsion`용 dense cloud에는 들어갈 수 있다.

## 세그먼트 선택이 실제로 이루어지는 단계

### 1. 손 전체 surface cloud 생성

`grasp_gen/hand_contacts.py`의 `build_surface_cloud(...)`는
`collision_hand_{side}_`로 시작하는 모든 collision geom에서 표면 점을
샘플링한다.

각 geom에 대해:

- geom type / size로 표면적을 추정하고
- `target_spacing`으로 목표 점 개수를 잡고
- 표면을 oversample한 뒤
- farthest-point sampling으로 줄이고
- 마지막에 `cloud_scale`만큼 손 루트 기준 바깥으로 약간 밀어낸다

즉, 이 단계는 특정 손가락 세그먼트만이 아니라 손 전체 collision surface에 대한
dense cloud를 만든다.

### 2. 세그먼트별 contact-point pool 생성

`sample_contacts(...)`가 위 dense cloud를 받아 실제 contact pool을 만든다.

각 세그먼트마다 다음 순서를 따른다.

1. `_frame(...)`로 세그먼트의 local frame을 만든다.
   - 축(`axis`)
   - 바깥쪽 면 방향(`face`)
   - 접선(`tangent`)
   - 중심과 half-length
2. `_blocked(...)`로 joint 주변, child joint 주변, palm 쪽 proximal 구간을
   막힌 축 구간으로 만든다.
3. `_contact_mask(...)`로 후보점을 거른다.
   - 세그먼트 길이 안에 있어야 하고
   - blocked interval 밖이어야 하고
   - 축 중심선 위 점은 제외하고
   - radial 방향이 `face` 주변 cone 안에 있어야 한다
4. 너무 적게 남으면 축 길이 조건만 남긴 relaxed 후보로 완화한다.
5. 그래도 부족하면 해당 세그먼트 raw candidate 전체로 되돌린다.
6. 최종적으로 farthest-point sampling으로 `n_per_seg`개를 고른다.

이 단계의 의미:

- contact pool은 "세그먼트당 동일 개수"로 균형 잡혀 생성된다
- 하지만 이것은 pool 수준의 균형일 뿐
- 최종 active contact가 세그먼트별로 균형 잡힌다는 뜻은 아니다

### 3. upstream optimizer가 active contact를 고른다

`grasp_gen/grasp_optimizer.py`를 보면 optimizer는 flatten된 contact pool 위에서
`contact_indices`를 고른다.

현재 구현은:

- 초기 `contact_indices`를 전체 pool 범위에서 uniform random integer로 뽑고
- 이후 각 slot을 `switch_possibility`에 따라 다른 random index로 바꿀 수 있다
- switching도 전체 pool 범위에서 uniform하다

즉:

- 각 contact slot은 어떤 세그먼트의 샘플 포인트로도 갈 수 있고
- 같은 세그먼트가 여러 slot을 차지할 수도 있고
- 어떤 세그먼트는 하나도 선택되지 않을 수도 있다

### 4. refine는 세그먼트를 다시 고르지 않는다

`grasp_refine`는 upstream artifact에 들어 있던 `contact_indices`를 그대로
받아 refine 내내 유지한다.

`scripts/_grasp_refine_bridge.py`의 callback은 `contact_indices`를 입력으로 받아
에너지와 gradient를 계산할 뿐, refine 중에 contact index를 바꾸는 로직이 없다.

이게 "조정할 세그먼트 선택 방법"에 대한 가장 중요한 결론이다.

- 세그먼트 선택은 refine 안에서 일어나지 않는다
- upstream contact selection 결과가 그대로 얼어붙는다
- refine는 그 고정된 선택 위에서 pose만 조정한다

## `contact_indices`는 정확히 무엇인가

`contact_indices`는 "세그먼트 id"가 아니라, flatten된 contact pool에서의 정수
인덱스다.

이 인덱스가 만들어지는 방식은 다음과 같다.

1. `sample_contacts(...)`가 `SEGS` 순서대로 세그먼트를 순회한다.
2. 각 세그먼트에서 정확히 `n_per_seg`개의 contact point를 뽑는다.
3. 그 결과를 append 방식으로 이어 붙인다.

따라서 현재 구현에서는:

- 같은 `n_per_seg`를 쓰는 한
- `contact_indices`는 세그먼트 블록 위의 정수 슬롯처럼 해석할 수 있다

예를 들어 `n_per_seg = 10`이면 현재 ordering은 아래와 같다.

- `0..9` -> `thumb_1`
- `10..19` -> `thumb_0`
- `20..29` -> `index_1`
- `30..39` -> `index_0`
- `40..49` -> `middle_1`
- `50..59` -> `middle_0`
- `60..69` -> `ring_1`
- `70..79` -> `ring_0`
- `80..89` -> `pinky_1`
- `90..99` -> `pinky_0`

중요한 점:

- 이 ordering은 "세그먼트별 균등 블록"이라는 점에서는 안정적이지만
- 각 세그먼트 블록 내부에서 어떤 표면점이 몇 번 인덱스가 되는지는
  `sample_contacts(...)`의 FPS 결과 순서에 따라 정해진다
- 즉 `45`가 `middle_1`이라는 사실은 안정적이지만,
  `middle_1` 내부에서 정확히 어떤 위치의 포인트인지는 블록 내부 ordering까지
  봐야 한다

## `contact_indices`는 처음 어떻게 뽑히나

`grasp_gen/grasp_optimizer.py`의 `init_state(...)`에서 initial
`contact_indices`가 만들어진다.

현재 기본 경로는:

- `initial_contact_indices`를 외부에서 따로 주지 않으면
- `jax.random.randint(...)`로 전체 pool 범위에서 uniform random으로 뽑는다

즉 초기 선택의 성격은:

- segment-aware heuristic이 없다
- 전체 flatten된 contact pool 위의 random selection이다
- 중복 인덱스도 허용된다

그래서 초기 상태부터:

- 같은 세그먼트가 여러 slot을 차지할 수 있고
- 어떤 세그먼트는 하나도 선택되지 않을 수 있다

## optimizer 중에 `contact_indices`는 어떻게 바뀌나

optimizer step에서는 pose update와 contact index switching이 같이 일어난다.

`grasp_gen/grasp_optimizer.py`의 step logic은:

1. 현재 pose에서 gradient 기반 제안을 만든다
2. 각 contact slot마다 switching 여부를 Bernoulli로 뽑는다
3. switch가 켜진 slot은 전체 pool 범위의 새 random index로 바꾼다
4. 바뀐 `proposed_contact_indices`와 새 pose를 함께 평가한다
5. Metropolis-style accept/reject를 한 뒤, 통과되면 `contact_indices`도 함께 갱신한다
6. 그 상태가 더 좋은 energy면 `best_contact_indices`로 저장한다

즉 refine가 읽는 것은 보통:

- 마지막 current index set인 `contact_indices`
- 또는 best energy 시점의 `best_contact_indices`

## switching 확률 스케줄

기본 `GraspBatchOptimizerConfig`는 아래 값을 쓴다.

- `switch_possibility = 0.1`
- `switch_start_step = 250`
- `switch_ramp_steps = 1000`

현재 구현상 실제 slot별 switching 확률은:

- step `< 250` 에서는 `0`
- step `250..1250` 구간에서는 `0 -> 0.1`로 선형 증가
- step `> 1250` 에서는 `0.1`

즉 초반에는 contact selection을 거의 고정하고 pose를 먼저 잡고,
중반 이후부터 contact slot이 조금씩 다른 세그먼트/포인트로 이동할 수 있게 해
둔 구조다.

## 실제 artifact에서 확인한 예시

로컬 결과 파일
`outputs/grasp_optimizer/run_cylinder_b64_s5000_seed0_eqwrench_pen.npz`를 실제로
읽어 확인한 내용은 아래와 같다.

- `n_per_seg = 10`
- `contact_count = 4`
- `switch_possibility = 0.1`
- `switch_start_step = 250`
- `switch_ramp_steps = 1000`
- contact pool 크기 = `10 segments * 10 = 100`
- `best_contact_indices` shape = `(64, 4)`

이 run의 best sample은 index `18`이고:

- `best_contact_indices = [45, 30, 12, 70]`
- 이는 각각
  - `45 -> middle_1`
  - `30 -> index_0`
  - `12 -> thumb_0`
  - `70 -> ring_0`
  로 해석된다

즉 이 sample에서 `surface_pull`이 직접 보는 세그먼트는:

- `middle_1`
- `index_0`
- `thumb_0`
- `ring_0`

였다.

또 같은 artifact 전체 64개 sample을 보면:

- exact duplicate index가 있는 row가 `12/64`
- 같은 세그먼트가 두 번 이상 등장하는 row가 `47/64`

였다.

이건 현재 선택 로직이 아래 제약을 두지 않는다는 뜻이다.

- "한 세그먼트당 최대 1개"
- "서로 다른 손가락만 선택"
- "distal/proximal 균형 유지"

즉 현재 `contact_indices`는 세그먼트 균형이 보장된 contact pool 위에서
uniform-random initialization과 slot-wise random switching을 거친 결과라고 보는
것이 가장 정확하다.

## 현재 `RefineEnergyTerms` 필드가 실제로 뜻하는 것

브리지 코드에서 `RefineEnergyTerms`의 필드는 예전 에너지 이름을 재사용하고
있다. 현재 의미는 아래와 같다.

- `total` -> 세 항의 가중합
- `distance` -> 실제로는 `surface_pull`
- `penetration` -> 실제로는 `external_repulsion`
- `contact` -> 실제로는 `self_repulsion`
- `equilibrium` -> 항상 `0.0`
- `root_reg` -> 항상 `0.0`
- `joint_reg` -> 항상 `0.0`

그래서 history도 이름과 실제 의미가 다르다.

- `history_distance`는 `surface_pull` 이력
- `history_penetration`은 `external_repulsion` 이력
- `history_contact`는 `self_repulsion` 이력

## `surface_pull`

### 현재 구현

`scripts/_grasp_refine_bridge.py`에서 `surface_pull`은 selected contact point만
쓴다.

흐름은 다음과 같다.

1. `contact_indices`가 가리키는 contact point를 고른다.
2. 그것을 object-local로 변환한다.
3. triangle nearest point를 찾는다.
4. unsigned nearest-surface distance를 계산한다.
5. `surface_pull_threshold` 이하인 점만 active로 둔다.
6. active 점들의 거리를 평균낸다.

최종 항은:

- `surface_pull_weight * surface_pull`

### 어떤 세그먼트를 조정하나

직접적으로는 selected contact point가 속한 세그먼트만 조정한다.

즉 active segment 집합은:

- 고정된 `contact_indices`에 실제로 등장하는 세그먼트

이 말은 곧:

- 어떤 세그먼트가 `contact_indices`에 없으면 direct `surface_pull`을 받지 못한다
- 반대로 같은 세그먼트에 contact가 여러 개 몰리면 그 세그먼트 비중이 커진다
- 선택된 세그먼트 구성이 나빠도 refine는 그것을 더 좋은 세그먼트로 바꾸지 못한다

### 구현 상태

이미 구현된 것:

- `surface_pull_threshold` 기반의 near-surface gate
- active selected contact에 대한 평균 거리
- projected hand pose에 대한 gradient 전파

아직 안 들어간 것:

- `contact_target_local`을 향한 explicit anchor term
- refine 중 contact reselection
- 실제 overlap이 남아 있을 때 pull을 꺼 주는 gating

코드상 근거:

- `contact_target_local`은 계산되고 저장된다
- 하지만 single-sample callback에서
  `del contact_indices_, contact_target_local_, cfg`로 버려진다

따라서 현재 `surface_pull`의 성격은:

- "원래 contact anchor로 복귀시키는 힘"이 아니라
- "이미 가깝게 붙어 있는 selected point를 표면 근처에 계속 두는 힘"

## `external_repulsion`

### 현재 구현

`external_repulsion`은 `contact_indices`가 아니라 dense hand cloud를 쓴다.

흐름은 다음과 같다.

1. `cloud_body_indices` / `cloud_local_positions` 전체를 가져온다.
2. world로 보낸 뒤 object-local로 변환한다.
3. 각 점마다 nearest triangle과 normal을 찾는다.
4. `(nearest_local - cloud_local) . nearest_normal`로 부호를 추정한다.
5. 그 부호가 양수인 점만 penetration candidate로 본다.
6. 그 점들 중 unsigned distance의 최대값만 남긴다.

최종 항은:

- `external_repulsion_weight * external_repulsion`

### 어떤 세그먼트를 조정하나

명목상으로는 dense cloud에 점이 있는 모든 세그먼트가 후보가 된다. palm도 여기에
포함될 수 있다.

하지만 현재 reduction이 `max(...)`라서 실제로는:

- 가장 깊게 침투한 한 점이 1차 gradient를 지배한다
- 즉 그 step에서 사실상 active한 세그먼트는 argmax penetration이 있는 세그먼트다

의미:

- 여러 세그먼트에 얕은 침투가 동시에 있어도
- 가장 나쁜 한 지점이 먼저 잡히고
- 나머지는 그 지점이 줄어들기 전까지 gradient가 약해질 수 있다

### 구현 상태

이미 구현된 것:

- dense-cloud 기반 hand/object overlap 체크
- nearest-triangle normal 기반 sign 판정
- max-based penalty

아직 안 들어간 것:

- `external_threshold`를 에너지 내부 deadband로 쓰는 것
- 모든 침투점 누적합/평균합 방식
- actual physics collision count/depth를 optimize target으로 직접 쓰는 것

관련 디테일:

- `external_threshold`는 `RefineConfig`에는 있다
- 하지만 현재 single-sample 에너지에는 연결되지 않는다
- `grasp_refine/batch.py`에서 "fixed 판정"용 mask를 만들 때만 쓰인다

## `self_repulsion`

### 현재 구현

`self_repulsion`은 dense cloud 전체를 쓰지 않는다.

`scripts/_grasp_refine_bridge.py`의 `_self_repulsion_points(...)`가 먼저
대표점을 만든다.

흐름은 다음과 같다.

1. `contact_body_indices`에 등장하는 unique body를 모은다.
2. 각 body에 속한 dense cloud point를 모은다.
3. 그 body의 dense cloud local point 평균을 representative point 하나로 만든다.

그 다음 `self_repulsion`은:

1. representative point를 world로 변환하고
2. 모든 pairwise distance를 계산하고
3. diagonal은 무시하고
4. `self_repulsion_threshold`보다 가까운 쌍에 대해 부족분을 합산한다

최종 항은:

- `self_repulsion_weight * self_repulsion`

### 어떤 세그먼트를 조정하나

이 항이 보는 대상은 selected contact point 자체가 아니다.

실제 active 집합은:

- contact-enabled body마다 대표점 하나

실제로는 대체로:

- finger segment body당 하나의 coarse representative point
- selected contact 개수와는 무관
- dense cloud의 모든 점을 pairwise로 보지도 않음
- 보통 palm은 제외됨

이유는:

- representative body 집합의 출처가 `contact_body_indices`인데
- 이것이 원래 sampled segment contact pool에서 오기 때문이다

### 구현 상태

이미 구현된 것:

- body-level representative point 생성
- thresholded pairwise repulsion

아직 안 들어간 것:

- dense self-collision 전체를 직접 보는 방식
- 세그먼트 내부의 미세한 self-intersection을 잡는 고해상도 항

따라서 현재 항의 성격은:

- body 간 간격 벌리기에는 의미가 있지만
- 한 세그먼트 내부나 body 표면의 세밀한 충돌은 잘 못 본다

## "조정할 세그먼트"를 항별로 다시 정리하면

질문이 "지금 이 항이 실제로 어느 세그먼트를 움직일 수 있나?"라면 답은
다음과 같다.

- `surface_pull`
  - frozen `contact_indices`에 포함된 세그먼트만
- `external_repulsion`
  - dense cloud가 덮는 전 세그먼트 후보
  - 하지만 실질적으로는 현재 worst penetration 세그먼트
- `self_repulsion`
  - contact-enabled body마다 하나의 representative point
  - 대체로 finger segment body 수준

질문이 "그 세그먼트 집합은 어디서 결정되나?"라면 답은 다음과 같다.

- 세그먼트별 candidate contact pool은 `sample_contacts(...)`가 만든다
- 실제 active contact set은 upstream optimizer의 random selection /
  switching이 만든다
- refine는 그 결과를 그대로 받는다

## 왜 이 구분이 튜닝에 중요한가

현재 코드에서 가장 중요한 실무적 함의는 이것이다.

- refine weight를 바꾼다고 해서 더 좋은 `surface_pull` 세그먼트를 새로 만들 수는 없다
- 바꿀 수 있는 것은 "이미 선택된 세그먼트를 얼마나 세게 쓸지"뿐이다

즉, 문제가 "pose가 조금 틀어져 있다"인지, 아니면 "애초에 active segment
selection이 잘못됐다"인지 분리해서 봐야 한다.

1. pose-only 문제라면
   - `surface_pull_weight`
   - `external_repulsion_weight`
   - `self_repulsion_weight`
   - threshold
   - `guidance_scale`
   - `grad_scale`
   튜닝이 의미가 있다.
2. segment-selection 문제라면
   - upstream contact sampling
   - upstream switching policy
   - refine-time contact reselection
   쪽을 손봐야 한다.

## 코드에서 바로 보이는 현재 구현 공백

코드만 보고 정리하면, 현재 주요 공백은 아래와 같다.

- `contact_target_local`은 계산되지만 refine energy에서 사용되지 않는다
- `external_threshold`는 single-sample energy에 연결되어 있지 않다
- `surface_pull`은 anchor recovery가 아니라 near-surface maintenance다
- `external_repulsion`은 worst-point-only다
- `self_repulsion`은 body당 mean point 하나만 쓴다
- actual overlap metric은 기록되지만 best pose 선택 기준은 여전히 surrogate total이다

## 짧은 결론

현재 `grasp_refine`는 "세그먼트를 다시 고르는 단계"가 아니라
"이미 선택된 세그먼트 집합 위에서 pose만 조정하는 DGA-style refine"이다.

세 항은 모두 구현되어 있지만, 세 항이 보는 세그먼트 집합은 서로 다르다.

- `surface_pull`은 selected contact segment를 본다
- `external_repulsion`은 손 전체 dense cloud를 보지만 worst segment가 지배한다
- `self_repulsion`은 contact-enabled body 대표점만 본다

그래서 "어떤 세그먼트를 조정해야 하는가"를 논리적으로 설명하려면,
가중치뿐 아니라 active segment set이 어디서 고정되는지도 같이 봐야 한다.
현재 코드에서는 그 고정 지점이 refine 이전, 즉 upstream contact selection 단계다.
