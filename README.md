# 제조 비전 이상탐지

정상 이미지만으로 학습해서, 새 이미지가 불량인지 판정하고 결함 위치를 이상 맵으로 보여 주는 문제임.
**학습 데이터에는 불량 이미지가 한 장도 없음.**

| 내는 것 | 형식 |
|---|---|
| 이상 맵 | 이미지마다 `(256, 256)`. 클수록 이상 |
| 이미지 점수 | 이미지마다 숫자 하나. 클수록 이상 |
| 판정 | 점수가 임계값 이상이면 불량 |

채점은 38개 카테고리 전체로 함. 지표를 카테고리별로 계산한 뒤 평균함.

학습 방식은 자유임. 전체를 섞어 학습해도 되고 카테고리별로 따로 학습해도 됨.

일정, 제출 방법, 배점은 수업 공지를 따름.

---

## 시작

```bash
uv sync                                  # 가상환경과 패키지. uv.lock 이 생김
unzip dataset.zip                        # dataset/ 이 생김

uv run python main.py train              # src/model.py 의 train() 으로 카테고리별 학습. 다른 방식이면 안 써도 됨
uv run python main.py check              # 출력 규격 검사
uv run python main.py eval               # val 점수
uv run python main.py baselines          # 참고용 점수 (무작위, 평균 이미지 거리)
```

torch 가 들어 있어 첫 설치는 오래 걸림.

실험할 때는 `--categories agri_01 mech_04` 로 몇 개만, `--limit 100` 으로 카테고리당 100장만 돌릴 수 있음.
마지막에는 전체로 `train` 과 `check` 를 돌려 볼 것.

## 구조

```
main.py                 학습, 검사, 채점
src/
  model.py                제출 파일. 여기를 채우면 됨
  data.py                 데이터 읽기
  schema.py               맵과 점수의 형식 검사
  metrics.py              지표
  evaluate.py             예측, 임계값, 채점
  explain.py              해석 도구 (occlusion, 비슷한 정상 이미지, 시각화)
  paths.py                폴더 위치
  features/
    patches.py              칸별 색·질감 특징, 맵 업샘플
    backbone.py             사전학습 ResNet 특징
example/
  1_eda.ipynb             데이터 살펴보기
  2_modeling.ipynb        모델 만들기
  sample_model.py         완성된 모델 예시
dataset/                데이터
artifacts/              학습 결과
```

---

## 지켜야 할 것

```python
# src/model.py
def load_model(category): ...         # 그 카테고리를 예측할 모델을 돌려줌
class Model:
    def score_map(self, images): ...  # (n, 256, 256, 3) uint8 -> (n, 256, 256)
    def score(self, images): ...      # (선택) (n,). 없으면 맵의 최댓값
    threshold = ...                   # (선택) 판정 임계값. 없으면 val 정상 점수의 상위 1%
```

학습 방법이나 위치에는 제약이 없음. 결과를 `artifacts/` 에 저장해 두고 `load_model()` 에서
읽기만 하면 됨. 전체로 학습한 모델 하나를 쓴다면 `category` 는 무시하고 같은 모델을 돌려주면 됨.
채점할 때는 학습을 다시 돌리지 않음.

클래스를 통째로 pickle 하지 말고 배열이나 `state_dict` 만 저장할 것. 노트북에서 정의한 클래스는 다른 곳에서
읽을 때 깨짐.

완성된 예시가 `example/sample_model.py` 에 있음. 클래스 이름을 `Model` 로, import 를
`from .features import ...` 로 바꿔 `src/model.py` 에 붙이면 됨.

### 출력 규격

| | 규칙 |
|---|---|
| 입력 | 항상 `(n, 256, 256, 3)` uint8 원본 이미지. 전처리는 모델 안에서 |
| `score_map` | `(n, 256, 256)`, NaN·inf 없음. 특징 격자를 **원본 해상도로 키워서** 돌려줄 것 |
| `score` | `(n,)`, NaN·inf 없음 |
| `threshold` | 유한한 실수 |

### 데이터 사용

- **train 정상 이미지만으로 학습함.** 정상 이미지에 가짜 결함을 합성해 쓰는 것은 괜찮음.
- **val 은 학습에 쓰지 않음.** 정상이든 불량이든 가중치나 메모리뱅크에 넣으면 안 됨.
  하이퍼파라미터와 임계값을 정하고 해석을 검증하는 데만 씀.
- 라이브러리는 자유임. 추가한 패키지는 `pyproject.toml` 과 `uv.lock` 에 넣을 것.

---

## 예측과 채점

```python
from src import Dataset, load_model, predict, evaluate

cat = Dataset().category("agri_02")
res, maps, thr = predict(load_model("agri_02"), cat, "val")    # 이미지 점수·판정, 맵, 임계값

evaluate(["agri_01", "mech_04"], "val")                        # 카테고리마다 predict 돌리고 점수까지
```

`eval` 은 `predict` 를 카테고리마다 반복하고 정답과 맞춰 점수를 내는 게 전부임.
결과는 `--out scores.csv`(이미지별 점수), `--table table.csv`(카테고리별 지표)로 저장할 수 있음.

---

## 예제

**`1_eda.ipynb`** — 데이터 살펴보기

무엇이 들어 있는지, 결함 유형과 면적, 정상끼리 얼마나 다른지, 쓸 만한 신호가 있는지 확인함.

**`2_modeling.ipynb`** — 모델 만들기

1. 카테고리 하나 잡기
2. 기준선 — 평균영상 거리
3. 칸 단위 특징
4. 패치 최근접 이웃
5. 맵에서 이미지 점수 뽑기
6. 전체로 돌려 보기 — 임계값, ROC, PRO 곡선, 사례 갤러리
7. 맵이 결함을 제대로 짚었나
8. 사전학습 CNN 특징
9. `model.py` 채우기

---

## 데이터

```python
from src import Dataset

ds = Dataset()
ds.categories            # 38개
ds.domains               # {"agriculture": ["agri_01", ...], ...}

cat = ds.category("agri_01")
cat.train                # 정상만 (DataFrame)
cat.val                  # 정상 + 불량
cat.images("train")      # (n, 256, 256, 3) uint8
cat.masks("val")         # (n, 256, 256) 0/1. 정상은 전부 0

whole = ds.all()         # 카테고리 구분 없이 전체. 쓰는 법은 같음
whole.train              # category 열로 어느 카테고리인지 알 수 있음
for rows, x in whole.batches("train"): ...    # 배치로 읽기
```

전체 train 은 42,966장이라 `whole.images("train")` 으로 한 번에 올리면 메모리가 약 8GB 필요함.

```
dataset/
  index.csv                                  이미지 목록. 한 줄이 한 장
  <도메인>/<카테고리>/
    train/good/0001.png                      학습용 정상
    val/good/0001.png                        검증용 정상
    val/<결함 유형>/0001.png                 검증용 불량
    ground_truth/val/<결함 유형>/0001.png    정답 마스크 (같은 파일명)
```

`index.csv` 의 열은 `group`(도메인), `category`, `split`, `label`(good 또는 결함 유형), `path`, `mask_path`,
`mask_area_pct`(결함 면적 %), `is_anomaly` 임.

| 도메인 | 카테고리 | train | val 정상 | val 불량 |
|---|---:|---:|---:|---:|
| agriculture | 4 | 4,000 | 400 | 198 |
| electronics | 11 | 13,490 | 685 | 338 |
| groceries | 3 | 3,000 | 300 | 150 |
| mechanics | 11 | 12,626 | 829 | 402 |
| medicine | 9 | 9,850 | 756 | 379 |

카테고리당 train 은 1,000~1,462장, val 불량은 9~52장임. val 불량이 적은 카테고리는 몇 장 맞히느냐에 따라
점수가 크게 흔들림. 카테고리별 상세는 `1_eda.ipynb` 에 있음.

---

## 평가

| 지표 | 평가 대상 | 보는 것 |
|---|---|---|
| **AUROC** | 이미지 점수 | 정상과 불량을 얼마나 잘 가르는지 |
| **F1** | 판정 | 임계값으로 판정했을 때 precision 과 recall 의 조화평균 |
| **AUPRO** | 이상 맵 | 맵이 결함 영역을 얼마나 덮는지. 작은 결함도 같은 무게 |

세 지표를 카테고리마다 계산하고, 지표별로 38개 카테고리의 평균을 냄.
**최종 점수는 이 세 평균의 산술평균임.**

```
AUROC = 카테고리별 AUROC 의 평균    (F1, AUPRO 도 같음)
최종 점수 = (AUROC + F1 + AUPRO) / 3
```

임계값은 val 을 보고 정하면 됨. 모델에 `threshold` 를 넣어 두면 test 에도
그 값을 그대로 씀.

### 채점

- 채점은 배포하지 않은 비공개 test 로 하고, `main.py eval` 과 같은 코드로 점수를 냄.
- 채점자가 학습한 모델과 추론 코드를 받아 직접 실행함.
