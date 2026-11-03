"""제출 파일.

필요한 건 두 가지임.

    load_model(category)    그 카테고리를 예측할 모델을 돌려줌
    Model.score_map(images) (n, 256, 256) 이상 맵. 클수록 이상

선택으로 둘 수 있는 것.

    Model.score(images)     (n,) 이미지 점수. 없으면 맵의 최댓값을 씀
    Model.threshold         판정 임계값 (float). 없으면 val 정상 점수의 상위 1% 를 씀.
                            load_model() 에서 artifacts 에 저장해 둔 값을 읽어 넣으면 됨

입력은 항상 (n, 256, 256, 3) uint8 원시 이미지임. 전처리는 모델 안에서 함.
그래야 학습과 추론에서 같은 처리가 보장됨.

학습 방법이나 장소에는 제약이 없음. 노트북, Colab, 대여 GPU 어디서 돌려도 되고
학습 결과를 `artifacts/` 에 저장해 두고 load_model() 에서 읽기만 하면 됨.
채점할 때는 학습을 다시 돌리지 않음.

    python main.py train     아래 train() 으로 전체 카테고리를 한 번에 학습해 저장

완성된 예시는 example/sample_model.py 에 있음.
"""

import numpy as np

from .paths import ARTIFACTS

class Model:
    """정상 이미지들의 평균에서 픽셀마다 얼마나 먼지. 여기서부터 고쳐 나가면 됨.

    위치도 질감도 보지 않는 아주 약한 방법이라 AUROC 가 0.5 근처임.
    """

    def fit(self, images):
        """정상 이미지 (n, 256, 256, 3) uint8 로 학습."""
        self.mean = np.asarray(images, dtype=np.float32).mean(axis=0)
        return self

    def score_map(self, images):
        """(n, 256, 256, 3) uint8 -> (n, 256, 256) float. 반드시 원본 해상도로."""
        d = np.asarray(images, dtype=np.float32) - self.mean
        return (d ** 2).sum(axis=3)

    def score(self, images):
        """(n,) 이미지 점수. 여기서는 맵의 최댓값을 씀."""
        m = self.score_map(images)
        return m.reshape(len(m), -1).max(axis=1)

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "mean.npy", self.mean)

    @classmethod
    def load(cls, folder):
        m = cls()
        m.mean = np.load(folder / "mean.npy")
        return m


def train(category, images):
    """카테고리 하나를 학습해서 artifacts/<category>/ 에 저장함. `main.py train` 이 부름.

    images 는 train 의 정상 이미지임. 다른 방법으로 학습했다면 이 함수는 안 써도 됨.
    """
    Model().fit(images).save(ARTIFACTS / category)


def load_model(category):
    """그 카테고리를 예측할 모델을 돌려줌. 전체 데이터로 학습했다면 category 는 무시해도 됨.

    채점할 때 이 함수로 모델을 가져와 score_map 만 돌림. 학습은 다시 하지 않음.

    클래스를 통째로 pickle 하면 안 됨. 노트북에서 정의한 클래스는 다른 곳에서 읽을 때
    깨짐. 학습 결과(배열, state_dict)만 저장하고 클래스는 이 파일에 둠.
    torch 를 쓴다면 state_dict 를 저장하고 여기서 불러옴.
    """
    folder = ARTIFACTS / category
    if not folder.exists():
        raise FileNotFoundError(f"{folder} 가 없음. 먼저 학습해서 저장할 것 (python main.py train)")
    return Model.load(folder)
