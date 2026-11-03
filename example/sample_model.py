"""완성된 모델 예시.

칸마다 색·질감을 재고(patch_stats), 정상 칸들을 메모리뱅크에 모아 둔 뒤
테스트 칸이 가장 가까운 정상 칸에서 얼마나 먼지를 이상 점수로 씀.
PatchCore 를 아주 단순하게 줄인 것임. 역전파 학습이 없어 CPU 로 금방 돌아감.

    from sample_model import SampleModel

    model = SampleModel().fit(train_images)
    model.score_map(test_images)      # (n, 256, 256)

`src/model.py` 로 옮길 때는 두 군데만 바꾸면 됨.

    from src.features import ...   ->  from .features import ...    (sys.path 줄은 지움)
    class SampleModel              ->  class Model

`train()` / `load_model()` 은 save / load 를 그대로 부르므로 고칠 필요가 없음.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import blur, patch_stats, upsample


class SampleModel:
    """패치 최근접 이웃.

    Args:
        cell: 칸 크기(px). 작을수록 작은 결함을 잡지만 잡음이 늘어남.
        bank_size: 메모리뱅크에 남길 정상 칸 수. 크면 정확하지만 느림.
        top: 이미지 점수로 쓸 맵 상위 비율. max 는 칸 하나의 잡음에 흔들림.
    """

    def __init__(self, cell=8, bank_size=20000, top=0.01, seed=0):
        self.cell, self.bank_size, self.top, self.seed = cell, bank_size, top, seed

    def _features(self, images):
        f = patch_stats(images, self.cell)              # (n, h, w, 8)
        return (f - self.mu) / self.sd if hasattr(self, "mu") else f

    def fit(self, images):
        f = patch_stats(images, self.cell)
        flat = f.reshape(-1, f.shape[-1])
        # 특징마다 단위가 달라서 정상 기준으로 표준화함
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        flat = (flat - self.mu) / self.sd

        rng = np.random.default_rng(self.seed)
        keep = rng.choice(len(flat), min(self.bank_size, len(flat)), replace=False)
        self.bank = flat[keep].astype(np.float32)
        self.bank_sq = (self.bank ** 2).sum(1)
        return self

    def _nearest(self, q, chunk=8192):
        """q 의 각 행에서 메모리뱅크 최근접 거리."""
        out = np.empty(len(q), dtype=np.float32)
        for i in range(0, len(q), chunk):
            x = q[i:i + chunk]
            d = (x ** 2).sum(1)[:, None] + self.bank_sq[None] - 2 * x @ self.bank.T
            out[i:i + chunk] = np.sqrt(np.maximum(d.min(1), 0))
        return out

    def score_map(self, images):
        f = self._features(images)
        n, h, w, d = f.shape
        grid = self._nearest(f.reshape(-1, d).astype(np.float32)).reshape(n, h, w)
        return blur(upsample(grid), sigma=4)

    def score(self, images):
        m = self.score_map(images).reshape(len(images), -1)
        k = max(1, int(m.shape[1] * self.top))
        return np.sort(m, axis=1)[:, -k:].mean(axis=1)

    # ------------------------------------------------------------ 저장과 불러오기
    # 학습 결과(배열)만 저장함. 클래스를 통째로 pickle 하면 다른 곳에서 읽을 때 깨짐.

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "model.npz", mu=self.mu, sd=self.sd, bank=self.bank,
                 config=np.array([self.cell, self.bank_size, self.top, self.seed]))

    @classmethod
    def load(cls, folder):
        z = np.load(folder / "model.npz")
        cell, bank_size, top, seed = z["config"]
        m = cls(int(cell), int(bank_size), float(top), int(seed))
        m.mu, m.sd, m.bank = z["mu"], z["sd"], z["bank"]
        m.bank_sq = (m.bank ** 2).sum(1)
        return m
