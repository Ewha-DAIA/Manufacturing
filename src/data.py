"""데이터를 읽고 카테고리별로 나눔.

    ds = Dataset()
    cat = ds.category("agri_02")

    cat.train               # 학습용. 정상만 (DataFrame)
    cat.val                 # 검증용. 정상과 불량이 섞여 있음
    cat.test                # 채점용. 배포 데이터에는 없음 (빈 표)

    cat.images("train")     # (n, 256, 256, 3) uint8
    cat.masks("val")        # (n, 256, 256) 0/1. 평가에만 씀

`dataset/index.csv` 한 줄이 이미지 한 장임.

    is_anomaly       1 이면 불량
    mask_area_pct    이미지에서 결함이 차지하는 면적 %. 정상은 0
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATASET

IMAGE_SIZE = 256

THRESHOLD_QUANTILE = 0.99   # val 정상 이미지 이상 점수의 상위 1% 지점이 임계값

# 결함 면적으로 불량을 나누는 구간 (mask_area_pct 기준). 작은 결함이 훨씬 어려움.
SIZE_BINS = [(0.0, 1.0, "소"), (1.0, 5.0, "중"), (5.0, 100.0, "대")]

SEED = 42


class Category:
    """카테고리 하나. train 은 전부 정상, val 과 test 는 정상과 불량이 섞여 있음."""

    def __init__(self, name, train, val, test, data):
        self.name = name
        self.train = train
        self.val = val
        self.test = test        # 배포 데이터에서는 비어 있음
        self._data = data

    def __repr__(self):
        v = self.val
        return (f"Category({self.name}: train {len(self.train)} / val {len(v)} 장 "
                f"(불량 {int(v.is_anomaly.sum())}) / test {len(self.test)} 장)")

    def _rows(self, part):
        return getattr(self, part) if isinstance(part, str) else part

    def images(self, part="train", limit=None):
        """이미지. (n, 256, 256, 3) uint8.

        Args:
            part: "train" / "val" / "test", 또는 그 일부를 고른 DataFrame.
            limit: 앞에서부터 몇 장만. 빠르게 돌려볼 때 씀.
        """
        rows = self._rows(part)
        rows = rows if limit is None else rows.head(limit)
        if rows.empty:
            return np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return np.stack([load_image(p, self._data) for p in rows["path"]])

    def masks(self, part="val", limit=None):
        """정답 마스크. (n, 256, 256) 0/1.

        평가에만 씀. 학습에 쓰면 지도학습이 됨.
        """
        rows = self._rows(part)
        rows = rows if limit is None else rows.head(limit)
        return np.stack([load_mask(p, self._data) for p in rows["mask_path"]])

    def batches(self, part="val", batch=64, limit=None):
        """이미지를 배치로 흘려보냄. 한 번에 다 올리지 않을 때 씀."""
        rows = self._rows(part)
        rows = rows if limit is None else rows.head(limit)
        for i in range(0, len(rows), batch):
            chunk = rows.iloc[i:i + batch]
            yield chunk, self.images(chunk)

    def summary(self):
        v = self.val
        return {"category": self.name, "train": len(self.train),
                "val_normal": int((v.is_anomaly == 0).sum()),
                "val_anomalous": int((v.is_anomaly == 1).sum()),
                "defect_types": int(v[v.is_anomaly == 1].label.nunique()),
                "test": len(self.test)}


class Dataset:
    """dataset/index.csv 를 읽어 카테고리별로 나눠 줌.

    Args:
        data: 데이터 폴더. 생략하면 dataset/.
        categories: 쓸 카테고리. 생략하면 전부.
    """

    def __init__(self, data=None, categories=None):
        self.data = Path(data) if data else DATASET
        index = pd.read_csv(self.data / "index.csv", keep_default_na=False)
        if categories:
            missing = set(categories) - set(index.category)
            if missing:
                raise ValueError(f"없는 카테고리: {sorted(missing)}")
            index = index[index.category.isin(categories)]
        self.index = index.reset_index(drop=True)

    def __repr__(self):
        return (f"Dataset({self.data.name}/: {len(self.categories)}개 카테고리, "
                f"{len(self.index):,} 장)")

    @property
    def categories(self):
        return sorted(self.index.category.unique())

    @property
    def domains(self):
        """도메인별 카테고리. {"agriculture": ["agri_01", ...], ...}"""
        g = self.index.groupby("group").category.unique()
        return {k: sorted(v) for k, v in g.items()}

    def category(self, name):
        """카테고리 하나를 train / val / test 로 돌려줌."""
        df = self.index[self.index.category == name]
        if df.empty:
            raise ValueError(f"없는 카테고리: {name}")
        part = {k: df[df.split == k].reset_index(drop=True) for k in ("train", "val", "test")}
        return Category(name, part["train"], part["val"], part["test"], self.data)

    def all(self):
        """카테고리 구분 없이 전체를 train / val / test 로 돌려줌. 쓰는 법은 category() 와 같음."""
        part = {k: self.index[self.index.split == k].reset_index(drop=True)
                for k in ("train", "val", "test")}
        return Category("all", part["train"], part["val"], part["test"], self.data)

    def group_of(self, name):
        """카테고리가 속한 도메인."""
        return self.index.loc[self.index.category == name, "group"].iloc[0]


def load_image(path, data=DATASET):
    """이미지 한 장. (256, 256, 3) uint8."""
    from PIL import Image

    with Image.open(Path(data) / path) as im:
        return np.array(im.convert("RGB"))


def load_mask(path, data=DATASET):
    """정답 마스크 한 장. (256, 256) 0/1. 정상은 전부 0."""
    from PIL import Image

    if not isinstance(path, str) or not path:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    with Image.open(Path(data) / path) as im:
        return (np.array(im.convert("L")) > 127).astype(np.uint8)
