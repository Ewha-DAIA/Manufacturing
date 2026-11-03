"""지표. 입력을 받아 숫자를 돌려주는 순수 함수만 둠.

채점 절차(임계값 규칙, 기준선, 판정)는 evaluate.py 에 있음.

주 지표 두 개는 임계값과 무관함. 운용 지표 하나는 팀이 정한 임계값으로 잼.

    Image AUROC   이미지 점수가 정상/불량을 얼마나 잘 가르는가
    Pixel AUPRO   이상 맵이 결함 위치를 얼마나 정확히 짚는가
    F1            실제 임계값으로 판정했을 때 불량 검출의 F1

pixel AUROC를 주 지표로 쓰지 않는 이유: 결함 픽셀이 전체의 1%도 안 돼서 거의 모든
모델이 0.97 이상을 받고 변별이 안 됨. AUPRO는 결함 영역마다 따로 정규화하므로
작은 결함을 놓치는 것을 제대로 벌함.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .data import SIZE_BINS


def auroc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=float)))


def pixel_auroc(masks, maps) -> float:
    """픽셀 전부를 한 줄로 펴서 계산. 참고용."""
    return auroc(np.asarray(masks).ravel() > 0, np.asarray(maps).ravel())


def aupro(masks, maps, fpr_limit: float = 0.3, n_steps: int = 100) -> float:
    """Per-Region Overlap 곡선의 면적 (FPR 0.3까지, 그 구간으로 정규화).

    임계값을 낮춰가며 결함 영역마다 덮인 비율을 재고, 영역 크기로 가중하지 않고
    평균함. 그래서 작은 결함도 큰 결함과 동등하게 반영됨.
    """
    masks = np.asarray(masks) > 0
    maps = np.asarray(maps, dtype=float)

    regions = []
    for m, s in zip(masks, maps):
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        regions.extend((s[lab == i], (lab == i).sum()) for i in range(1, n + 1))
    if not regions:
        return float("nan")

    normal = maps[~masks]
    pros, fprs = [], []
    for t in np.quantile(maps, np.linspace(1.0, 0.0, n_steps)):
        pros.append(np.mean([(s >= t).sum() / size for s, size in regions]))
        fprs.append((normal >= t).mean())
        if fprs[-1] > fpr_limit:
            break

    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    return float(np.trapezoid(pros[keep], fprs[keep]) / fpr_limit)


def f1(labels, pred) -> dict[str, float]:
    """정해진 임계값으로 낸 0/1 판정의 F1, 정밀도, 재현율."""
    labels, pred = np.asarray(labels, int), np.asarray(pred, int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1)}


def by_defect_size(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """결함 면적 구간별 AUROC. 각 구간의 불량을 정상 전체와 비교함."""
    normal = df[df["is_anomaly"] == 0]
    rows = []
    for lo, hi, name in SIZE_BINS:
        part = df[(df["is_anomaly"] == 1) &
                  (df["mask_area_pct"] > lo) & (df["mask_area_pct"] <= hi)]
        if part.empty:
            continue
        sub = pd.concat([normal, part])
        rows.append({"구간": f"{name} ({lo}~{hi}%)", "n": len(part),
                     "auroc": auroc(sub["is_anomaly"], sub[score_col])})
    return pd.DataFrame(rows)


def localization_agreement(heatmap, mask, top_pct: float = 0.05) -> dict[str, float]:
    """해석 맵이 실제 결함 위치를 짚었는지 정량화함.

    맵 상위 top_pct 픽셀을 '모델이 본 곳'으로 보고 정답 마스크와 비교함.
    이상 맵이든 CAM이든 occlusion이든 VLM이 말한 위치든 이 대조를 거쳐야 함.

    Returns:
        iou / hit_rate(상위 영역 중 결함 비율) / mask_coverage(결함 중 잡힌 비율)
    """
    mask = np.asarray(mask) > 0
    if not mask.any():
        return {"iou": float("nan"), "hit_rate": float("nan"), "mask_coverage": float("nan")}

    # 임계값 비교로 하면 동점에서 상위 영역이 이미지 전체로 번짐. 정확히 k개만 고름.
    k = max(1, int(heatmap.size * top_pct))
    top = np.zeros(heatmap.shape, dtype=bool)
    top.ravel()[np.argpartition(np.asarray(heatmap).ravel(), -k)[-k:]] = True

    inter = (top & mask).sum()
    return {"iou": float(inter / (top | mask).sum()),
            "hit_rate": float(inter / top.sum()),
            "mask_coverage": float(inter / mask.sum())}
