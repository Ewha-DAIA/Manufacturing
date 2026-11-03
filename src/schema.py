"""모델 출력의 형식. 이상 맵과 점수를 낼 때마다 여기를 통과함.

    check_maps(maps, n)            맵이 (n, 256, 256) 이고 NaN/inf 가 없는가
    check_scores(scores, maps)     점수 개수가 맞고 NaN/inf 가 없는가

어기면 그 자리에서 멈춤. 숫자가 통째로 무의미해지는 지점만 막고,
모델 구조에는 간섭하지 않음.
"""

import numpy as np

from .data import IMAGE_SIZE


def check_maps(maps, n) -> np.ndarray:
    """score_map 결과를 검사해서 float32 배열로 돌려줌.

    맵을 원본 해상도로 되돌리지 않거나 정렬이 어긋나면 픽셀 지표가 무의미해짐.
    """
    try:
        maps = np.asarray(maps, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise ValueError(f"score_map 은 숫자 배열을 돌려줘야 함: {e}")
    want = (n, IMAGE_SIZE, IMAGE_SIZE)
    if maps.shape != want:
        raise ValueError(f"score_map 형태가 {want} 가 아님: {maps.shape}. "
                         "특징 격자를 원본 해상도(256)로 되돌렸는지 확인할 것")
    bad = ~np.isfinite(maps)
    if bad.any():
        raise ValueError(f"score_map 에 NaN 또는 inf 가 {int(bad.sum())}개 있음")
    return maps


def check_scores(scores, maps) -> np.ndarray:
    """score 결과를 검사해서 float 배열로 돌려줌. 개수와 NaN/inf 만 봄."""
    s = np.asarray(scores, dtype=float).ravel()
    if len(s) != len(maps):
        raise ValueError(f"score 개수({len(s)})가 이미지 수({len(maps)})와 다름")
    if not np.isfinite(s).all():
        raise ValueError("score 에 NaN 또는 inf 가 있음")
    return s
