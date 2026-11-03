"""해석을 만들고 검증하는 도구.

    show(image, mask, heatmap)            원본 / 정답 / 맵을 나란히
    occlusion(score_fn, image, fill)      가려 가며 점수 변화를 재는 사후 해석
    agreement(heatmap, mask)              맵이 결함 위치를 짚었는지 (iou 등)
    nearest_normal(feat_fn, image, train) 가장 비슷한 정상 이미지

어떤 해석을 쓰든 정답 마스크와 대조해야 함. 그럴듯해 보이는 설명과
실제로 맞는 설명은 다름.
"""

import numpy as np

from .metrics import localization_agreement as agreement


def show(image, mask=None, heatmap=None, title=None, ax=None):
    """원본, 정답 마스크, 이상 맵을 한 줄에 그림."""
    import matplotlib.pyplot as plt

    panels = [("원본", image, None)]
    if mask is not None:
        panels.append((f"정답 ({mask.mean() * 100:.1f}%)", mask, "gray"))
    if heatmap is not None:
        name = "이상 맵"
        if mask is not None and mask.any():
            name += f" (IoU {agreement(heatmap, mask)['iou']:.2f})"
        panels.append((name, heatmap, "inferno"))

    if ax is None:
        _, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
    for a, (name, img, cmap) in zip(np.atleast_1d(ax), panels):
        if cmap == "inferno":
            a.imshow(image)
            a.imshow(img, cmap=cmap, alpha=0.55)
        else:
            a.imshow(img, cmap=cmap)
        a.set_title(name, fontsize=10)
        a.axis("off")
    if title:
        plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    return ax


def occlusion(score_fn, image, patch=32, stride=16, fill="blur"):
    """패치를 가려 가며 점수가 얼마나 떨어지는지 잼. (256, 256) 맵을 돌려줌.

    가렸을 때 점수가 크게 떨어지는 자리가 이상 판정을 만들던 곳임. 모델 내부를
    보지 않아 어떤 모델에나 붙음. 이미지 한 장에 수백 번 추론하니 소수에만 쓸 것.

    가린 자리를 무엇으로 채우느냐가 결과를 바꿈. 평균색으로 채우면 검은 배경 위의 회색 칸
    자체가 새 이상이 되어 점수를 올리기도 함. 그래서 기본은 흐린 사본으로 채움.
    보고서에는 어떤 채우기를 썼는지 적을 것.

    Args:
        score_fn: (n, 256, 256, 3) -> (n,) 이미지 점수 함수. 보통 model.score.
        fill: "blur" (흐린 사본, 기본) / "mean" (이미지 평균색) / "black".
    """
    from scipy.ndimage import gaussian_filter

    h, w = image.shape[:2]
    base = float(score_fn(image[None])[0])
    if fill == "blur":
        filler = gaussian_filter(image.astype(np.float32), sigma=(8, 8, 0)).astype(image.dtype)
    elif fill == "mean":
        filler = np.broadcast_to(image.mean(axis=(0, 1)).astype(image.dtype), image.shape)
    elif fill == "black":
        filler = np.zeros_like(image)
    else:
        raise ValueError(f"fill 은 blur / mean / black 중 하나: {fill}")

    batch, coords = [], []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            occluded = image.copy()
            occluded[y:y + patch, x:x + patch] = filler[y:y + patch, x:x + patch]
            batch.append(occluded)
            coords.append((y, x))

    scores = np.asarray(score_fn(np.stack(batch)), dtype=float)
    total, count = np.zeros((h, w)), np.zeros((h, w))
    for (y, x), s in zip(coords, scores):
        total[y:y + patch, x:x + patch] += base - s
        count[y:y + patch, x:x + patch] += 1
    return total / np.maximum(count, 1)


def nearest_normal(feat_fn, image, train_images, k=1):
    """학습셋에서 가장 비슷한 정상 이미지의 순번 k 개.

    "이건 정상 중 이것과 가장 닮았는데 여기가 다르다"는 식으로 나란히 보여 줄 때 씀.

    Args:
        feat_fn: (n, 256, 256, 3) -> (n, ...) 특징 함수. 전역 비교라 뭉개도 됨.
    """
    q = np.asarray(feat_fn(image[None]), dtype=np.float32).reshape(1, -1)
    bank = np.asarray(feat_fn(train_images), dtype=np.float32).reshape(len(train_images), -1)
    d = ((bank - q) ** 2).sum(axis=1)
    return np.argsort(d)[:k]
