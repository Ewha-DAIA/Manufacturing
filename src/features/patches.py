"""격자 단위 특징과 맵 되돌리기. numpy 만 씀.

    pool(x, k)              (n, H, W, c) -> (n, H/k, W/k, c) 평균 풀링
    patch_stats(images)     (n, 256, 256, 3) -> (n, 32, 32, 8) 칸마다 색·질감
    upsample(grid)          (n, h, w) -> (n, 256, 256) 원본 해상도로
    blur(maps, sigma)       맵을 부드럽게

이상 맵은 격자 칸마다 점수를 매긴 뒤 원본 해상도로 되돌려 만듦.
공간 축을 유지하는 것이 핵심임. 전역 평균으로 뭉개면 위치 정보가 사라짐.
"""

import numpy as np


def pool(x, k):
    """(n, H, W, c) -> (n, H/k, W/k, c). k 칸씩 묶어 평균."""
    n, h, w, c = x.shape
    return x.reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4))


def gradient(gray):
    """(n, H, W) 밝기의 그래디언트 크기. 균열·스크래치는 대개 경계로 드러남."""
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    return np.sqrt(gx ** 2 + gy ** 2)


def patch_stats(images, cell=8):
    """칸마다 색과 질감을 잼. (n, 256, 256, 3) uint8 -> (n, 256/cell, 256/cell, 8).

        0~2   RGB 평균          얼룩, 변색, 곰팡이
        3~5   RGB 표준편차      칸 안의 얼룩덜룩함
        6     그래디언트 평균    균열, 스크래치의 경계
        7     그래디언트 표준편차

    cell 이 작을수록 작은 결함을 잡지만 잡음도 늘어남.
    """
    x = np.asarray(images, dtype=np.float32) / 255.0
    grad = gradient(x.mean(axis=3))[..., None]

    mean = pool(x, cell)
    std = np.sqrt(np.maximum(pool(x ** 2, cell) - mean ** 2, 0))
    g_mean = pool(grad, cell)
    g_std = np.sqrt(np.maximum(pool(grad ** 2, cell) - g_mean ** 2, 0))
    return np.concatenate([mean, std, g_mean, g_std], axis=3)


def upsample(grid, size=256, mode="nearest"):
    """(n, h, w) 격자 점수 -> (n, size, size) 맵.

    nearest 는 칸을 그대로 펼침. 경계가 또렷해 작은 결함 위치가 잘 맞음.
    bilinear 는 부드럽지만 경계가 흐려짐. 둘 다 써 보고 AUPRO 로 비교할 것.
    """
    grid = np.asarray(grid, dtype=np.float32)
    n, h, w = grid.shape
    if size % h or size % w:
        raise ValueError(f"격자 {h}x{w} 를 {size} 로 정확히 되돌릴 수 없음")
    if mode == "nearest":
        return np.repeat(np.repeat(grid, size // h, axis=1), size // w, axis=2)
    if mode == "bilinear":
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(grid)[:, None]
        return F.interpolate(t, size=(size, size), mode="bilinear",
                             align_corners=False)[:, 0].numpy()
    raise ValueError(f"mode 는 nearest / bilinear 중 하나: {mode}")


def blur(maps, sigma=4):
    """(n, H, W) 맵에 가우시안 블러. 칸 하나짜리 잡음을 줄임."""
    from scipy.ndimage import gaussian_filter

    maps = np.asarray(maps, dtype=np.float32)
    return np.stack([gaussian_filter(m, sigma) for m in maps])
