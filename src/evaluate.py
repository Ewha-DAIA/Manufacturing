"""예측과 채점.

    predict(model, cat, "val")      학습된 모델로 맵과 점수를 내고 판정까지
    run(make_model, cat)            train 으로 새로 학습한 뒤 predict. 실험용
    evaluate(categories, "val")     카테고리마다 load_model 로 불러 채점. 생략하면 전체
    check(categories)               출력 형식만 확인

임계값은 팀이 정함. load_model 이 돌려준 모델에 `threshold` 속성이 있으면 그 값을 쓰고,
없으면 val 정상 이미지 점수의 상위 1% 지점을 씀. 채점은 같은 evaluate 를 비공개 test 로 돌림.
"""

import time

import numpy as np
import pandas as pd

from . import metrics
from .data import SEED, THRESHOLD_QUANTILE, Dataset
from .schema import check_maps, check_scores



def threshold(normal_scores) -> float:
    """정상 이미지 점수의 상위 1% 지점. 모델에 threshold 가 없을 때 쓰는 기본값."""
    return float(np.quantile(np.asarray(normal_scores, dtype=float), THRESHOLD_QUANTILE))


def decide(scores, thr) -> np.ndarray:
    """이상 점수 -> 0/1 판정."""
    return (np.asarray(scores, float) >= thr).astype(int)


def infer(model, images, batch=64):
    """맵과 점수를 내고 형식을 검사함.

    Returns:
        maps (n, 256, 256) float32, scores (n,) float.
    """
    maps, scores = [], []
    for i in range(0, len(images), batch):
        x = images[i:i + batch]
        m = check_maps(model.score_map(x), len(x))
        s = model.score(x) if hasattr(model, "score") else m.reshape(len(m), -1).max(1)
        maps.append(m)
        scores.append(check_scores(s, m))
    if not maps:
        return np.zeros((0, 256, 256), np.float32), np.zeros(0)
    return np.concatenate(maps), np.concatenate(scores)


def predict(model, cat, part="val", limit=None, batch=64):
    """학습된 모델로 한 카테고리를 예측함.

    Args:
        model: score_map(images) 를 가진 객체.
        cat: Category.
        part: "val" 또는 "test" (test 는 채점 때만 있음).
        limit: 정상과 불량을 반반 섞어 앞의 몇 장만. 빠르게 볼 때.

    Returns:
        res: 이미지 목록에 score / pred 열을 붙인 표.
        maps: (n, 256, 256) 이상 맵.
        thr: 임계값. 모델의 threshold, 없으면 val 정상 이미지 점수의 상위 1%.
    """
    rows = getattr(cat, part)
    if rows.empty:
        raise ValueError(f"{cat.name}: {part} 데이터가 없음. test 는 채점 때만 있음")
    rows = rows if limit is None else _balanced(rows, limit)

    thr = model_threshold(model)
    if thr is None:
        normal = cat.val[cat.val.is_anomaly == 0]
        normal = normal if limit is None else normal.head(limit)
        _, s_normal = infer(model, cat.images(normal), batch)
        thr = threshold(s_normal)

    maps, scores = [], []
    for _, x in cat.batches(rows, batch):
        m, s = infer(model, x, batch)
        maps.append(m)
        scores.append(s)

    res = rows.copy()
    res["score"] = np.concatenate(scores)
    res["pred"] = decide(res.score, thr)
    return res, np.concatenate(maps), thr


def model_threshold(model):
    """모델이 정한 임계값. 없으면 None."""
    thr = getattr(model, "threshold", None)
    if thr is None:
        return None
    thr = float(thr)
    if not np.isfinite(thr):
        raise ValueError(f"model.threshold 가 유한한 실수가 아님: {thr}")
    return thr


def run(make_model, cat, limit=None, batch=64, verbose=True):
    """train 으로 새 모델을 학습하고 val 을 예측함. 실험할 때 씀.

    Args:
        make_model: 새 모델을 돌려주는 함수나 클래스.
        limit: train 과 val 에서 앞의 몇 장만.
    """
    t0 = time.time()
    model = make_model().fit(cat.images("train", limit))
    t_fit = time.time() - t0
    res, maps, thr = predict(model, cat, "val", limit, batch)
    if verbose:
        print(f"  {cat.name}: 학습 {t_fit:.0f}초, 전체 {time.time() - t0:.0f}초", flush=True)
    return res, maps, thr


def report(res, masks=None, maps=None) -> dict:
    """카테고리 하나의 지표 묶음.

    Args:
        res: score / pred / is_anomaly 열이 있는 표.
        masks, maps: 주면 픽셀 지표와 위치 대조를 함께 냄.
    """
    y, s = res.is_anomaly, res.score
    out = {"image_auroc": metrics.auroc(y, s),
           **metrics.f1(y, res.pred),
           "n_images": len(res)}

    if masks is not None and maps is not None:
        out["pixel_aupro"] = metrics.aupro(masks, maps)
        out["pixel_auroc"] = metrics.pixel_auroc(masks, maps)
        loc = localization(res, maps, masks)
        out["mean_iou"] = float(loc.iou.mean()) if len(loc) else np.nan
        # 불량으로 걸러냈는데 이상 맵은 결함과 거의 안 겹친 이미지
        out["wrong_place"] = int(((loc.pred == 1) & (loc.iou < 0.05)).sum())
    return out


def localization(res, maps, masks) -> pd.DataFrame:
    """불량 이미지마다 이상 맵이 결함 위치를 짚었는지.

    i 는 res / maps / masks 에서의 순번임. path / pred / iou / hit_rate / mask_coverage.
    """
    rows = []
    for i in np.flatnonzero(res.is_anomaly.to_numpy() == 1):
        a = metrics.localization_agreement(maps[i], masks[i])
        rows.append({"i": int(i), "path": res.path.iloc[i], "pred": int(res.pred.iloc[i]), **a})
    return pd.DataFrame(rows, columns=["i", "path", "pred", "iou", "hit_rate", "mask_coverage"])


def evaluate(categories=None, split="val", data=None, limit=None, out=None, verbose=True):
    """카테고리마다 load_model 로 학습된 모델을 불러 채점함.

    Args:
        categories: 생략하면 전체 카테고리. 일부만 빠르게 볼 때 넘김.
        split: "val" (학생) 또는 "test" (채점).
        data: 데이터 폴더. 생략하면 dataset/.
        limit: 카테고리마다 앞의 몇 장만. 빠르게 볼 때.
        out: 이미지 점수를 저장할 csv 경로.

    Returns:
        카테고리별 지표 표.
    """
    from .model import load_model

    categories = categories or Dataset(data).categories
    ds = Dataset(data, categories)
    if verbose:
        print(ds)
        print(f"카테고리 {len(categories)}개 · {split}"
              f"{f' · 카테고리당 {limit}장만' if limit else ''}\n")

    rows, sizes, all_res = [], [], []
    for name in categories:
        t0 = time.time()
        cat = ds.category(name)
        res, maps, thr = predict(load_model(name), cat, split, limit)
        masks = cat.masks(res)
        rows.append({"category": name, "threshold": thr, **report(res, masks, maps)})
        if verbose:
            print(f"  {name}: {time.time() - t0:.0f}초", flush=True)

        sz = metrics.by_defect_size(res)
        sz.insert(0, "category", name)
        sizes.append(sz)
        all_res.append(res.assign(threshold=thr))
        del maps, masks

    table = pd.DataFrame(rows).set_index("category")
    if verbose:
        show(table, pd.concat(sizes, ignore_index=True))
    if out:
        cols = ["category", "path", "label", "is_anomaly", "mask_area_pct",
                "score", "pred", "threshold"]
        pd.concat(all_res)[cols].to_csv(out, index=False)
        print(f"\n저장 {out}")
    return table


def show(table, sizes=None):
    """카테고리별 결과표를 출력함."""
    def fmt(v):
        return "  -  " if v != v else f"{v:.3f}" if isinstance(v, float) else str(v)

    if sizes is not None and len(sizes):
        print("\n결함 크기별 image AUROC")
        p = sizes.pivot(index="category", columns="구간", values="auroc")
        order = [c for c in sizes["구간"].drop_duplicates() if c in p]
        print(p[sorted(order, key=lambda c: float(c.split("(")[1].split("~")[0]))]
              .map(fmt).to_string())

    scores = table[["image_auroc", "f1", "pixel_aupro"]].rename(
        columns={"image_auroc": "AUROC", "f1": "F1", "pixel_aupro": "AUPRO"})
    shown = scores.assign(산술평균=scores.mean(axis=1))
    print("\n" + shown.map(fmt).to_string())

    means = scores.mean()
    print("\n전체 평균")
    for name, value in means.items():
        print(f"  {name:6} {fmt(float(value))}")
    print(f"\n최종 점수 (AUROC + F1 + AUPRO) / 3 = {means.mean():.3f}")


def baselines(cat, limit=None, batch=64) -> pd.DataFrame:
    """넘어야 할 바닥선 2종. 학습이 전혀 없는 값임.

    평균영상 거리는 "정상 이미지들의 평균에서 얼마나 먼가"만 봄. 위치도 질감도
    보지 않음. 여기에 못 미치는 모델은 아무 일도 안 한 것임.
    """
    total, n = None, 0
    for _, x in cat.batches("train", batch, limit):
        b = x.astype(np.float64).sum(0)
        total = b if total is None else total + b
        n += len(x)
    mean_image = total / n

    rows = cat.val if limit is None else _balanced(cat.val, limit)
    scores = np.concatenate([
        np.sqrt(((x.astype(np.float64) - mean_image) ** 2).sum(axis=(1, 2, 3)))
        for _, x in cat.batches(rows, batch)])

    y = rows.is_anomaly.to_numpy()
    rng = np.random.default_rng(SEED)
    return pd.DataFrame([
        {"기준선": "무작위 점수", "image_auroc": metrics.auroc(y, rng.normal(size=len(y)))},
        {"기준선": "평균영상 거리", "image_auroc": metrics.auroc(y, scores)},
    ])


def check(categories=None, data=None, n=16):
    """제출 형식을 확인함. 점수는 보지 않음.

    - load_model 과 추론이 되는지
    - 맵과 점수의 형식, threshold 가 있으면 유한한 실수인지
    """
    from .model import load_model

    ds = Dataset(data)
    categories = categories or ds.categories
    missing = sorted(set(categories) - set(ds.categories))
    if missing:
        print(f"  없는 카테고리: {missing}")
        return 1
    print(f"카테고리 {len(categories)}개")

    print(f"카테고리당 {n}장으로 검사\n")

    for name in categories:
        cat = ds.category(name)
        try:
            model = load_model(name)
            thr = model_threshold(model)
            maps, scores = infer(model, cat.images(_balanced(cat.val, n)))
        except (ValueError, TypeError, FileNotFoundError, OSError) as e:
            print(f"  {name:20} 실패\n{e}")
            return 1
        except NotImplementedError as e:
            print(f"  {name:20} 아직 구현 전: {e}")
            return 1
        print(f"  {name:20} 맵 {maps.shape}  점수 {scores.min():.3g} ~ {scores.max():.3g}"
              f"  임계값 {'기본' if thr is None else f'{thr:.3g}'}")

    print("\n형식 확인 완료")
    return 0


def _balanced(rows, n):
    """정상과 불량을 반반씩 n 장. 빠르게 볼 때 AUROC 가 계산되도록."""
    bad = rows[rows.is_anomaly == 1]
    good = rows[rows.is_anomaly == 0]
    k = n // 2
    return pd.concat([good.head(n - min(k, len(bad))), bad.head(k)]).reset_index(drop=True)
