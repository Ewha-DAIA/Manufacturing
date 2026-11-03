#!/usr/bin/env python
"""학습, 검사, 채점.

    python main.py train                       전체 카테고리를 학습해 artifacts/ 에 저장
    python main.py check                       제출 형식 검사
    python main.py eval                        val 로 점수
    python main.py eval --limit 100            카테고리당 100장만, 빠르게
    python main.py baselines                   넘어야 할 바닥선

기본은 전체 카테고리. --categories 로 일부만 돌릴 수 있음 (실험용. 제출은 전체).
채점자는 같은 eval 을 비공개 test 로 돌림 (--split test).

파이썬에서 직접 쓸 때는 이렇게 함.

    from src import Dataset, load_model, predict

    cat = Dataset().category("agri_02")
    res, maps, thr = predict(load_model("agri_02"), cat, "val")
"""

import argparse
import time

from src import Dataset, baselines, check, evaluate
from src.model import train


def main():
    ap = argparse.ArgumentParser(description="학습, 검사, 채점")
    ap.add_argument("command", choices=["train", "check", "eval", "baselines"])
    ap.add_argument("--categories", nargs="+", help="대상 카테고리. 생략하면 전체")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="채점할 구간. 학생은 val, 채점자는 test")
    ap.add_argument("--data", help="데이터 폴더. 생략하면 dataset/")
    ap.add_argument("--limit", type=int, help="카테고리당 앞의 몇 장만 씀")
    ap.add_argument("--out", help="이미지 점수를 저장할 csv 경로")
    ap.add_argument("--table", help="카테고리별 지표표를 저장할 csv 경로")
    args = ap.parse_args()

    cats = args.categories or Dataset(args.data).categories

    if args.command == "train":
        ds = Dataset(args.data, cats)
        for name in cats:
            t0 = time.time()
            train(name, ds.category(name).images("train", args.limit))
            print(f"  {name:20} 학습·저장 {time.time() - t0:.0f}초", flush=True)
    elif args.command == "check":
        return check(cats, args.data)
    elif args.command == "eval":
        table = evaluate(cats, args.split, args.data, args.limit, args.out)
        if args.table:
            table.to_csv(args.table)
    else:
        ds = Dataset(args.data, cats)
        for name in cats:
            print(name)
            print(baselines(ds.category(name), args.limit).to_string(index=False), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
