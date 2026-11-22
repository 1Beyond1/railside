"""坏帧考题：从 T01 原图拷贝一份，造缺帧、整帧压暗、过曝、模糊、遮挡，真值写 标注/评测/坏帧.json。

  python -m railside.eval.badframes
  railside ingest 数据/中间/评测/T01_bad/camImg --trip T01B
  python -m railside.eval.score

缺帧只从中段抽（末帧缺失按帧号连续性查不出来，见 Issue #18，另记局限）。
模糊、遮挡挑车体内的帧（空场本来就是横条纹，模糊和遮挡无意义）；压暗、过曝车内车外都挑。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import io
from .frameset import FrameCopy, default_raw_src

SEED = 20261118
TRAIN = (40, 275)        # 车体内的帧（T01 实测 29–285，两头各让 10 帧）
PER_KIND = 3


def default_paths(trip='T01'):
    return {
        'src': default_raw_src(trip),
        'out': io.MID / '评测' / f'{trip}_bad' / 'camImg',
        'truth': io.ROOT / '标注' / '评测' / '坏帧.json',
    }


def dark(im, rng):
    k = rng.uniform(0.08, 0.15)
    return np.clip(im.astype(np.float32) * k, 0, 255).astype(np.uint8), {'gain': round(k, 3)}


def over(im, rng):
    k = rng.uniform(2.6, 3.4)
    return np.clip(im.astype(np.float32) * k + 90, 0, 255).astype(np.uint8), {'gain': round(k, 2), 'offset': 90}


def blur(im, rng):
    s = rng.uniform(5, 8)
    return cv2.GaussianBlur(im, (0, 0), s), {'sigma': round(s, 2)}


def occlusion(im, rng):
    """一大块低纹理遮挡（贴纸、污物、镜头前挡板）盖住帧高的 50–70%。"""
    h, w = im.shape
    frac = rng.uniform(0.5, 0.7)
    y0 = int(rng.uniform(0, h * (1 - frac)))
    y1 = y0 + int(h * frac)
    level = rng.uniform(40, 200)
    out = im.copy()
    patch = level + cv2.GaussianBlur(rng.normal(0, 4, (y1 - y0, w)).astype(np.float32), (0, 0), 6)
    out[y0:y1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out, {'rows': [y0, y1], 'level': round(float(level), 1)}


KINDS = {'dark': dark, 'over': over, 'blur': blur, 'occlusion': occlusion}


def build(src, out, truth_path, seed=SEED, per_kind=PER_KIND, train=TRAIN):
    rng = np.random.default_rng(seed)
    fc = FrameCopy(src, out)
    n = max(fc.frames)
    taken = set()

    def draw(lo, hi):
        while True:
            f = int(rng.integers(lo, hi + 1))
            if f in fc.frames and f not in taken and f - 1 not in taken and f + 1 not in taken:
                taken.add(f)
                return f

    bad = []
    drop = [draw(train[0], train[1]) for _ in range(per_kind)]
    bad += [{'f': f, 'kind': 'missing', 'params': {}} for f in drop]
    for kind, fn in KINDS.items():
        lo, hi = train if kind in ('blur', 'occlusion') else (2, n - 1)
        for _ in range(per_kind):
            f = draw(lo, hi)
            im, prm = fn(fc._frame(f), rng)
            fc.set_frame(f, im)
            bad.append({'f': f, 'kind': kind, 'params': prm})
    fc.write(drop=set(drop))
    bad.sort(key=lambda b: b['f'])
    io.write_json(truth_path, {'trip': 'T01B', 'base_trip': 'T01', 'seed': seed, 'src': io.rel(src), 'out': io.rel(out),
                               'frames': n, 'bad': bad})
    by = {}
    for b in bad:
        by[b['kind']] = by.get(b['kind'], 0) + 1
    return {'step': 'eval badframes', 'bad': len(bad), 'by_kind': by, 'out': io.rel(out), 'truth': io.rel(truth_path)}


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8')
    d = default_paths()
    ap = argparse.ArgumentParser(description='坏帧考题')
    ap.add_argument('--src', default=d['src'], help='原图目录（默认读 RAILSIDE_EVAL_DATA_ROOT / RAILSIDE_RAW_ROOT、trip.json 或 数据/原始/T01/camImg）')
    ap.add_argument('--out', default=d['out'])
    ap.add_argument('--truth', default=d['truth'])
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.src), Path(a.out), Path(a.truth), a.seed), ensure_ascii=False))


if __name__ == '__main__':
    main()
