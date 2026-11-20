"""合成缺陷考题：往 T02 原图的拷贝里造缺陷，真值写 标注/评测/合成缺陷.json。

  python -m railside.eval.synth                   # 默认读 T02 原图和 T02 的 parts.json / compare.json
  railside run 数据/中间/评测/T02_synth/camImg --trip T02S
  python -m railside.eval.score                   # 算各类召回

拷贝只改像素不改几何，T02S 和 T02 的趟坐标相同，真值外框直接用 T02 趟坐标。

挑哪些零件：T02 上像素匹配找到（found_by=pixel、分数 ≥ 0.8）、T02 对比本身没告警的子件，
各缺陷之间横向至少隔 SPACING px，按车轮流挑，免得互相影响、也免得把 T02 原有的告警算成召回。
零件位置用的是系统自己在 T02 上的匹配结果，这一点不独立（结论里写明）；造缺陷的像素操作不依赖实现。

缺陷：
  missing   抹掉零件：掩膜外扩后用周围纹理修补（cv2.inpaint）
  rotate    绕零件质心转 5°/10°/20°，原位置先修补成背景
  shift     沿随机方向平移该类型位移阈值的 2 倍（mm 按 E1 的 sx、sy 换 px），原位置先修补
  occlude   一块软边泥污遮住零件外框的 50–80%
  foreign   转向架外框内放一个异物（不规则块或一段线），明暗和局部背景反着来
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import io
from ..template import load_types
from .frameset import FrameCopy

SEED = 20261118
SPACING = 1500
SX, SY = 0.2335, 0.2255   # mm/px，实验/E1（T02 横向；纵向取 T01 的 sy）
PLAN = [  # (类型, 允许的零件类型, 每处参数)
    ('foreign', ('BOGIE',), [{'shape': 'blob'}] * 4 + [{'shape': 'wire'}] * 2),   # 先放：要整台转向架
    ('missing', ('BOLT', 'AXLE_END', 'LINK'), [{}] * 10),
    ('rotate', ('BOLT', 'AXLE_END', 'DAMPER', 'LINK'), [{'deg': d} for d in (5, 10, 20) for _ in range(4)]),
    ('shift', ('BOLT', 'AXLE_END', 'DAMPER', 'LINK'), [{'k': 2.0}] * 8),
    ('occlude', ('BOLT', 'AXLE_END', 'DAMPER', 'LINK', 'SPRING1'), [{}] * 6),
]
CHILD_MIN_SCORE = 0.8


def default_paths():
    return {
        'src': io.ROOT / '数据/原始/两次数据/ZXB_LC05D（2）/camImg',
        'parts': io.MID / 'T02' / 'parts.json',
        'compare': io.MID / 'T02' / 'compare.json',
        'out': io.MID / '评测' / 'T02_synth' / 'camImg',
        'truth': io.ROOT / '标注' / '评测' / '合成缺陷.json',
    }


# ------------------------------------------------------------------ 选位置

def candidates(parts, compare, types):
    alarmed = {i['code'] for i in (compare or {}).get('items', []) if i.get('status') != 'ok'}
    out = []
    for p in parts['instances']:
        if p['code'] in alarmed or p.get('found_by') != 'pixel':
            continue
        if p['type'] == 'BOGIE':
            out.append(p)
        elif p.get('parent') and (p.get('match_score') or 0) >= CHILD_MIN_SCORE and p['type'] in types:
            out.append(p)
    return out


class Picker:
    """按车轮流挑，类型合格、离已造的缺陷（按实际外框中心）够远。"""

    def __init__(self, cands, rng, spacing=None):
        self.rng, self.spacing = rng, SPACING if spacing is None else spacing
        self.by_car = {}
        for p in cands:
            self.by_car.setdefault(p.get('car'), []).append(p)
        self.cars = sorted(self.by_car, key=lambda c: (c is None, c))
        self.k = 0
        self.used, self.xs = set(), []

    def next(self, allow):
        for _ in range(len(self.cars)):
            car = self.cars[self.k % len(self.cars)]
            self.k += 1
            pool = [p for p in self.by_car[car] if p['type'] in allow and p['code'] not in self.used
                    and all(abs(_cx(p) - x) >= self.spacing for x in self.xs)]
            if pool:
                p = pool[self.rng.integers(len(pool))]
                self.used.add(p['code'])
                return p
        raise SystemExit(f'挑不够零件（{allow}），放宽 SPACING 或检查 parts.json')

    def placed(self, bbox):
        self.xs.append((bbox[0] + bbox[2]) / 2)


def _cx(p):
    b = p['bbox_xy']
    return (b[0] + b[2]) / 2


# ------------------------------------------------------------------ 造缺陷

def _region(p, margin):
    x0, y0, x1, y1 = [int(round(v)) for v in p['bbox_xy']]
    return x0 - margin, x1 + margin, max(0, y0 - margin), min(4096, y1 + margin)


def _mask(p, shape, origin):
    m = np.zeros(shape, np.uint8)
    poly = np.asarray(p['poly_xy'], float) - origin
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    return m


def _background(im, mask):
    m = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return cv2.inpaint(im, m, 7, cv2.INPAINT_TELEA)


def _composite(bg, obj, alpha):
    a = alpha.astype(np.float32) / 255.0
    return np.clip(bg * (1 - a) + obj * a, 0, 255).astype(np.uint8)


def _soft(mask, s=1.5):
    return cv2.GaussianBlur(mask, (0, 0), s)


def make_missing(fc, p, prm, rng):
    size = max(p['bbox_xy'][2] - p['bbox_xy'][0], p['bbox_xy'][3] - p['bbox_xy'][1])
    x0, x1, y0, y1 = _region(p, int(max(30, 0.6 * size)))
    im, o = fc.get(x0, x1, y0, y1)
    m = _mask(p, im.shape, o)
    fc.put(_background(im, m), o)
    return [round(v) for v in p['bbox_xy']], {}


def _move(fc, p, M_fn, margin):
    x0, x1, y0, y1 = _region(p, margin)
    im, o = fc.get(x0, x1, y0, y1)
    m = _mask(p, im.shape, o)
    bg = _background(im, m)
    M = M_fn(o)
    h, w = im.shape
    obj = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = cv2.warpAffine(_soft(m), M, (w, h), flags=cv2.INTER_LINEAR)
    fc.put(_composite(bg.astype(np.float32), obj.astype(np.float32), a), o)
    ys, xs = np.nonzero((a > 20) | (m > 0))
    return [int(xs.min() + o[0]), int(ys.min() + o[1]), int(xs.max() + o[0] + 1), int(ys.max() + o[1] + 1)]


def make_rotate(fc, p, prm, rng):
    deg = prm['deg'] * (1 if rng.random() < 0.5 else -1)
    size = max(p['bbox_xy'][2] - p['bbox_xy'][0], p['bbox_xy'][3] - p['bbox_xy'][1])
    poly = np.asarray(p['poly_xy'], float)
    c = poly.mean(0)
    bbox = _move(fc, p, lambda o: cv2.getRotationMatrix2D((float(c[0] - o[0]), float(c[1] - o[1])), deg, 1.0),
                 int(max(30, 0.6 * size)))
    return bbox, {'deg': deg, 'center_xy': [round(float(c[0]), 1), round(float(c[1]), 1)]}


def make_shift(fc, p, prm, rng, types):
    lim = float(types.get(p['type'], {}).get('shift_mm_max') or 5)
    mm = prm['k'] * lim
    ang = rng.uniform(0, 2 * np.pi)
    dx_mm, dy_mm = mm * np.cos(ang), mm * np.sin(ang)
    dx, dy = dx_mm / SX, dy_mm / SY
    size = max(p['bbox_xy'][2] - p['bbox_xy'][0], p['bbox_xy'][3] - p['bbox_xy'][1])
    bbox = _move(fc, p, lambda o: np.float32([[1, 0, dx], [0, 1, dy]]), int(max(30, 0.6 * size) + np.hypot(dx, dy)))
    return bbox, {'shift_mm': round(mm, 2), 'shift_vec_mm': [round(dx_mm, 2), round(dy_mm, 2)],
                  'shift_px': [round(dx, 1), round(dy, 1)], 'threshold_mm': lim}


def make_occlude(fc, p, prm, rng):
    bx0, by0, bx1, by1 = p['bbox_xy']
    w, h = bx1 - bx0, by1 - by0
    frac = rng.uniform(0.5, 0.8)
    x0, x1, y0, y1 = _region(p, int(max(20, 0.3 * max(w, h))))
    im, o = fc.get(x0, x1, y0, y1)
    cx = (bx0 + bx1) / 2 - o[0] + rng.uniform(-0.15, 0.15) * w
    cy = (by0 + by1) / 2 - o[1] + rng.uniform(-0.15, 0.15) * h
    ax, ay = np.sqrt(frac) * w / 2 * 1.15, np.sqrt(frac) * h / 2 * 1.15
    m = np.zeros(im.shape, np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (max(4, int(ax)), max(4, int(ay))), rng.uniform(0, 180), 0, 360, 255, -1)
    level = float(np.median(im)) * rng.uniform(0.35, 0.55)
    tex = cv2.GaussianBlur(rng.normal(0, 10, im.shape).astype(np.float32), (0, 0), 3) + level
    a = _soft(m, 2.5)
    fc.put(_composite(im.astype(np.float32), tex, a), o)
    ys, xs = np.nonzero(a > 20)
    return ([int(xs.min() + o[0]), int(ys.min() + o[1]), int(xs.max() + o[0] + 1), int(ys.max() + o[1] + 1)],
            {'cover_frac': round(frac, 2), 'level': round(level, 1)})


def make_foreign(fc, p, prm, rng):
    bx0, by0, bx1, by1 = [int(v) for v in p['bbox_xy']]
    r = int(rng.uniform(45, 110))
    cx = int(rng.uniform(bx0 + 0.2 * (bx1 - bx0), bx1 - 0.2 * (bx1 - bx0)))
    cy = int(rng.uniform(by0 + 0.25 * (by1 - by0), by1 - 0.25 * (by1 - by0)))
    x0, x1, y0, y1 = cx - 3 * r, cx + 3 * r, cy - 3 * r, cy + 3 * r
    im, o = fc.get(x0, x1, y0, y1)
    c = np.array([cx - o[0], cy - o[1]], float)
    m = np.zeros(im.shape, np.uint8)
    if prm['shape'] == 'blob':
        n = int(rng.integers(7, 12))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = r * rng.uniform(0.55, 1.0, n)
        pts = np.stack([c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang) * rng.uniform(0.6, 1.0)], 1)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    else:
        n = 5
        t = np.linspace(-1, 1, n)
        th = rng.uniform(0, np.pi)
        pts = np.stack([c[0] + 2.5 * r * t * np.cos(th) + rng.normal(0, r * 0.25, n),
                        c[1] + 2.5 * r * t * np.sin(th) + rng.normal(0, r * 0.25, n)], 1)
        cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 255, int(rng.integers(8, 16)))
    local = float(np.median(im[m > 0])) if (m > 0).any() else float(np.median(im))
    level = rng.uniform(25, 55) if local > 100 else rng.uniform(185, 225)
    g = np.linspace(-15, 15, im.shape[1], dtype=np.float32)[None, :]
    tex = level + g + cv2.GaussianBlur(rng.normal(0, 9, im.shape).astype(np.float32), (0, 0), 1.2)
    a = _soft(m, 1.5)
    fc.put(_composite(im.astype(np.float32), tex, a), o)
    ys, xs = np.nonzero(a > 20)
    return ([int(xs.min() + o[0]), int(ys.min() + o[1]), int(xs.max() + o[0] + 1), int(ys.max() + o[1] + 1)],
            {'shape': prm['shape'], 'level': round(float(level), 1), 'area_px': int((m > 0).sum())})


# ------------------------------------------------------------------ 主流程

def build(src, parts_path, compare_path, out, truth_path, seed=SEED, plan=PLAN):
    rng = np.random.default_rng(seed)
    types = load_types()
    parts = io.read_json(parts_path)
    compare = io.read_json(compare_path) if Path(compare_path).is_file() else None
    allow = {t for _, a, _ in plan for t in a}
    picker = Picker(candidates(parts, compare, allow), rng)
    fc = FrameCopy(src, out)
    defects = []
    chosen = [(kind, allow_t, prm) for kind, allow_t, params in plan for prm in params]
    for i, (kind, allow_t, prm) in enumerate(chosen, 1):
        p = picker.next(allow_t)
        prm = dict(prm)
        if kind == 'shift':
            bbox, extra = make_shift(fc, p, prm, rng, types)
        else:
            bbox, extra = {'missing': make_missing, 'rotate': make_rotate, 'occlude': make_occlude,
                           'foreign': make_foreign}[kind](fc, p, prm, rng)
        picker.placed(bbox)
        defects.append({'id': f'D{i:02d}', 'kind': kind, 'code': p['code'], 'part_type': p['type'], 'car': p.get('car'),
                        'part_bbox_xy': [round(v) for v in p['bbox_xy']], 'bbox_xy': bbox, 'params': extra})
    fc.write()
    truth = {'trip': 'T02S', 'base_trip': 'T02', 'seed': seed, 'src': io.rel(src), 'out': io.rel(out),
             'parts_json': io.rel(parts_path), 'parts_sha256': _sha(parts_path),
             'frames_modified': sorted(int(f) for f in fc.dirty), 'jpeg_quality': fc.quality,
             'sx_mm_per_px': SX, 'sy_mm_per_px': SY, 'spacing_px': SPACING, 'defects': defects}
    io.write_json(truth_path, truth)
    kinds = {}
    for d in defects:
        kinds[d['kind']] = kinds.get(d['kind'], 0) + 1
    return {'step': 'eval synth', 'defects': len(defects), 'by_kind': kinds,
            'frames_modified': len(fc.dirty), 'out': io.rel(out), 'truth': io.rel(truth_path)}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8')
    d = default_paths()
    ap = argparse.ArgumentParser(description='合成缺陷考题')
    ap.add_argument('--src', default=d['src'])
    ap.add_argument('--parts', default=d['parts'])
    ap.add_argument('--compare', default=d['compare'])
    ap.add_argument('--out', default=d['out'])
    ap.add_argument('--truth', default=d['truth'])
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.src), Path(a.parts), Path(a.compare), Path(a.out), Path(a.truth), a.seed),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
