"""S1 收录与质检：登记一趟原图，查缺帧、过暗过曝、模糊、大片均匀遮挡。

阈值按 T01、T02 实测定（2026-11-17）：
- 车内帧灰度均值 39–157，p99 ≥ 109，饱和像素占比 ≤ 0.28，均匀块占比 ≤ 0.36。
- 模糊指标 R = 强边缘处 |拉普拉斯| / |梯度|（半分辨率）：真实帧 ≥ 0.123，σ=1 高斯模糊后 ≤ 0.099。
遮挡和轻度模糊单帧判断不可靠（真实图里本来就有大片黑背景），对齐后在 align 里和模板逐帧比（质检第二段）。
"""
import os
import stat
from pathlib import Path

import cv2
import numpy as np

from . import io
from .frames import Trip

TH = {
    'dark_mean': 20, 'dark_p99': 60,
    'over_mean': 200, 'over_sat': 0.5,
    'blur_r': 0.11,
    'flat': 0.6,
}


def blur_r(im):
    im = im.astype(np.float32)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    g = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(cv2.Laplacian(im, cv2.CV_32F))
    m = g > np.percentile(g, 90)
    if not m.any():
        return 0.0
    return float(lap[m].mean() / (g[m].mean() + 1e-6))


def frame_stats(im):
    s = im[::4, ::4]
    h, w = s.shape
    blk = s[: h - h % 16, : w - w % 20].reshape(h // 16, 16, w // 20, 20).std(axis=(1, 3))
    return {
        'mean': round(float(s.mean()), 2),
        'p1': float(np.percentile(s, 1)),
        'p99': float(np.percentile(s, 99)),
        'sat': round(float((s >= 250).mean()), 4),
        'flat': round(float((blk < 2).mean()), 4),
        'act': round(float(np.abs(np.diff(s.astype(np.int16), axis=1)).mean()), 3),
        'blur_r': round(blur_r(im[::2, ::2]), 4),
    }


def train_frames(acts):
    """静止背景每列几乎相同，列间变化量小；按它分出有车的帧。返回布尔数组。"""
    acts = np.asarray(acts, float)
    base = np.median(np.sort(acts)[: max(5, len(acts) // 10)])
    on = acts > 3 * base + 1
    if not on.any():
        return on
    idx = np.where(on)[0]
    out = np.zeros_like(on)
    out[idx.min(): idx.max() + 1] = True  # 车内偶有平滑帧（车头罩），按首尾之间都算车内
    return out


def is_readonly(path):
    return not (os.stat(path).st_mode & stat.S_IWRITE)


def add_parser(sub, step):
    p = sub.add_parser(step, help='S1 收录质检：登记一趟原图并质检')
    p.add_argument('src', help='原图目录（含 1.jpg..N.jpg，如 .../camImg）')
    p.add_argument('--trip', required=True, help='趟次号，如 T01')
    p.add_argument('--camera', default=None, help='相机名，默认取目录名')
    p.set_defaults(func=run)


def run(args):
    src = Path(args.src).resolve()
    t = Trip(src)
    out = io.trip_dir(args.trip)
    missing = [f for f in range(1, t.n + 1) if f not in t.frames]
    per = []
    for f in range(1, t.n + 1):
        if f in missing:
            continue
        try:
            st = frame_stats(t.frame(f))
        except (IOError, ValueError) as e:
            per.append({'f': f, 'flags': ['unreadable'], 'error': str(e)})
            continue
        st['f'] = f
        st['readonly'] = is_readonly(t.frames[f])
        per.append(st)
    ok = [p for p in per if 'act' in p]
    inside = train_frames([p['act'] for p in ok])
    for p, ins in zip(ok, inside):
        p['in_train'] = bool(ins)
        fl = []
        if p['mean'] < TH['dark_mean'] or p['p99'] < TH['dark_p99']:
            fl.append('dark')
        if p['mean'] > TH['over_mean'] or p['sat'] > TH['over_sat']:
            fl.append('over')
        if ins and p['blur_r'] < TH['blur_r']:
            fl.append('blur')
        if ins and p['flat'] > TH['flat']:
            fl.append('occlusion')
        p['flags'] = fl
    train_f = [p['f'] for p in ok if p.get('in_train')]
    f0, f1 = (min(train_f), max(train_f)) if train_f else (None, None)
    miss_in = [f for f in missing if f0 and f0 <= f <= f1]
    bad = [p['f'] for p in per if p.get('flags')]
    bad_in = [p['f'] for p in per if p.get('flags') and (p.get('in_train') or 'unreadable' in p['flags'])]
    if miss_in or len(bad_in) >= 3 or not train_f:
        verdict = 'fail'
    elif missing or bad:
        verdict = 'warn'
    else:
        verdict = 'pass'
    reasons = []
    if not train_f:
        reasons.append('没找到列车')
    if missing:
        reasons.append(f'缺帧 {missing}' + ('（在车内）' if miss_in else ''))
    for p in per:
        if p.get('flags'):
            reasons.append(f"第 {p['f']} 帧: {','.join(p['flags'])}")
    readonly = all(p.get('readonly', True) for p in per)
    camera = args.camera or src.name
    io.write_json(out / 'trip.json', {'trip': args.trip, 'src_dir': io.rel(src), 'camera': camera, 'n_frames': t.n})
    io.write_json(out / 'qc.json', {
        'trip': args.trip, 'camera': camera, 'frames': t.n, 'missing': missing,
        'train_frames': [f0, f1], 'readonly': readonly, 'thresholds': TH,
        'per_frame': per, 'verdict': verdict, 'reasons': reasons,
    })
    return {'step': 'ingest', 'trip': args.trip, 'verdict': verdict, 'frames': t.n, 'missing': len(missing),
            'flagged': len(bad), 'train_frames': [f0, f1], 'readonly': readonly, 'out': io.rel(out / 'qc.json')}

