"""S2 拼接：把连续帧按时间顺序横拼成 train_panels（每节车一块），写 manifest 和预览图。

全分辨率整节车不落盘（单节约 360 MB）；后续步骤用 frames.Trip.region 按趟坐标从原图取。
预览图给人看，按设计.md §2 水平翻转，车头在右边。
"""
import cv2
import numpy as np

from . import io
from .coords import W, H
from .frames import Trip

S8 = 8  # 概览缩小倍数


def column_activity(strip):
    """每列与前一列的平均绝对差。静止背景接近噪声，车体明显大。"""
    a = np.abs(np.diff(strip.astype(np.int16), axis=1)).mean(axis=0)
    return np.r_[a[0], a]


def find_span(act, thr):
    on = np.where(act > thr)[0]
    return (int(on.min()), int(on.max()) + 1) if len(on) else (None, None)


def find_car_gaps(s8, x0, x1, min_gap=60):
    """车体裙板（顶部亮带）在车厢连接处断开。s8 是整趟 1/8 条带，返回车内各断口中心（1/8 列号）。"""
    band = s8[6:70, x0:x1].mean(axis=0)
    body = np.median(band)
    dark = band < 0.55 * body
    gaps, i, n = [], 0, len(dark)
    while i < n:
        if dark[i]:
            j = i
            while j < n and dark[j]:
                j += 1
            if j - i >= min_gap and i > 400 and j < n - 400:
                gaps.append(x0 + (i + j) // 2)
            i = j
        else:
            i += 1
    return gaps


def refine_edge(t, x_coarse, thr, rising):
    """在全分辨率上把车头/车尾定到列：±1500 px 窗口里找活动量越过阈值的第一列/最后一列。"""
    a0 = max(0, x_coarse - 1500)
    reg = t.region(a0, min(t.width, x_coarse + 1500), step=1)[::4]
    act = column_activity(reg)
    k = np.convolve(act > thr, np.ones(9), 'same') >= 7  # 连续 7/9 列超阈值才算
    idx = np.where(k)[0]
    if not len(idx):
        return x_coarse
    return int(a0 + (idx.min() if rising else idx.max() + 1))


def seam_check(t, f0, f1):
    """车内每个帧接缝：接缝处列差 / 帧内末 50 列的列差中位数。实测正常约 1.2。"""
    out = []
    for f in range(f0, f1):
        a, b = t.frame(f), t.frame(f + 1)
        if a is None or b is None:
            out.append({'after_frame': f, 'ratio': None, 'flag': 'missing'})
            continue
        a4, b4 = a[::4].astype(np.int16), b[::4].astype(np.int16)
        seam = np.abs(b4[:, 0] - a4[:, -1]).mean()
        inner = np.median(np.abs(np.diff(a4[:, -51:], axis=1)).mean(axis=0))
        r = float(seam / (inner + 1e-6))
        out.append({'after_frame': f, 'x': f * W, 'ratio': round(r, 3), 'flag': 'seam' if r > 4 else None})
    return out


def add_parser(sub, step):
    p = sub.add_parser(step, help='S2 拼接：manifest（train_panels）和预览图')
    p.add_argument('trip')
    p.add_argument('--car-previews', action='store_true', help='另出每节车 1/2 预览（每张约 17 MB）')
    p.set_defaults(func=run)


def run(args):
    t = Trip.from_id(args.trip)
    out = io.trip_dir(args.trip)
    s8 = t.strip(1, t.n, S8)
    act = column_activity(s8)
    w8 = s8.shape[1]
    base = np.median(np.sort(act)[: max(50, w8 // 10)])
    thr8 = 3 * base + 1
    c0, c1 = find_span(np.convolve(act, np.ones(25) / 25, 'same'), thr8)
    if c0 is None:
        raise SystemExit('没找到列车')
    # 全分辨率定车头、车尾到列；阈值用全分辨率空场的列差
    blank = t.region(0, min(4000, c0 * S8 - 2000), step=1)[::4] if c0 * S8 > 6000 else None
    thr1 = 3 * np.median(column_activity(blank)) + 1 if blank is not None else 3 * base + 1
    x0 = refine_edge(t, c0 * S8, thr1, rising=True)
    x1 = refine_edge(t, c1 * S8, thr1, rising=False)
    gaps = [g * S8 for g in find_car_gaps(s8, c0, c1)]
    edges = [x0] + gaps + [x1]
    panels = [{'panel': k, 'car': k, 'x0': a, 'x1': b, 'width_px': b - a, 'height_px': H}
              for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1)]
    f0, f1 = x0 // W + 1, (x1 - 1) // W + 1
    seams = seam_check(t, f0, f1)
    bad_seams = [s for s in seams if s['flag']]
    manifest = {
        'trip': args.trip, 'src_dir': io.rel(t.src), 'n_frames': t.n, 'frame_w': W, 'frame_h': H,
        'width_px': t.width, 'height_px': H, 'display_flip': True,
        'train_x0': x0, 'train_x1': x1, 'head_x': x0, 'train_frames': [f0, f1],
        'blank': [[0, x0], [x1, t.width]],
        'head_protect': {'x0': max(0, x0 - 1000), 'x1': x0 + 8000},
        'train_panels': panels,
        'seams': seams, 'bad_seams': len(bad_seams),
    }
    io.write_json(out / 'manifest.json', manifest)
    # 预览：翻转显示，车头在右
    pv = out / '预览'
    whole = s8[:, x0 // S8: x1 // S8][:, ::-1]
    io.imwrite(pv / '整车_1-8.jpg', whole)
    rows = []
    wmax = max(p['width_px'] for p in panels) // 16
    for p in panels:
        seg = s8[:, p['x0'] // S8: p['x1'] // S8][:, ::-1]
        seg = cv2.resize(seg, (seg.shape[1] // 2, seg.shape[0] // 2), interpolation=cv2.INTER_AREA)
        row = np.zeros((seg.shape[0] + 30, wmax + 2), np.uint8)
        row[30:, wmax + 2 - seg.shape[1]:] = seg  # 右对齐：车头一侧对齐
        cv2.putText(row, f"car {p['car']}  x {p['x0']}-{p['x1']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        rows.append(row)
    io.imwrite(pv / '分节总览_1-16.jpg', np.vstack(rows))
    if args.car_previews:
        for p in panels:
            io.imwrite(pv / f"车厢{p['car']}_1-2.jpg", t.region(p['x0'], p['x1'], step=2)[:, ::-1])
    return {'step': 'stitch', 'trip': args.trip, 'train_x': [x0, x1], 'train_frames': [f0, f1],
            'cars': [p['width_px'] for p in panels], 'bad_seams': len(bad_seams), 'out': io.rel(out / 'manifest.json')}
