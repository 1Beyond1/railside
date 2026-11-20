"""定位靶：在模板趟上自动挑选，在新趟上自动识别。

挑选标准：纹理强（灰度标准差、梯度能量），附近 ±3000 px 内不重复（自相关第二峰低），
每节车至少 8 个、沿车长铺开。结果写 标注/模板T01/定位靶.json，人工看总览图确认。
"""
import cv2
import numpy as np

from . import io
from .frames import Trip

PATCH = 512          # 全分辨率靶块边长
STEP = 2             # 匹配用的缩小倍数
ROWS = (1800, 2400, 3000, 3400)  # 靶块中心行候选：避开顶部裙板和司机室外壳（平坦、反光随趟变）


def _patch(t, x, y, step=STEP, size=PATCH):
    h = size // 2
    return t.region(x - h, x + h, y - h, y + h, step=step).astype(np.float32)


def texture(p):
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1)
    return float(p.std()), float(np.sqrt(gx * gx + gy * gy).mean())


def uniqueness(t, x, y, p, reach=3000):
    """以 (x, y) 为中心的块在 ±reach 内的第二高 NCC（越低越独特）。"""
    h = PATCH // 2
    reg = t.region(x - reach - h, x + reach + h, y - h - 64, y + h + 64, step=STEP).astype(np.float32)
    r = cv2.matchTemplate(reg, p, cv2.TM_CCOEFF_NORMED)
    c = r.shape[1] // 2
    r[:, max(0, c - 48): c + 48] = -1   # 去掉自身附近 ±96 px
    return float(r.max())


def select(trip="T01", per_car=12, spacing=4000, every=1000):
    m = io.read_json(io.trip_dir(trip) / 'manifest.json')
    t = Trip.from_id(trip)
    out = []
    for p in m['train_panels']:
        cands = []
        for x in range(p['x0'] + 1200, p['x1'] - 1200, every):
            best = None
            for y in ROWS:
                pt = _patch(t, x, y)
                sd, ge = texture(pt)
                if sd < 18 or ge < 12:
                    continue
                if best is None or ge > best[2]:
                    best = (y, sd, ge, pt)
            if best is None:
                continue
            y, sd, ge, pt = best
            u2 = uniqueness(t, x, y, pt)
            if u2 > 0.65:
                continue
            cands.append({'x': x, 'y': y, 'score': round(ge * (1 - u2), 2), 'tex': round(ge, 1), 'second_peak': round(u2, 3)})
        chosen = []
        for c in sorted(cands, key=lambda c: -c['score']):
            if all(abs(c['x'] - d['x']) >= spacing for d in chosen):
                chosen.append(c)
            if len(chosen) >= per_car:
                break
        # 车头段保护要用：1 号车车头 8000 px 内至少一个靶
        if p['car'] == 1 and not any(c['x'] < p['x0'] + 8000 for c in chosen):
            head = [c for c in cands if c['x'] < p['x0'] + 8000]
            if head:
                chosen.append(max(head, key=lambda c: c['score']))
        for c in sorted(chosen, key=lambda c: c['x']):
            out.append({'id': f"A{p['car']}{len([o for o in out if o['car'] == p['car']]) + 1:02d}", 'car': p['car'],
                        'x': c['x'] - PATCH // 2, 'y': c['y'] - PATCH // 2, 'w': PATCH, 'h': PATCH,
                        'score': c['score'], 'second_peak': c['second_peak']})
    return out


def locate(t_new, t_tpl, anchor, x_pred, y_pred=None, reach_x=1500, reach_y=60):
    """在新趟上找模板靶。先 1/2 尺度粗搜，再全分辨率 ±6 px 精搜，抛物线插值到亚像素。

    返回 (x, y, ncc, peak_ratio)：x、y 是靶块左上角在新趟的趟坐标。
    """
    x0, y0, w, h = anchor['x'], anchor['y'], anchor['w'], anchor['h']
    y_pred = y0 if y_pred is None else y_pred
    tp = t_tpl.region(x0, x0 + w, y0, y0 + h, step=STEP).astype(np.float32)
    sx0 = int(x_pred) - reach_x
    sy0 = max(0, int(y_pred) - reach_y)
    sy1 = min(4096, int(y_pred) + h + reach_y)
    reg = t_new.region(sx0, sx0 + w + 2 * reach_x, sy0, sy1, step=STEP).astype(np.float32)
    r = cv2.matchTemplate(reg, tp, cv2.TM_CCOEFF_NORMED)
    _, v1, _, (cx, cy) = cv2.minMaxLoc(r)
    rr = r.copy()
    rr[max(0, cy - 20): cy + 21, max(0, cx - 20): cx + 21] = -1
    v2 = float(rr.max()) if rr.size else -1
    xc, yc = sx0 + cx * STEP, sy0 + cy * STEP
    # 全分辨率精搜
    tp1 = t_tpl.region(x0, x0 + w, y0, y0 + h, step=1).astype(np.float32)
    k = 6
    reg1 = t_new.region(xc - k, xc + w + k, max(0, yc - k), min(4096, yc + h + k), step=1).astype(np.float32)
    r1 = cv2.matchTemplate(reg1, tp1, cv2.TM_CCOEFF_NORMED)
    _, v, _, (bx, by) = cv2.minMaxLoc(r1)
    dx = _parabola(r1[by, max(0, bx - 1): bx + 2]) if 0 < bx < r1.shape[1] - 1 else 0.0
    dy = _parabola(r1[max(0, by - 1): by + 2, bx]) if 0 < by < r1.shape[0] - 1 else 0.0
    return xc - k + bx + dx, max(0, yc - k) + by + dy, float(v), float(v1 - max(v2, 0))


def _parabola(v):
    if len(v) != 3:
        return 0.0
    d = v[0] - 2 * v[1] + v[2]
    return float(0.5 * (v[0] - v[2]) / d) if d < 0 else 0.0
