"""S3 打尺标定：用车上的已知尺寸把像素换成毫米。

做法（2026-11-17 在 T01/T02 上实测）：

- 轴箱端盖是圆的，圆心就是车轴中心。每节车先用霍夫圆在车轴高度那一带找 4 个端盖，
  再按几何配成 2 台转向架：同一台两轴中心距 = 固定轴距（2300 mm），
  同一节车两台转向架中心距 = 车辆定距（12600 mm）。
- 左端盖（x 小的那台轴，平圆盖）外面有一圈暗环：沿 180 条射线在 [330, 490) px 内取最暗点，
  鲁棒拟合椭圆。圆心给横向位置，a/b 给 sy/sx（a、b 是水平、竖直像素半轴）。
- 右端盖（球面凸盖）中心有一个小凸台，用凸台的圆边定圆心；两种盖在同一根轴上，
  圆心都是车轴中心。
- 横向比例沿车身会漂，所以 12 台转向架各给一个 sx 样本（在转向架中心 x），
  两台之间线性插值、两端按端点斜率外推；6 节车的定距样本当另一种尺子，互相印证。
- 纵向是透视：圆的纵横比只在端盖那一层深度成立，别处的 sy 要用该处的深度换算，报告写明。

已知尺寸的出处见 参考/车型资料/ 和 参考/README.md。车型具体型号未知，按 B 型车通用参数。
"""
import numpy as np
import cv2

from . import io
from .frames import Trip

# 已知尺寸（mm）。出处见 参考/README.md 的车型资料索引。
WHEELBASE_MM = 2300.0      # 固定轴距：GB/T 7928 给 B 型车 2000~2300；广州 3 号线、北京 DKZ76 均 2300
BOGIE_CENTER_MM = 12600.0  # 车辆定距：GB/T 7928 表、中车天津地铁 2 号线（B1 型）
WHEEL_DIAM_MM = 840.0      # 新轮轮径：中车天津地铁 2 号线、北京 DKZ76

# 车轴中心所在的行：相机机位固定，两趟实测 2750~2840。
Y_BAND = (2400, 3400)
HOUGH = dict(dp=1.2, minDist=130, param1=100, param2=38, minRadius=85, maxRadius=115)
STEP = 4
CAP_BAND = (330, 490)   # 左端盖（平盖）外圈暗环
HUB_BAND = (55, 115)    # 右端盖（凸盖）中心凸台
CAP_CROP = 600
HUB_CROP = 300


def add_parser(sub, step):
    p = sub.add_parser(step, help='S3 打尺标定：像素→毫米（轴距/定距/端盖圆）')
    p.add_argument('trip')
    p.add_argument('--out', default=None, help='输出路径，默认 数据/中间/<趟>/calib.json')
    p.set_defaults(func=run)


# ---------------------------------------------------------------- 圆/椭圆

def _ring_points(im, cx, cy, lo, hi, n=180, step=1.0):
    """从 (cx, cy) 出发沿 n 条射线，在半径 [lo, hi) 内找最暗的点（亚像素抛物线）。"""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rs = np.arange(float(lo), float(hi), step)
    xs = (cx + rs[None, :] * np.cos(th)[:, None]).astype(np.float32)
    ys = (cy + rs[None, :] * np.sin(th)[:, None]).astype(np.float32)
    v = cv2.remap(np.ascontiguousarray(im, dtype=np.float32), xs, ys, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)
    k = np.clip(v.argmin(1), 1, v.shape[1] - 2)
    i = np.arange(n)
    y0, y1, y2 = v[i, k - 1], v[i, k], v[i, k + 1]
    den = y0 - 2.0 * y1 + y2
    off = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / np.where(den == 0, 1.0, den), 0.0)
    rr = rs[k] + off * step
    return np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)


def _axis_ab(r1, r2, ang):
    """cv2.fitEllipse 的半轴 (r1, r2)、倾角 ang → 水平半轴 a、竖直半轴 b。"""
    t = np.deg2rad(float(ang))
    c, s = abs(np.cos(t)), abs(np.sin(t))
    return float(np.hypot(r1 * c, r2 * s)), float(np.hypot(r1 * s, r2 * c))


def _radial_residual(pts, cx, cy, a, b):
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    rho = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    re = 1.0 / np.sqrt((np.cos(th) / max(a, 1e-6)) ** 2 + (np.sin(th) / max(b, 1e-6)) ** 2)
    return rho - re


def fit_ellipse(pts, iters=6, k=2.0):
    """轴对齐椭圆的鲁棒拟合：绝对坐标、cv2.fitEllipse、按径向残差去离群点。

    返回 dict(cx, cy, a, b, ang, n, sd, ok)；a 是水平半轴，b 是竖直半轴，
    n 是内点数，sd 是内点径向残差的标准差（px）。
    """
    P = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    if len(P) < 6:
        return dict(cx=0.0, cy=0.0, a=0.0, b=0.0, ang=0.0, n=len(P), sd=0.0, ok=False)
    w = np.ones(len(P), bool)
    cx = cy = a = b = ang = 0.0
    q = np.zeros(len(P))
    for _ in range(iters):
        if int(w.sum()) < 6:
            break
        try:
            (cx, cy), (d1, d2), ang = cv2.fitEllipse(P[w])
        except cv2.error:
            break
        a, b = _axis_ab(d1 / 2.0, d2 / 2.0, ang)
        q = _radial_residual(P[:, 0], cx, cy, a, b)
        med = np.median(q[w])
        mad = np.median(np.abs(q[w] - med)) + 1e-6
        w = np.abs(q - med) < k * 1.4826 * mad
    sd = float(np.std(q[w])) if w.any() else 0.0
    n = int(w.sum())
    off = min(abs(((float(ang) - t + 90.0) % 180.0) - 90.0) for t in (0.0, 90.0, 180.0))
    return dict(cx=float(cx), cy=float(cy), a=float(a), b=float(b), ang=float(ang),
                n=n, sd=sd, off=float(off), ok=bool(n >= 6))


# ---------------------------------------------------------------- 找车轴

def _candidates(trip, panel, y_band=Y_BAND, step=STEP, hough=None):
    """一节车里的候选圆：车轴高度那一带做霍夫，返回 [(x, y, r)]（全分辨率趟坐标）。"""
    y0, y1 = y_band
    im = trip.region(panel['x0'], panel['x1'], y0, y1, step=step)
    im = cv2.GaussianBlur(im, (5, 5), 0)
    cs = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, **(hough or HOUGH))
    if cs is None:
        return []
    return [(float(x * step + panel['x0']), float(y * step + y0), float(r * step)) for x, y, r in cs[0]]


def pick_quad(cands, wheelbase=(8900.0, 11100.0), dy_max=170.0, center=(49500.0, 58000.0)):
    """从候选圆里挑 4 个端盖 = 2 台转向架。

    约束：每对（同一台转向架的两个轴）横向间距在 wheelbase 内、纵向差 < dy_max；
    两对中心（两台转向架中心）间距在 center 内。评分越小越好。
    返回 (score, [(x, y, r), ...])，按 x 排好；凑不齐返回 (None, [])。
    """
    best = None
    n = len(cands)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            d1 = cands[j][0] - cands[i][0]
            if not (wheelbase[0] < d1 < wheelbase[1]) or abs(cands[j][1] - cands[i][1]) > dy_max:
                continue
            for k in range(n):
                for l in range(n):
                    if len({i, j, k, l}) < 4:
                        continue
                    d2 = cands[l][0] - cands[k][0]
                    if not (wheelbase[0] < d2 < wheelbase[1]) or abs(cands[l][1] - cands[k][1]) > dy_max:
                        continue
                    dc = (cands[k][0] + cands[l][0]) / 2.0 - (cands[i][0] + cands[j][0]) / 2.0
                    if not (center[0] < dc < center[1]):
                        continue
                    sc = (abs(d1 - d2) + abs(cands[i][1] - cands[j][1]) + abs(cands[k][1] - cands[l][1])
                          + 0.2 * abs(dc - (center[0] + center[1]) / 2.0))
                    if best is None or sc < best[0]:
                        best = (float(sc), i, j, k, l)
    if best is None:
        return None, []
    _, i, j, k, l = best
    quad = sorted([cands[i], cands[j], cands[k], cands[l]])
    return best[0], quad


def refine_cap(trip, x, y, cap=True, crop=None, y_fix=None, band=None):
    """端盖圆心精定位：平盖用外圈暗环，凸盖用中心凸台；返回拟合结果（趟坐标）+ b/a。"""
    c = crop or (CAP_CROP if cap else HUB_CROP)
    y = y if y_fix is None else y_fix     # 同一台转向架两个轴等高，用另一端定高度
    x0, y0 = int(round(x)) - c, int(round(y)) - c
    im = trip.region(x0, x0 + 2 * c, y0, y0 + 2 * c)
    band = band or (CAP_BAND if cap else HUB_BAND)
    pts = _ring_points(im, float(c), float(c), *band)
    f = fit_ellipse(pts)
    f['x'] = x0 + f['cx']
    f['y'] = y0 + f['cy']
    f['ratio'] = f['b'] / f['a'] if f['a'] > 0 else float('nan')
    f['kind'] = 'cap' if cap else 'hub'
    # 圆盖和凸台的验收标准不同：凸台小、边缘糙，只要圆心稳；平盖还要 b/a 能用。
    if cap:
        # 近圆时椭圆倾角没有意义，只有明显偏圆（长短轴差 > 5%）才查倾角。
        flat = abs(f['a'] - f['b']) / max(f['a'], f['b']) < 0.05
        f['ok'] = (f['n'] >= 0.6 * len(pts) and f['sd'] <= 8.0
                   and (f['off'] <= 30.0 or flat) and 0.90 < f['ratio'] < 1.12)
    else:
        f['ok'] = f['n'] >= 0.5 * len(pts) and f['sd'] <= 25.0
    if abs(f['cx'] - c) > 0.4 * c or abs(f['cy'] - c) > 0.4 * c:
        f['ok'] = False          # 圆心跑太远，说明拟合咬到了别的东西
    return f


def detect_axles(trip, manifest, quad=None, band_cap=CAP_BAND, band_hub=HUB_BAND,
                 crop_cap=CAP_CROP, crop_hub=HUB_CROP, wb_range=(8500.0, 11500.0), **kw):
    """整趟车 12 台转向架：每节车 2 台，每台两个轴（左平盖、右凸台）。

    返回 (bogies, cars)。bogies 每项：car、zone（B1/B2）、left、right、wheelbase_px、x（转向架中心）。
    """
    bogies, cars = [], []
    for panel in manifest['train_panels']:
        cands = _candidates(trip, panel, **kw)
        _, picked = pick_quad(cands, **(quad or {}))
        car = {'car': panel['car'], 'candidates': len(cands), 'found': len(picked) == 4}
        if len(picked) < 4:
            cars.append(car)
            continue
        for z, (a, b) in enumerate([(picked[0], picked[1]), (picked[2], picked[3])], start=1):
            left = refine_cap(trip, a[0], a[1], cap=True, band=band_cap, crop=crop_cap)
            right = refine_cap(trip, b[0], b[1], cap=False, y_fix=left['y'], band=band_hub, crop=crop_hub)
            wb = right['x'] - left['x']
            bogies.append({
                'car': panel['car'], 'zone': f'B{z}',
                'left': left, 'right': right,
                'wheelbase_px': float(wb),
                'x': float((left['x'] + right['x']) / 2.0),
                'ok': bool(left['ok'] and right['ok'] and wb_range[0] < wb < wb_range[1]
                           and abs(right['y'] - left['y']) < 40.0),
            })
        car['bogies'] = [b['zone'] for b in bogies if b['car'] == panel['car']]
        cars.append(car)
    return bogies, cars


# ---------------------------------------------------------------- 比例与曲线

def sx_at(calib, x):
    """按 sx_curve 求 x 处的 mm/px：节点间线性插值，两端按端点斜率外推。"""
    pts = calib['sx_curve'] if isinstance(calib, dict) else calib
    if not pts:
        raise ValueError('sx_curve 为空，没法换算')
    xs = np.array([p['x'] for p in pts], float)
    ys = np.array([p['mm_per_px'] for p in pts], float)
    x = np.asarray(x, float)
    if len(xs) == 1:
        return np.full_like(x, ys[0])
    out = np.interp(x, xs, ys)
    k0 = (ys[1] - ys[0]) / (xs[1] - xs[0])
    k1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    out = np.where(x < xs[0], ys[0] + (x - xs[0]) * k0, out)
    out = np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * k1, out)
    return out


def make_curve(samples):
    """把 12 个 sx 样本做成曲线节点：按 x 排序，同一个 x 取中值。"""
    if not samples:
        return []
    by_x = {}
    for s in samples:
        by_x.setdefault(round(float(s['x']), 1), []).append(float(s['mm_per_px']))
    return [{'x': x, 'mm_per_px': float(np.median(v)), 'n': len(v)} for x, v in sorted(by_x.items())]


def _spread(vals):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return None
    m = float(np.mean(v))
    return {'n': int(len(v)), 'mean': round(m, 6), 'sd': round(float(np.std(v)), 6),
            'min': round(float(v.min()), 6), 'max': round(float(v.max()), 6),
            'range_pct': round(float((v.max() - v.min()) / m * 100.0), 3),
            'sd_pct': round(float(np.std(v) / m * 100.0), 3)}


# ---------------------------------------------------------------- 命令

def run(args):
    trip = args.trip
    manifest = io.read_json(io.trip_dir(trip) / 'manifest.json')
    t = Trip.from_id(trip)
    bogies, cars = detect_axles(t, manifest)

    ok_bogies = [b for b in bogies if b['ok']]
    wb_samples, dj_samples, refs = [], [], []
    for b in bogies:
        left, right = b['left'], b['right']
        ref = {
            'name': '固定轴距', 'nominal_mm': WHEELBASE_MM,
            'measured_px': round(b['wheelbase_px'], 2), 'x': round(b['x'], 1),
            'car': b['car'], 'zone': b['zone'],
            'method': '轴箱端盖圆心距：左平盖外圈暗环椭圆 + 右凸盖中心凸台椭圆',
            'sd_px': [round(left['sd'], 2), round(right['sd'], 2)],
            'ok': b['ok'],
        }
        if b['ok']:
            ref['mm_per_px'] = round(WHEELBASE_MM / b['wheelbase_px'], 6)
            wb_samples.append({'x': b['x'], 'mm_per_px': ref['mm_per_px'], 'car': b['car'], 'zone': b['zone']})
        refs.append(ref)

    for c in cars:
        bs = [b for b in bogies if b['car'] == c['car']]
        if len(bs) != 2 or not all(b['ok'] for b in bs):
            continue
        d = bs[1]['left']['x'] - bs[0]['left']['x']
        if not 50000.0 < d < 60000.0:
            continue
        x = float((bs[0]['left']['x'] + bs[1]['left']['x']) / 2.0)
        ref = {
            'name': '车辆定距', 'nominal_mm': BOGIE_CENTER_MM,
            'measured_px': round(float(d), 2), 'x': round(x, 1), 'car': c['car'], 'zone': 'CAR',
            'method': '同一节车两台转向架左端盖圆心距（两台转向架同型，偏心相同，差值里抵消）',
            'sd_px': [round(bs[0]['left']['sd'], 2), round(bs[1]['left']['sd'], 2)],
            'ok': True, 'mm_per_px': round(BOGIE_CENTER_MM / d, 6),
        }
        dj_samples.append({'x': x, 'mm_per_px': ref['mm_per_px'], 'car': c['car']})
        refs.append(ref)

    # 两种尺子互证：每节车里，两个轴距样本的平均 与 该车定距样本 的差
    pairs = []
    for dj in dj_samples:
        w = [s for s in wb_samples if s['car'] == dj['car']]
        if len(w) == 2:
            m = float(np.mean([s['mm_per_px'] for s in w]))
            pairs.append({'car': dj['car'], 'dingju_mm_per_px': dj['mm_per_px'], 'zhouju_mm_per_px': round(m, 6),
                          'diff_pct': round((m - dj['mm_per_px']) / dj['mm_per_px'] * 100.0, 3)})
    for r in refs:
        if r['name'] == '固定轴距' and r.get('mm_per_px'):
            p = next((p for p in pairs if p['car'] == r['car']), None)
            if p:
                r['residual_pct'] = round((r['mm_per_px'] - p['dingju_mm_per_px']) / p['dingju_mm_per_px'] * 100.0, 3)
        elif r['name'] == '车辆定距':
            p = next((p for p in pairs if p['car'] == r['car']), None)
            if p:
                r['residual_pct'] = round(p['diff_pct'], 3)

    curve = make_curve(wb_samples)
    if not curve:
        raise SystemExit(f'{trip}: 没找到可用的轴距样本，标定失败（候选/拟合太差）')
    sx_mean = float(np.mean([p['mm_per_px'] for p in curve]))

    # 同一个圆的物理半径满足 a*sx = b*sy，因此 sy = sx / (b/a)。
    # sy_ratio_ba 保留其原有含义：观测到的像素轴比 b/a，而非 sy/sx。
    ratios = [b['left']['ratio'] for b in bogies if b['left']['ok'] and np.isfinite(b['left']['ratio'])]
    sy_ratio = float(np.median(ratios)) if ratios else 1.0
    sy_vals = [float(sx_at({'sx_curve': curve}, b['left']['x'])) / b['left']['ratio']
               for b in bogies if b['left']['ok'] and np.isfinite(b['left']['ratio'])]
    sy = float(np.median(sy_vals)) if sy_vals else sx_mean / sy_ratio

    out = {
        'trip': trip,
        'sx_curve': curve,
        'sx_mean_mm_per_px': round(sx_mean, 6),
        'sy_mm_per_px': round(sy, 6),
        'sy_ratio_ba': round(sy_ratio, 5),
        'refs': refs,
        'bogies': [{'car': b['car'], 'zone': b['zone'], 'x': round(b['x'], 1),
                    'left': [round(b['left']['x'], 1), round(b['left']['y'], 1)],
                    'right': [round(b['right']['x'], 1), round(b['right']['y'], 1)],
                    'wheelbase_px': round(b['wheelbase_px'], 1),
                    'cap_a': round(b['left']['a'], 1), 'cap_b': round(b['left']['b'], 1),
                    'b_over_a': round(b['left']['ratio'], 4), 'sd': [round(b['left']['sd'], 2), round(b['right']['sd'], 2)],
                    'ok': b['ok']} for b in bogies],
        'consistency': {
            'wheelbase': _spread([s['mm_per_px'] for s in wb_samples]),
            'bogie_center': _spread([s['mm_per_px'] for s in dj_samples]),
            'two_rulers': pairs,
            'b_over_a': _spread(ratios),
            'sy': _spread(sy_vals),
        },
        'source': {
            'vehicle': '按 B 型车通用参数（具体型号未知；有准确车型参数时替换）',
            'wheelbase_mm': {'value': WHEELBASE_MM, 'basis': 'GB/T 7928 B 型车 2000~2300；本车像素比 定距/轴距≈5.48=12600/2300 支持 2300',
                             'files': ['参考/车型资料/crrc_广州3号线增购.html', '参考/车型资料/wiki_北京DKZ76.html']},
            'bogie_center_mm': {'value': BOGIE_CENTER_MM, 'basis': 'GB/T 7928 表 + 中车天津地铁 2 号线（B1 型）',
                                'files': ['参考/车型资料/crrc_天津2号线.html']},
            'wheel_diameter_mm': {'value': WHEEL_DIAM_MM, 'basis': '新轮轮径，中车天津地铁 2 号线、北京 DKZ76',
                                  'files': ['参考/车型资料/crrc_天津2号线.html', '参考/车型资料/wiki_北京DKZ76.html']},
            'sx_method': '固定轴距 2300 mm / 端盖圆心距（12 处），两台之间线性插值，两端外推',
            'sy_method': '横向比例 ÷ 平端盖椭圆 b/a（a*sx=b*sy）；只在端盖那一层深度成立',
        },
    }
    path = args.out or (io.trip_dir(trip) / 'calib.json')
    io.write_json(path, out)
    return {'step': 'calib', 'trip': trip, 'bogies': f"{len(ok_bogies)}/{len(bogies)}",
            'sx_mean_mm_per_px': round(sx_mean, 4), 'sx_curve': len(curve),
            'sy_mm_per_px': round(sy, 4), 'sy_ratio_ba': round(sy_ratio, 4),
            'sx_sd_pct': out['consistency']['wheelbase']['sd_pct'] if out['consistency']['wheelbase'] else None,
            'dingju_n': len(dj_samples), 'out': io.rel(path)}

