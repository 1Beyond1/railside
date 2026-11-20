"""S8 两趟对比：逐件比外观、位移、转角、面积，报缺失和异物，出告警（设计.md §3 S8、§4.1）。

- 外观：局部再配准（亚像素）、亮度归一后，在零件掩膜内算 NCC；
- 几何：质心偏移按 calib 换算毫米（没有 calib 时毫米字段为 null）；子件报相对父件的偏移，父件报相对整车对齐的偏移；
- 转角：±30° 里搜 Δθ，螺栓这类 5° 级的转动要测得出；
- 面积：看配准得到的局部比例（像素匹配）或分割掩膜（seg 找到的），和模板面积比；
- 缺失：S6b 的 missing；异物：父件框内、零件掩膜外、对齐后差得成片的区域，
  分割出来却没对上任何零件的检出也算（src=seg）；
- qc2 标了质量差的位置：status=skip + qc 告警，不判别的。

阈值按类型读 主数据/零件类型.csv，缺列或缺文件用下面的默认值。
--suggest 仅允许 T01 开发数据，T02 只评测、不产生调参建议。
"""
import csv
from collections import defaultdict

import cv2
import numpy as np

from . import io
from .coords import AlignMap
from .frames import Trip
from .match import aligned_region, read_region, _centroid, _resize, _tpl_patch
from .template import crop_margin, load_template, resolve_ver

DEFAULT = {'ncc_min': 0.65, 'shift_mm_max': 4.0, 'dtheta_max_deg': 3.0, 'area_change_max': 0.35}
MAXWORK = 420          # 配准时局部图长边的目标像素
THETA_RANGE = 30.0     # ±30° 搜转角
COARSE_THETA_STEP = 5.0
THETA_COARSE = 0.5
THETA_FINE = 0.1
SCALE_GRID = (0.88, 0.94, 1.0, 1.06, 1.12)
SIG = 0.003            # 转角的 NCC 增益小于这个值就当测不出（对称件的曲线是平的）
QC_HALF = 512          # qc2 块半宽（align.BLOCK/2）
FOREIGN_DIFF = 30.0    # 对齐并做增益归一后，灰度差超过这个值算“不一样”（0–255）
FOREIGN_MIN_AREA = 1200
FOREIGN_MIN_SCORE = 35.0
FOREIGN_DILATE = 9
FILL_MIN = 0.35        # 连通块面积 / 外框面积：太细长的不算异物
ROTATING = {'WHEEL'}   # 会转的件：标注只盖构架以下的可见部分，异物检测把整个轮盘排除（零件类型.csv“车轮会转”）


# ---------------------------------------------------------------- 参数与阈值

def add_parser(sub, step):
    p = sub.add_parser(step, help='S8 两趟对比：外观、位移、转角、面积、告警')
    p.add_argument('trip')
    p.add_argument('--template', default=None, help='模板版本；默认取 parts.json / align.json 里的')
    p.add_argument('--parts', default=None, help='默认 数据/中间/<趟>/parts.json')
    p.add_argument('--types', default='主数据/零件类型.csv', help='阈值表；缺列或缺文件用代码默认值')
    p.add_argument('--out', default=None, help='默认 数据/中间/<趟>/compare.json')
    p.add_argument('--suggest', action='store_true', help='按本趟统计建议阈值，写 --suggest-out')
    p.add_argument('--suggest-out', default='实验/E3', help='建议阈值的输出目录')
    p.add_argument('--quantile', type=float, default=99.0, help='取这个分位数（默认 p99）')
    p.add_argument('--relax', type=float, default=1.5, help='在分位数上再放宽的倍数')
    p.add_argument('--foreign-diff', type=float, default=FOREIGN_DIFF)
    p.add_argument('--foreign-min-area', type=int, default=FOREIGN_MIN_AREA)
    p.add_argument('--foreign-min-score', type=float, default=FOREIGN_MIN_SCORE)
    p.set_defaults(func=run)


def load_thresholds(path):
    """主数据/零件类型.csv → {type: 阈值 dict}；文件没有就返回 {}（用 DEFAULT）。"""
    p = path if hasattr(path, 'exists') else io.ROOT / str(path)
    if not p.exists():
        print(f'提示：没有 {io.rel(p) if p.is_relative_to(io.ROOT) else p}，阈值全用代码默认值 {DEFAULT}')
        return {}
    rows = {}
    with open(p, encoding='utf-8-sig', newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get('type'):
                rows[r['type']] = r
    return rows


def thresh_of(types, typ):
    row = types.get(typ) or {}
    out = {}
    for k, d in DEFAULT.items():
        v = (row.get(k) or '').strip()
        try:
            out[k] = float(v) if v else d
        except ValueError:
            out[k] = d
    return out


def _sx_of(calib):
    """calib.json 的 sx_curve → x(mm/px) 函数。"""
    if not calib or not calib.get('sx_curve'):
        return None
    xs = np.array([p['x'] for p in calib['sx_curve']], float)
    ss = np.array([p['mm_per_px'] for p in calib['sx_curve']], float)
    if len(xs) < 2:
        return lambda x: float(ss[0])
    return lambda x: float(np.interp(float(x), xs, ss))


# ---------------------------------------------------------------- 配准与指标

def _ncc(a, b, m):
    aa, bb = a[m], b[m]
    if aa.size < 12:
        return -1.0
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    return float((aa * bb).sum() / den) if den > 0 else -1.0


def _parabola(y0, y1, y2):
    d = y0 - 2 * y1 + y2
    return 0.0 if abs(d) < 1e-9 else float(0.5 * (y0 - y2) / d)


def offset_of(it, q, amap):
    """S6b 在对齐后的图上量到的平移 (du, dv)（模板 px）；旧 parts.json 没有这个字段时按质心反算。

    大件不能拿“本趟多边形质心 − 模板质心映射过去的位置”当偏移：对齐映射在十几米长的转向架上
    横向比例不是常数，两种质心本来就差几十到上百 px（真实 T02 曾因此报出 442 条假位移）。"""
    o = it.get('offset_uv')
    if o is not None:
        return float(o[0]), float(o[1])
    u, v = amap.to_template(*it['centroid_xy'])
    return float(u) - q['centroid_uv'][0], float(v) - q['centroid_uv'][1]


def measure(t_new, q, it, amap):
    """对比一个零件：配准（比例、转角、亚像素平移）后算 NCC、Δθ、面积比。

    配准中心用 S6b 给的实测质心（模板质心应该落在它上面），再在这个中心附近搜比例、转角和平移。
    返回 {'ncc','ncc0','dtheta','s','s_amap','w'}。
    """
    P, M = _tpl_patch(q)
    hp, wp = P.shape
    b = [int(v) for v in it['bbox_xy']]
    mx, my = crop_margin(b[2] - b[0], b[3] - b[1])
    rad = max(8, int(0.15 * max(wp, hp)))    # 留够边：s 在 ±12%、转角 ±30° 时还采得到整块
    rx0, ry0 = b[0] - mx - rad, b[1] - my - rad
    R, org = read_region(t_new, rx0, b[2] + mx + rad, ry0, b[3] + my + rad)
    if R.shape[0] < 8 or R.shape[1] < 8:
        return None
    w = min(1.0, MAXWORK / max(hp, wp))
    Pq, Mq, Rq = _resize(P, w), _resize(M, w), _resize(R, w)
    if Pq.shape[0] > Rq.shape[0] or Pq.shape[1] > Rq.shape[1]:
        return None
    cu, cv = q['centroid_uv']
    s0 = float(amap.scale_at_u(cu))
    cT = _centroid(Mq) or (wp * w / 2.0, hp * w / 2.0)
    du, dv = offset_of(it, q, amap)
    xm, ym = amap.to_trip(cu + du, cv + dv)
    cR = ((float(xm) - org[0]) * w, (float(ym) - org[1]) * w)

    # 比例 × 转角粗搜（转角和比例互相影响，一起搜），再交替细化
    best = (-2.0, 0.0, s0)
    for k in SCALE_GRID:
        sk = s0 * k
        for t in np.arange(-THETA_RANGE, THETA_RANGE + 1e-9, COARSE_THETA_STEP):
            v = _ncc_at(Pq, Mq, Rq, float(t), sk, cR, cT, (0.0, 0.0), s0=s0)
            if v > best[0]:
                best = (v, float(t), sk)
    ncc, th, s = best
    v, t0 = _theta_scan(Pq, Mq, Rq, s, cR, cT,
                        np.arange(th - COARSE_THETA_STEP, th + COARSE_THETA_STEP + 1e-9, THETA_COARSE), (0.0, 0.0), s0)
    th, ncc = t0, max(ncc, v)
    # 比例细搜（0.5% 一档），面积变化就是从这里量出来的
    best = (-2.0, s)
    for k in np.arange(0.98, 1.0201, 0.005):
        v = _ncc_at(Pq, Mq, Rq, th, s * k, cR, cT, (0.0, 0.0), s0=s0)
        if v > best[0]:
            best = (v, s * k)
    s, ncc = best[1], max(ncc, best[0])
    # 亚像素平移：整数网格 + 抛物线
    rr = 3
    grid = {(dx, dy): _ncc_at(Pq, Mq, Rq, th, s, cR, cT, (float(dx), float(dy)), s0=s0)
            for dy in range(-rr, rr + 1) for dx in range(-rr, rr + 1)}
    (bx, by), ncc = max(grid.items(), key=lambda kv: kv[1])
    fx, fy = float(bx), float(by)
    if -rr < bx < rr:
        fx = bx + _parabola(grid[(bx - 1, by)], grid[(bx, by)], grid[(bx + 1, by)])
    if -rr < by < rr:
        fy = by + _parabola(grid[(bx, by - 1)], grid[(bx, by)], grid[(bx, by + 1)])
    ncc = max(ncc, _ncc_at(Pq, Mq, Rq, th, s, cR, cT, (fx, fy), s0=s0))
    # 转角再细搜一档
    v, t2 = _theta_scan(Pq, Mq, Rq, s, cR, cT,
                        np.arange(th - THETA_COARSE, th + THETA_COARSE + 1e-9, THETA_FINE), (fx, fy), s0)
    if v > ncc:
        th, ncc = t2, v
    ncc0 = _ncc_at(Pq, Mq, Rq, 0.0, s, cR, cT, (fx, fy), s0=s0)
    if ncc - ncc0 <= SIG:    # 相关曲线是平的：这个形状分不出转角
        th = 0.0
    return {'ncc': round(float(ncc), 4), 'ncc0': round(float(ncc0), 4), 'dtheta': round(float(th), 2),
            's': float(s), 's_amap': s0, 'w': w}

def _ncc_at(Pq, Mq, Rq, th, s, cR, cT, d, s0=1.0):
    """零件掩膜内、亮度归一后的 NCC：新趟按 s、θ 采样到模板帧再比。

    比较区域固定是模板掩膜（不随 s 涨缩），所以不会靠“缩小比较区”把分数抬上去；
    采样落到搜索区外的像素剔掉（search 区要留够边，见 _measure 里的 rad）。
    """
    A = _aff(th, s, cR, cT, d, s0)
    size = (Pq.shape[1], Pq.shape[0])
    Q = cv2.warpAffine(Rq, A, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inb = cv2.warpAffine(np.full(Rq.shape, 255, np.uint8), A, size,
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _ncc(Pq, Q, (Mq > 127) & (inb > 127))


def _aff(th, s, cR, cT, d, s0=1.0):
    """模板帧 → 新趟帧的仿射：q = diag(1, 1/s0)·s·Rot(−θ)(p − cT) + cR + d。

    s0 是对齐映射给的横向比例（线阵相机的时间轴随车速伸缩，纵向不变），s = s0·k：
    横向总比例 s0·k，纵向只有零件自己的 k。"""
    a = np.radians(th)
    M = np.diag([1.0, 1.0 / s0]) @ (np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]], float) * s)
    off = np.asarray(cR, float) + np.asarray(d, float) - M @ np.asarray(cT, float)
    return np.hstack([M, off[:, None]])


def _theta_scan(Pq, Mq, Rq, s, cR, cT, thetas, d, s0=1.0):
    best = (-2.0, 0.0)
    for th in thetas:
        v = _ncc_at(Pq, Mq, Rq, float(th), s, cR, cT, d, s0)
        if v > best[0]:
            best = (v, float(th))
    return best


# ---------------------------------------------------------------- 异物

def _gain(a, b):
    """估 b ≈ g·a + o 的增益和偏置：用分位数，不怕被一小块异物带偏。"""
    av, bv = a.ravel().astype(np.float32), b.ravel().astype(np.float32)
    pa5, pa95 = np.percentile(av, [5, 95])
    pb5, pb95 = np.percentile(bv, [5, 95])
    g = float((pb95 - pb5) / (pa95 - pa5)) if pa95 - pa5 > 1e-6 else 1.0
    return g, float(pb5 - g * pa5)


def _raster(poly, x0, y0, shape, s):
    """趟坐标多边形画到 (x0, y0) 起、缩小 s 倍的掩膜里。"""
    m = np.zeros(shape, np.uint8)
    p = (np.asarray(poly, float) - [x0, y0]) / s
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
    return m


def _disc(bbox, ddu, ddv, x0, y0, shape, s):
    """车轮的整个轮盘：直径 = 外框宽，圆底贴外框下沿（标注是构架以下那段，宽就是轮径）。"""
    a, _, b, d = [float(v) for v in bbox]
    r = (b - a) / 2
    m = np.zeros(shape, np.uint8)
    c = ((a + b) / 2 + ddu - x0) / s, (d - r + ddv - y0) / s
    cv2.circle(m, (int(round(c[0])), int(round(c[1]))), int(round(r / s)), 255, -1)
    return m


def find_foreign(t_tpl, t_new, tpl, tparts, inst, extra, amap, cfg):
    """父件框内、零件掩膜外、对齐后差得成片的地方。

    只在有子件的父件（转向架这类“容器”）里找。没有子件的顶层件（设备箱、风缸、标记）框里全是它自己，
    差异是它本身的外观变化，归 ncc/area 告警管；当异物算会把反光、贴纸都报成异物。"""
    out = []
    head = tpl.get('head_x') or 0
    tmap = {k['code']: k for k in tparts}
    containers = {k.get('parent') for k in tparts if k.get('parent')}
    for q in tparts:
        if q.get('parent') or q['code'] not in containers:
            continue
        it = inst.get(q['code'])
        if not it or it['found_by'] == 'missing':
            continue
        u0, v0, u1, v1 = [int(v) for v in q['bbox_uv']]
        area = max(1, (u1 - u0) * (v1 - v0))
        S = 1 if area < 6e6 else (2 if area < 2.4e7 else 3)
        du, dv = offset_of(it, q, amap)
        a = t_tpl.region(u0 + head, u1 + head, v0, v1, step=S)
        # 本趟按对齐映射 + 父件平移采到模板网格上：两张图逐像素对应
        wa = aligned_region(t_new, amap, u0 + du, u0 + du + a.shape[1] * S, v0 + dv, v0 + dv + a.shape[0] * S, S)
        if wa.shape != a.shape or a.shape[0] < 4 or a.shape[1] < 4:
            continue
        g, o = _gain(a, wa)
        d = cv2.blur(np.abs(wa.astype(np.float32) - (g * a.astype(np.float32) + o)), (3, 3))
        d = np.maximum(d - cv2.blur(d, (151, 151)) - 10.0, 0)   # 去掉沿车身的缓变残留
        keep = np.ones(d.shape, bool)
        for k in tparts:
            if k.get('parent') != q['code'] or k['code'] not in inst:
                continue
            ki = inst[k['code']]
            if ki['found_by'] == 'missing':
                continue
            kdu, kdv = offset_of(ki, k, amap)
            poly = np.asarray(k['poly_uv'], float) + [kdu - du, kdv - dv]   # 子件在这张网格上的位置
            mk = _raster(poly, u0, v0, d.shape, S)
            if k.get('type') in ROTATING:
                mk |= _disc(k['bbox_uv'], kdu - du, kdv - dv, u0, v0, d.shape, S)
            keep &= cv2.dilate(mk, np.ones((FOREIGN_DILATE, FOREIGN_DILATE), np.uint8)) == 0
        edge = 2 * FOREIGN_DILATE
        keep[:edge] = keep[-edge:] = False
        keep[:, :edge] = keep[:, -edge:] = False
        d = np.where(keep, d, 0)
        bw = (d > cfg['diff']).astype(np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
        for i in range(1, n):
            cx0, cy0, cw, ch, ar = st[i]
            if ar * S * S < cfg['min_area'] or ar < FILL_MIN * cw * ch:
                continue   # 太小、或者太细长（细长的是错位边缘，不是异物）
            sc = float(d[lab == i].mean())
            if sc < cfg['min_score']:
                continue
            bu = [u0 + cx0 * S, v0 + cy0 * S, u0 + (cx0 + cw) * S, v0 + (cy0 + ch) * S]
            xa, ya = amap.to_trip(bu[0] + du, bu[1] + dv)
            xb, yb = amap.to_trip(bu[2] + du, bu[3] + dv)
            out.append({'bbox_uv': [int(v) for v in bu],
                        'bbox_xy': [int(np.floor(xa)), int(np.floor(ya)), int(np.ceil(xb)), int(np.ceil(yb))],
                        'area': int(ar * S * S), 'score': round(sc, 2), 'src': 'diff'})
    # 分割出来、没对上任何零件的检出，落在父件框里的也算异物
    for e in extra or []:
        b = [int(v) for v in (e.get('bbox_xy') or [0, 0, 0, 0])]
        cxx, cyy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        uu, vv = amap.to_template(cxx, cyy)
        for q in tparts:
            if q.get('parent'):
                continue
            u0, v0, u1, v1 = q['bbox_uv']
            if not (u0 <= uu <= u1 and v0 <= vv <= v1):
                continue
            if any(_iom2(b, f['bbox_xy']) > 0.3 for f in out):
                break
            uu0, vv0 = amap.to_template(b[0], b[1])
            uu1, vv1 = amap.to_template(b[2], b[3])
            out.append({'bbox_uv': [int(uu0), int(vv0), int(uu1), int(vv1)], 'bbox_xy': b,
                        'area': int(max(1, (b[2] - b[0]) * (b[3] - b[1]))),
                        'score': round(float(e.get('score') or 0), 3), 'src': 'seg'})
            break
    return out


def _iom2(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / min(aa, bb)


# ---------------------------------------------------------------- 主流程

def run(args):
    trip = args.trip
    if args.suggest and trip != 'T01':
        raise SystemExit('--suggest 仅允许 T01 开发数据；T02 只做评测，不参与阈值调参')
    d = io.trip_dir(trip)
    pj = (io.ROOT / args.parts) if args.parts else (d / 'parts.json')
    if not pj.exists():
        raise SystemExit(f'没有 {io.rel(pj)}，先跑 railside match {trip}')
    partsj = io.read_json(pj)
    al_p = d / 'align.json'
    if not al_p.exists():
        raise SystemExit(f'没有 {io.rel(al_p)}，先跑 railside align {trip}')
    al = io.read_json(al_p)
    if not al.get('g'):
        raise SystemExit(f'{io.rel(al_p)} 里没有 g[]（对齐采样点），重跑 railside align {trip}')
    ver = resolve_ver(args.template or partsj.get('template_ver') or al.get('template_ver'))
    tpl = load_template(ver)
    tparts = tpl.get('parts') or []
    tmap = {q['code']: q for q in tparts}
    amap = AlignMap.from_json(al)
    types = load_thresholds(args.types)
    cal = io.read_json(d / 'calib.json') if (d / 'calib.json').exists() else tpl.get('calib')
    sx, sy = _sx_of(cal), (cal or {}).get('sy_mm_per_px')
    t_tpl = Trip.from_id(tpl['trip'])
    t_new = Trip.from_id(trip)
    inst = {i['code']: i for i in partsj.get('instances') or []}
    qc2 = [q for q in (al.get('qc2') or []) if q.get('flags')]
    cfg = {'diff': args.foreign_diff, 'min_area': args.foreign_min_area, 'min_score': args.foreign_min_score}

    items = []
    for code, it in inst.items():
        q = tmap.get(code)
        th = thresh_of(types, it.get('type'))
        if q is None:
            continue
        item = {'code': code, 'type': it['type'], 'status': 'ok', 'ncc': None, 'diff_area': None,
                'shift_mm': None, 'shift_vec_mm': None, 'dtheta_deg': None, 'alarms': []}
        if it['found_by'] == 'missing':
            item |= {'status': 'missing', 'alarms': [{'type': 'missing', 'value': 1, 'threshold': 0}]}
            items.append(item)
            continue
        cu = q['centroid_uv'][0]
        bad = next((f for f in qc2 if abs(f['u'] - cu) <= QC_HALF), None)
        r = measure(t_new, q, it, amap)
        if r is None:
            item |= {'status': 'skip', 'alarms': [{'type': 'qc', 'value': None, 'threshold': None,
                                                   'detail': '局部配准失败（零件图比搜索区还大，或位置离谱）'}]}
            items.append(item)
            continue
        item['ncc'] = r['ncc']
        item['dtheta_deg'] = r['dtheta']
        item['diff_area'] = round(float(r['s'] ** 2 / r['s_amap'] ** 2 - 1), 4)
        if it['found_by'] == 'seg' and it.get('area_px'):
            ref = float(q['area_px']) * max(r['s_amap'], 1e-6) ** 2
            if ref > 0:
                item['diff_area'] = round(float(it['area_px']) / ref - 1, 4)
        du, dv = offset_of(it, q, amap)
        pi = inst.get(q.get('parent') or '')
        if pi and pi['found_by'] != 'missing' and q['parent'] in tmap:
            # 子件报相对父件的位移：转向架整台相对车体本来就会动，那不是子件的毛病
            pdu, pdv = offset_of(pi, tmap[q['parent']], amap)
            du, dv = du - pdu, dv - pdv
        dxp, dyp = du * r['s_amap'], dv          # 模板 px → 本趟 px（横向乘局部比例，纵向不变）
        if sx and sy:
            vx, vy = dxp * sx(it['centroid_xy'][0]), dyp * sy
            item['shift_vec_mm'] = [round(vx, 2), round(vy, 2)]
            item['shift_mm'] = round(float(np.hypot(vx, vy)), 2)
        if bad is not None:
            item |= {'status': 'skip', 'alarms': [{'type': 'qc', 'value': bad.get('ncc'),
                                                   'threshold': th['ncc_min'],
                                                   'detail': 'qc2 ' + ','.join(bad['flags']) + f" @u={bad['u']}"}]}
            items.append(item)
            continue
        if item['ncc'] < th['ncc_min']:
            item['alarms'].append({'type': 'ncc', 'value': item['ncc'], 'threshold': th['ncc_min']})
        if item['shift_mm'] is not None and item['shift_mm'] > th['shift_mm_max']:
            item['alarms'].append({'type': 'shift', 'value': item['shift_mm'], 'threshold': th['shift_mm_max']})
        if abs(item['dtheta_deg']) > th['dtheta_max_deg']:
            item['alarms'].append({'type': 'dtheta', 'value': item['dtheta_deg'], 'threshold': th['dtheta_max_deg']})
        if abs(item['diff_area']) > th['area_change_max']:
            item['alarms'].append({'type': 'area', 'value': item['diff_area'], 'threshold': th['area_change_max']})
        item['status'] = 'alarm' if item['alarms'] else 'ok'
        items.append(item)

    foreign = find_foreign(t_tpl, t_new, tpl, tparts, inst, partsj.get('extra'), amap, cfg)
    out = io.ROOT / args.out if args.out else d / 'compare.json'
    io.write_json(out, {'trip': trip, 'template_ver': ver, 'items': items, 'foreign': foreign})
    cnt = defaultdict(int)
    for i in items:
        cnt[i['status']] += 1
        for a in i['alarms']:
            cnt['alarm_' + a['type']] += 1
    summary = {'step': 'compare', 'trip': trip, 'template': ver, 'items': len(items), 'status': dict(cnt),
               'alarms': sum(1 for i in items for _ in i['alarms']) + len(foreign), 'foreign': len(foreign),
               'mm': bool(sx and sy), 'out': io.rel(out)}
    if args.suggest:
        summary['suggest'] = suggest(items, args)
    return summary


# ---------------------------------------------------------------- 建议阈值

def _pct(vals, q):
    return float(np.percentile(np.asarray(vals, float), q)) if vals else None


def suggest(items, args):
    """按类型统计指标分布，给建议阈值；写 --suggest-out 下的 csv/json。"""
    if args.trip != 'T01':
        raise SystemExit('--suggest 仅允许 T01 开发数据；T02 只做评测，不参与阈值调参')
    outdir = io.ROOT / args.suggest_out
    outdir.mkdir(parents=True, exist_ok=True)
    q = args.quantile
    groups = defaultdict(list)
    for it in items:
        if it['status'] in ('ok', 'alarm'):
            groups[it['type']].append(it)
    stat_rows, sug_rows = [], []
    for typ in sorted(groups):
        g = groups[typ]
        ncc = [i['ncc'] for i in g if i['ncc'] is not None]
        sh = [i['shift_mm'] for i in g if i['shift_mm'] is not None]
        dt = [abs(i['dtheta_deg']) for i in g if i['dtheta_deg'] is not None]
        ar = [abs(i['diff_area']) for i in g if i['diff_area'] is not None]
        ncc_lo, sh_hi, dt_hi, ar_hi = _pct(ncc, 100 - q), _pct(sh, q), _pct(dt, q), _pct(ar, q)
        row = {'type': typ, 'n': len(g)}
        for name, arr in (('ncc', ncc), ('shift_mm', sh), ('dtheta_deg', dt), ('area', ar)):
            for p in (50, 90, 99):
                row[f'{name}_p{p}'] = round(_pct(arr, p), 4) if arr and len(arr) > 1 else (round(arr[0], 4) if arr else None)
            row[f'{name}_max'] = round(max(arr), 4) if arr else None
        stat_rows.append(row)
        sug_rows.append({
            'type': typ, 'n': len(g),
            'ncc_min': round(max(0.2, min(0.95, (ncc_lo or 0.5) - 0.05)), 3),
            'shift_mm_max': round(max(0.5, (sh_hi or 0.0) * args.relax), 2) if sh_hi is not None else None,
            'dtheta_max_deg': round(max(1.0, (dt_hi or 0.0) * args.relax), 2) if dt_hi is not None else None,
            'area_change_max': round(max(0.1, (ar_hi or 0.0) * args.relax), 3) if ar_hi is not None else None,
        })
    _write_csv(outdir / '指标统计.csv', stat_rows)
    _write_csv(outdir / '阈值建议.csv', sug_rows)
    io.write_json(outdir / '阈值建议.json', {'trip': args.trip, 'quantile': q, 'relax': args.relax,
                                            'defaults': DEFAULT, 'stats': stat_rows, 'suggest': sug_rows})
    return {'types': len(sug_rows), 'quantile': q, 'relax': args.relax,
            'out': io.rel(outdir / '阈值建议.csv')}


def _write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)