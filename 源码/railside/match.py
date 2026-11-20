"""S6b 身份匹配：把模板里的每个零件在新趟上找到，给出模板里同一个编码（设计.md §3 S6b、§4.1）。

三层，和 设计.md 一致：

都在“对齐后的图”上做：新趟按 S5 的映射重采样到模板坐标（aligned_region），横向比例和漂移
已由对齐吸收，所以每层只估平移，不搜比例（各向同性地搜比例会把纵向也缩放，3300 px 高的转向架
纵向缩 1.5% 就偏 50 px，真实 T02 上曾因此把子件整批带偏）。

1. 父件（没有父件的，如 BOGIE、EQUIP）：整件 NCC 估局部平移；
2. 子件：带上父件的平移，在小窗口里做像素匹配，并和 seg.json 里同类型的检出配对（重叠最大的那个）；
3. 都没找到的，按父件相对位置再找一次（窗口放大、门槛放低），还不行记 missing。

seg.json 没有时退回纯模板匹配，流程照跑。找到的位置写 数据/中间/<趟>/parts.json，
每个零件的局部图和掩膜写 数据/中间/<趟>/parts/<编码>.png、<编码>_mask.png。
"""
import cv2
import numpy as np

from . import io
from .coords import AlignMap, H, trip_to_raw
from .frames import Trip
from .template import crop_margin, load_template, resolve_ver

# 搜索范围：局部图尺度（px）
CHILD_FRAC, CHILD_MIN = 0.18, 18     # 子件搜索半径 = 包围框长边 × CHILD_FRAC，至少 CHILD_MIN
PARENT_FRAC, PARENT_MIN = 0.35, 60   # 第三层（按父件相对位置再找）半径
PARENT_REACH = 180                   # 父件第一层：预测框外扩
# 接受门槛（掩膜内真 NCC），只用 T01 定（实验/E4）：真匹配 = T01 原位加退化模拟第二次过车，
# 错位置 = 窗口平移到零件外取最高峰。T02 不参与。
ACCEPT_TOP = 0.55                    # 第一层父件：真匹配 126/126 ≥ 0.61；宁放不漏，父件真丢了子件会成批缺失
ACCEPT_PIXEL = 0.65                  # 第二层子件小窗：真匹配 100% 通过，错位置 0.5%
ACCEPT_PARENT = 0.70                 # 第三层大窗：窗口大、错位置峰更高，门槛反而比第二层高；真 99.9%，错 1.2%
SEG_IOMIN = 0.30                     # 和分割检出配对的交集/较小面积
COARSE_MAX = 900                     # 局部图长边超过这个值就降采样粗搜
CENTRAL = 512                        # 粗搜后全分辨率精修用的中心块边长


# ---------------------------------------------------------------- 工具

def read_region(t, x0, x1, y0, y1, step=1):
    """按趟坐标取区域，纵向夹到 [0, H)。返回 (图, 实际原点)。"""
    x0, x1 = int(x0), int(x1)
    y0, y1 = min(max(0, int(y0)), H - 1), min(H, int(y1))
    if y1 <= y0 or x1 <= x0:
        return np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8), (x0, y0)
    return t.region(x0, x1, y0, y1, step), (x0, y0)


def frames_of(x0, x1):
    """[x0, x1) 覆盖的帧号 [f0, f1]（帧号从 1 起，含两端）。"""
    f0 = int(trip_to_raw(int(x0))[0])
    f1 = int(trip_to_raw(max(int(x0), int(x1) - 1))[0])
    return [f0, f1]


def orientation_deg(xs, ys):
    """掩膜像素的主方向（度，0–180），和 template._orientation 同一算法。"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 5:
        return None
    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    _, v = np.linalg.eigh(c)
    vx, vy = v[:, 1]
    return round(float(np.degrees(np.arctan2(vy, vx))) % 180, 2)


def mask_stats(mask, origin=(0, 0)):
    """掩膜的质心（趟坐标）、面积（px）、主方向。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0, None
    return (float(xs.mean()) + origin[0], float(ys.mean()) + origin[1]), int(len(xs)), orientation_deg(xs, ys)


def poly_bbox(poly, shape=None):
    """多边形的整数外框，可夹到 (高, 宽) 里。"""
    p = np.asarray(poly, float)
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int)
    if shape is not None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1], x1), min(shape[0], y1)
    return [int(x0), int(y0), int(max(x1, x0)), int(max(y1, y0))]


def _resize(im, s):
    return _resize_xy(im, s, s)


def _resize_xy(im, sx, sy, interp=None):
    h, w = im.shape[:2]
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return im
    if interp is None:
        interp = cv2.INTER_AREA if min(sx, sy) < 1 else cv2.INTER_LINEAR
    return cv2.resize(im, (max(8, int(round(w * sx))), max(8, int(round(h * sy)))), interpolation=interp)


def _match_mask(mask):
    """匹配用的掩膜：零件掩膜外扩一圈（长边的 6%，3–25 px）。

    只比掩膜内：背景（软管、线缆、阴影）两趟不同也不影响打分。外扩一圈是为了留住轮廓：
    颜色均匀的零件（T 形标记、铸件、细杆）只看掩膜内是一块常数，NCC 无从算起。"""
    m = np.asarray(mask) > 0
    if not m.any() or m.all():
        return np.full(m.shape, 255, np.uint8)
    k = min(25, max(3, int(round(0.06 * max(m.shape))))) | 1
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) * 255


def _tpl_patch(q):
    """模板里这个零件的局部图和掩膜（全分辨率）。"""
    crop = io.imread(io.ROOT / q['crop'])
    if q.get('mask'):
        mask = io.imread(io.ROOT / q['mask'])
        if mask.shape != crop.shape:
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.full(crop.shape, 255, np.uint8)
    return crop, mask


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def aligned_region(t_new, amap, u0, u1, v0, v1, step=1):
    """新趟按 S5 的对齐映射重采样到模板坐标 [u0, u1) × [v0, v1)，step>1 时缩小 step 倍。

    这就是大纲说的“对齐后的图”：横向比例、沿车身的漂移、纵向小平移都由 align.json 的 g[] 吸收，
    在这张图上找零件只剩平移要估。像素 j 代表 u0 + j·step + (step−1)/2（和区域平均缩小一致）。
    """
    nu, nv = max(1, int((u1 - u0) // step)), max(1, int((v1 - v0) // step))
    half = (step - 1) / 2.0
    us = u0 + np.arange(nu) * step + half
    vs = v0 + np.arange(nv) * step + half
    xs = amap.u_to_x(us)
    dys = amap.dy_at_x(xs)
    x_lo = int(np.floor(xs.min())) - step
    x_hi = int(np.ceil(xs.max())) + 2 * step
    y_lo = min(max(0, int(np.floor(vs[0] + dys.min())) - step), H - 1)   # dy = 新趟 y − 模板 v
    y_hi = min(H, int(np.ceil(vs[-1] + dys.max())) + 2 * step)
    src = t_new.region(x_lo, x_hi, y_lo, y_hi, step)
    kx = (x_hi - x_lo) / src.shape[1]
    ky = (y_hi - y_lo) / src.shape[0]
    mapx = np.tile(((xs - x_lo - (kx - 1) / 2) / kx).astype(np.float32), (nv, 1))
    mapy = ((vs[:, None] + dys[None, :] - y_lo - (ky - 1) / 2) / ky).astype(np.float32)
    return cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _search(t_new, amap, crop, mask, org_uv, off, reach):
    """在对齐后的图上找模板局部图：预测位置是模板位置平移 off，在 ±reach（模板 px）里搜。

    crop、mask 是模板局部图，org_uv 是它左上角的模板坐标。只搜平移，比例已由对齐映射吸收。
    返回 {'du','dv','score'}：实测相对模板位置的平移（模板坐标 px）；找不到返回 None。
    大件先缩小粗搜（score 取粗搜的整件相似度），再在掩膜质心处取中心块全分辨率精修位置。
    """
    if not np.any(mask):          # 空掩膜没有可比的像素，不能当找到
        return None
    h, w = crop.shape
    tu0, tv0 = org_uv[0] + off[0], org_uv[1] + off[1]
    r0 = max(0, int(np.ceil(-tv0)))                  # 预测落到图像上下边外的行裁掉
    r1 = min(h, int(np.floor(H - tv0)))
    if r1 - r0 < 16 or w < 4:
        return None
    crop, mask = crop[r0:r1], mask[r0:r1]
    h, tv0, ov = r1 - r0, tv0 + r0, org_uv[1] + r0
    k = max(1, int(np.ceil(max(h, w) / COARSE_MAX)))
    reach = max(int(round(reach)), 2 * k)
    ru0, rv0 = int(np.floor(tu0)) - reach, max(0, int(np.floor(tv0)) - reach)
    ru1, rv1 = int(np.ceil(tu0)) + w + reach, min(H, int(np.ceil(tv0)) + h + reach)
    R = aligned_region(t_new, amap, ru0, ru1, rv0, rv1, k).astype(np.float32)
    tp = _resize(crop, 1 / k).astype(np.float32)
    tm = _match_mask(_resize_xy(mask, 1 / k, 1 / k, cv2.INTER_NEAREST))
    if tp.shape[0] > R.shape[0] or tp.shape[1] > R.shape[1]:
        return None
    if np.ptp(tp[tm > 0]) <= 1e-6:   # 纯色模板没有定位信息，CCOEFF 的退化值不能当作找到
        return None
    # 掩膜内的真 NCC。旧做法把掩膜外填成均值再整块相关，背景没法填，分数被压到 0.2–0.6、峰还会偏
    # （真实 T02：空气弹簧正确位置掩膜内 NCC 0.80，旧打分 0.19，判成没找到）
    res = cv2.matchTemplate(R, tp, cv2.TM_CCOEFF_NORMED, mask=tm)
    res = np.clip(np.where(np.isfinite(res), res, -1.0), -1.0, 1.0)
    _, score, _, (bx, by) = cv2.minMaxLoc(res)
    if k == 1 and 0 < bx < res.shape[1] - 1 and 0 < by < res.shape[0] - 1:
        fx = bx + _parabola(res[by, bx - 1], res[by, bx], res[by, bx + 1])
        fy = by + _parabola(res[by - 1, bx], res[by, bx], res[by + 1, bx])
    else:
        fx, fy = float(bx), float(by)
    du, dv = ru0 + fx * k - org_uv[0], rv0 + fy * k - ov
    if k > 1:
        c = _centroid(mask) or (w / 2, h / 2)
        bh, bw = min(CENTRAL, h), min(CENTRAL, w)
        yb = int(np.clip(c[1] - bh / 2, 0, h - bh))
        xb = int(np.clip(c[0] - bw / 2, 0, w - bw))
        f2 = _search(t_new, amap, crop[yb:yb + bh, xb:xb + bw], mask[yb:yb + bh, xb:xb + bw],
                     (org_uv[0] + xb, ov + yb), (du, dv), 2 * k + 4)
        if f2 and f2['score'] >= ACCEPT_TOP:      # 中心块小窗精修，窗口只有 ±(2k+4) px
            du, dv = f2['du'], f2['dv']
    return {'du': float(du), 'dv': float(dv), 'score': float(score)}


def _parabola(y0, y1, y2):
    d = float(y0) - 2 * float(y1) + float(y2)
    return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (float(y0) - float(y2)) / d, -0.5, 0.5))


def uv_poly_to_trip(amap, poly_uv, du=0.0, dv=0.0):
    """模板坐标多边形平移 (du, dv) 后，按对齐映射逐点换到本趟趟坐标。"""
    p = np.asarray(poly_uv, float)
    x, y = amap.to_trip(p[:, 0] + du, p[:, 1] + dv)
    return np.stack([np.asarray(x, float), np.asarray(y, float)], 1)


def _iom(a, b):
    """交集 / 较小面积。"""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(aa, ab)


# ---------------------------------------------------------------- 主流程

def add_parser(sub, step):
    p = sub.add_parser(step, help='S6b 身份匹配：模板零件 → 新趟同一编码')
    p.add_argument('trip')
    p.add_argument('--template', default=None, help='模板版本；默认取 align.json 的 template_ver')
    p.add_argument('--seg', default=None, help='分割结果；默认 数据/中间/<趟>/seg.json（没有就纯模板匹配）')
    p.add_argument('--out', default=None, help='默认 数据/中间/<趟>/parts.json')
    p.set_defaults(func=run)


def run(args):
    trip = args.trip
    d = io.trip_dir(trip)
    al_p = d / 'align.json'
    if not al_p.exists():
        raise SystemExit(f'没有 {io.rel(al_p)}，先跑 railside align {trip}')
    al = io.read_json(al_p)
    ver = resolve_ver(args.template or al.get('template_ver'))
    tpl = load_template(ver)
    tparts = tpl.get('parts') or []
    tmap = {q['code']: q for q in tparts}
    if not tparts:
        raise SystemExit(f'模板 {ver} 里没有零件：先在 标注/{tpl["trip"]}/零件.json 标好，'
                         f'再跑 railside template build --trip {tpl["trip"]} --ver {ver}')
    amap = AlignMap.from_json(al)
    seg_p = (io.ROOT / args.seg) if args.seg else (d / 'seg.json')
    seg = io.read_json(seg_p) if seg_p.exists() else None
    t_new = Trip.from_id(trip)

    # 1. 父件：在对齐后的图上估局部平移（模板坐标 px）
    pfound = {}
    for q in tparts:
        if q.get('parent'):
            continue
        crop, mask = _tpl_patch(q)
        pfound[q['code']] = _search(t_new, amap, crop, mask, q['crop_origin_uv'], (0.0, 0.0), PARENT_REACH)

    # 2. 子件：带上父件的平移，小窗口像素匹配；3. 找不到的窗口放大、门槛放低再找一次
    inst = []
    for q in tparts:
        code = q['code']
        cu, cv = q['centroid_uv']
        w = max(4.0, (q['bbox_uv'][2] - q['bbox_uv'][0]))
        h = max(4.0, (q['bbox_uv'][3] - q['bbox_uv'][1]))
        par = q.get('parent')
        pr = pfound.get(par) if par else None
        # 父件低分（随机峰）不提供位姿，子件退回整车对齐的预测位置
        off = (pr['du'], pr['dv']) if pr and pr['score'] >= ACCEPT_TOP else (0.0, 0.0)
        crop, mask = _tpl_patch(q)
        if par is None:   # 父件用第一层的结果，不再搜
            r = pfound.get(code)
        else:
            r = _search(t_new, amap, crop, mask, q['crop_origin_uv'], off, max(CHILD_MIN, CHILD_FRAC * max(w, h)))
        by = 'pixel' if (r and r['score'] >= (ACCEPT_PIXEL if par else ACCEPT_TOP)) else None
        if by is None:
            r3 = _search(t_new, amap, crop, mask, q['crop_origin_uv'], off, max(PARENT_MIN, PARENT_FRAC * max(w, h)))
            if r3 and r3['score'] >= ACCEPT_PARENT:
                r, by = r3, ('parent' if par else 'pixel')
        du, dv = (r['du'], r['dv']) if by else off
        item = {'code': code, 'type': q['type'], 'parent': par, 'car': q.get('car'), 'zone': q.get('zone'),
                'bbox_uv': list(q['bbox_uv']), 'offset_uv': [round(float(du), 2), round(float(dv), 2)]}
        poly = uv_poly_to_trip(amap, q['poly_uv'], du, dv)
        cx, cy = amap.to_trip(cu + du, cv + dv)
        c = (float(cx), float(cy))
        if by is None:
            b = poly_bbox(poly, (H, 10 ** 9))
            item |= {'bbox_xy': b, 'poly_xy': _round_poly(poly),
                     'centroid_xy': [round(c[0], 1), round(c[1], 1)], 'area_px': 0,
                     'angle_deg': q.get('angle_deg'), 'match_score': 0.0, 'found_by': 'missing',
                     'seg_score': None, 'crop': None, 'mask': None, 'src_frames': frames_of(b[0], b[2])}
        else:
            item |= {'bbox_xy': poly_bbox(poly), 'poly_xy': _round_poly(poly),
                     'centroid_xy': [round(c[0], 1), round(c[1], 1)],
                     'area_px': int(q['area_px']), 'angle_deg': q.get('angle_deg'),
                     'match_score': round(r['score'], 4), 'found_by': by, 'seg_score': None, '_c': c}
        inst.append(item)

    # 分割检出配对：同类、重叠最大，一个检出只配一个实例
    extra = []
    dets = (seg or {}).get('dets') or []
    used = [False] * len(dets)
    order = sorted(range(len(inst)), key=lambda i: -_area(inst[i]['bbox_xy']))
    for i in order:
        it = inst[i]
        best, bi = SEG_IOMIN, None
        for j, det in enumerate(dets):
            if used[j] or det.get('cls') != it['type']:
                continue
            v = _iom(it['bbox_xy'], det['bbox_xy'])
            if v > best:
                best, bi = v, j
        if bi is None:
            continue
        used[bi] = True
        det = dets[bi]
        poly = np.asarray(det['poly_xy'], float)
        if it['found_by'] == 'missing':
            it['match_score'] = round(float(det.get('score') or 0), 4)
        it['found_by'] = 'seg'
        it['seg_score'] = det.get('score')
        it['poly_xy'] = _round_poly(poly)
        it['bbox_xy'] = poly_bbox(poly)
        it['_c'] = _poly_center(poly)
        it['centroid_xy'] = [round(it['_c'][0], 1), round(it['_c'][1], 1)]
        cu_, cv_ = amap.to_template(*it['_c'])
        q_ = tmap[it['code']]['centroid_uv']
        it['offset_uv'] = [round(float(cu_) - q_[0], 2), round(float(cv_) - q_[1], 2)]
    for j, det in enumerate(dets):
        if not used[j]:
            extra.append({'cls': det.get('cls'), 'score': det.get('score'),
                          'bbox_xy': det.get('bbox_xy'), 'poly_xy': det.get('poly_xy')})

    # 局部图、掩膜、面积、质心、转角一律按本趟最终多边形重算
    for it in inst:
        it.pop('_c', None)
        if it['found_by'] == 'missing':
            continue
        poly = np.asarray(it['poly_xy'], float)
        mask, org = _write_part(t_new, d, it['code'], poly)
        c, area, ang = mask_stats(mask, org)
        it['crop'] = io.rel(d / 'parts' / f"{it['code']}.png")
        it['mask'] = io.rel(d / 'parts' / f"{it['code']}_mask.png")
        it['centroid_xy'] = [round(c[0], 1), round(c[1], 1)] if c else it['centroid_xy']
        it['area_px'] = area
        it['angle_deg'] = ang
        it['bbox_xy'] = poly_bbox(poly, (H, 10 ** 9))
        it['src_frames'] = frames_of(it['bbox_xy'][0], it['bbox_xy'][2])

    cnt = {k: sum(1 for i in inst if i['found_by'] == k) for k in ('seg', 'pixel', 'parent', 'missing')}
    out = io.ROOT / args.out if args.out else d / 'parts.json'
    io.write_json(out, {'trip': trip, 'template_ver': ver, 'instances': inst, 'extra': extra})
    return {'step': 'match', 'trip': trip, 'template': ver, 'instances': len(inst), 'found': len(inst) - cnt['missing'],
            'by': cnt, 'extra': len(extra), 'seg': bool(seg), 'out': io.rel(out)}


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _round_poly(poly):
    return [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(poly, float)]


def _poly_center(poly):
    p = np.asarray(poly, float)
    m = np.zeros((int(p[:, 1].max() - p[:, 1].min()) + 3, int(p[:, 0].max() - p[:, 0].min()) + 3), np.uint8)
    cv2.fillPoly(m, [np.round(p - [p[:, 0].min() - 1, p[:, 1].min() - 1]).astype(np.int32)], 255)
    c, _, _ = mask_stats(m, (p[:, 0].min() - 1, p[:, 1].min() - 1))
    return c or (float(p[:, 0].mean()), float(p[:, 1].mean()))


def _write_part(t_new, d, code, poly):
    """写局部图和掩膜，返回 (掩膜, 掩膜左上角的趟坐标)。"""
    b = poly_bbox(poly, (H, 10 ** 9))
    mx, my = crop_margin(b[2] - b[0], b[3] - b[1])
    cx0, cy0 = b[0] - mx, max(0, b[1] - my)
    cx1, cy1 = b[2] + mx, min(H, b[3] + my)
    crop = t_new.region(cx0, cx1, cy0, cy1)
    mask = np.zeros(crop.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(poly, float) - [cx0, cy0]).astype(np.int32)], 255)
    io.imwrite(d / 'parts' / f'{code}.png', crop)
    io.imwrite(d / 'parts' / f'{code}_mask.png', mask)
    return mask, (cx0, cy0)
