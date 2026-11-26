"""S5 对齐：新趟 → 模板坐标；顺带做质检第二段（逐块和模板比）。

1. 定位靶：按 u 顺序在预测窗口里找（横向 ±1500 px，纵向 ±60 px），预测用上一个成功靶的偏移。
2. 加密：模板车内每 1000 px 取一块 1024 宽的竖条（行 600–3800），在新趟上找对应位置。
3. 稳健拟合：偏移 o = x_new − x_tpl 沿车身做滑动中值，剔除离群点，得到单调映射 u = g(x)。
4. 裁剪：u = 0 是模板车头；新趟车头前的空白段裁掉、按模板坐标重编号。车头保护：1 号车车头段的靶必须找到，否则报警，不裁。
5. 质检第二段：每块的 NCC（低 → 遮挡或变化）、清晰度比（新/模板，低 → 模糊）。
"""
import cv2
import numpy as np

from . import anchors as anc
from . import io
from .coords import AlignMap
from .frames import Trip
from .qc import blur_r
from .template import load_template

BLOCK = 1024
EVERY = 1000
BAND = (600, 3800)
REACH = 200      # 加密块相对靶插值预测的搜索半径（px）


def add_parser(sub, step):
    p = sub.add_parser(step, help='S5 对齐到模板坐标 + 质检第二段')
    p.add_argument('trip')
    p.add_argument('--template', default=None, help='模板版本；默认 数据/成品/模板 下最新的')
    p.set_defaults(func=run)


def match_anchors(t_new, t_tpl, tpl, x_offset0):
    head_t = tpl['head_x']
    off = x_offset0
    out = []
    for a in sorted(tpl['anchors'], key=lambda a: a['u']):
        ax = a['u'] + head_t
        anchor = {'x': ax, 'y': a['v'], 'w': a['w'], 'h': a['h']}
        x, y, ncc, ratio = anc.locate(t_new, t_tpl, anchor, ax + off, a['v'])
        ok = ncc > 0.6 and ratio > 0.05
        if ok:
            off = x - ax
        out.append({'id': a['id'], 'car': a['car'], 'u': a['u'], 'v': a['v'], 'x_tpl': ax,
                    'x': round(x, 2), 'y': round(y, 2), 'ncc': round(ncc, 4), 'peak_margin': round(ratio, 4), 'ok': ok})
    # 离群剔除：偏移与前后成功靶线性插值的差，超过 60 px + 间距的 3% 就剔除
    good = [r for r in out if r['ok']]
    xs = np.array([r['x_tpl'] for r in good], float)
    offs = np.array([r['x'] - r['x_tpl'] for r in good])
    for i, r in enumerate(good):
        if 0 < i < len(good) - 1:
            pred = np.interp(xs[i], [xs[i - 1], xs[i + 1]], [offs[i - 1], offs[i + 1]])
            tol = 60 + 0.03 * min(xs[i] - xs[i - 1], xs[i + 1] - xs[i])
            if abs(offs[i] - pred) > tol:
                r['ok'] = False
                r['reject'] = 'outlier'
    return out


def dense(t_new, t_tpl, x1s, pred):
    """对模板车内各块，在新趟上找对应位置。返回 [(x_tpl 块中心, x_new 块中心, dy, ncc)]。"""
    h = BLOCK // 2
    out = []
    for x1 in x1s:
        x2p = int(round(pred(x1)))
        tp = t_tpl.region(x1 - h, x1 + h, *BAND, step=4).astype(np.float32)
        reg = t_new.region(x2p - h - REACH, x2p + h + REACH, BAND[0] - 40, BAND[1] + 40, step=4).astype(np.float32)
        if np.ptp(tp) <= 1e-6 or np.ptp(reg) <= 1e-6:
            out.append((x1, float(x2p), 0.0, -1.0))
            continue
        r = cv2.matchTemplate(reg, tp, cv2.TM_CCOEFF_NORMED)
        r = np.where(np.isfinite(r), r, -1.0)
        _, v, _, (bx, by) = cv2.minMaxLoc(r)
        xc = x2p - h - REACH + bx * 4
        dyc = by * 4 - 40
        # 全分辨率精修（行隔 4 取 1）
        k = 6
        tp1 = t_tpl.region(x1 - h, x1 + h, *BAND)[::4].astype(np.float32)
        reg1 = t_new.region(xc - k, xc + BLOCK + k, BAND[0] + dyc, BAND[1] + dyc)[::4].astype(np.float32)
        if np.ptp(tp1) <= 1e-6 or np.ptp(reg1) <= 1e-6:
            out.append((x1, float(x2p), float(dyc), -1.0))
            continue
        r1 = cv2.matchTemplate(reg1, tp1, cv2.TM_CCOEFF_NORMED)[0]
        r1 = np.where(np.isfinite(r1), r1, -1.0)
        b1 = int(r1.argmax())
        sub = anc._parabola(r1[b1 - 1:b1 + 2]) if 0 < b1 < len(r1) - 1 else 0.0
        x2 = xc - k + b1 + sub + h
        out.append((x1, float(x2), float(dyc), float(r1[b1])))
    return out


def refine(t_new, t_tpl, x1s, amap, head_t, dys=None, reach=40):
    """第二遍：按第一遍映射的局部比例缩放模板块，再在 ±reach 内精配。消除比例变化带来的偏差。"""
    h = BLOCK // 2
    out = []
    for i, x1 in enumerate(x1s):
        u = x1 - head_t
        s = float(amap.scale_at_u(u))
        x2p = float(amap.u_to_x(u))
        dy = float(amap.dy_at_x(x2p)) if dys is None else float(dys[i])
        w2 = max(16, int(round(BLOCK * s)))
        tp = t_tpl.region(x1 - h, x1 + h, *BAND)[::4].astype(np.float32)
        tp = cv2.resize(tp, (w2, tp.shape[0]), interpolation=cv2.INTER_LINEAR)
        r0 = int(round(x2p - w2 / 2)) - reach
        reg = t_new.region(r0, r0 + w2 + 2 * reach, BAND[0] + int(round(dy)), BAND[1] + int(round(dy)))[::4].astype(np.float32)
        if np.ptp(tp) <= 1e-6 or np.ptp(reg) <= 1e-6:
            out.append((x1, float(x2p), float(dy), -1.0))
            continue
        r = cv2.matchTemplate(reg, tp, cv2.TM_CCOEFF_NORMED)[0]
        r = np.where(np.isfinite(r), r, -1.0)
        b = int(r.argmax())
        sub = anc._parabola(r[b - 1:b + 2]) if 0 < b < len(r) - 1 else 0.0
        out.append((x1, r0 + b + sub + w2 / 2, float(dy), float(r[b])))
    return out


def robust_fit(x1, x2, w=9, tol=12.0):
    """偏移 o = x2 − x1 做滑动中值；离群点（偏离中值 > tol px）去掉。返回保留的下标。"""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    o = x2 - x1
    keep = np.isfinite(x1) & np.isfinite(x2)
    for _ in range(2):
        idx = np.where(keep)[0]
        if len(idx) == 0:
            return keep
        med = np.array([np.median(o[idx[max(0, i - w // 2): i + w // 2 + 1]]) for i in range(len(idx))])
        bad = np.abs(o[idx] - med) > tol
        keep[idx[bad]] = False
    # 保证单调
    idx = np.where(keep)[0]
    last = -np.inf
    for i in idx:
        if x2[i] <= last:
            keep[i] = False
        else:
            last = x2[i]
    return keep


def run(args):
    tpl = load_template(args.template)
    head_t = tpl['head_x']
    t_tpl = Trip.from_id(tpl['trip'])
    t_new = Trip.from_id(args.trip)
    m_new = io.read_json(io.trip_dir(args.trip) / 'manifest.json')
    m_tpl = io.read_json(io.trip_dir(tpl['trip']) / 'manifest.json')
    off0 = m_new['head_x'] - head_t

    am = match_anchors(t_new, t_tpl, tpl, off0)
    good = [a for a in am if a['ok']]
    if len(good) < 4:
        raise SystemExit(f'定位靶只找到 {len(good)} 个，无法对齐')
    ax1 = np.array([a['x_tpl'] for a in good], float)
    ax2 = np.array([a['x'] for a in good], float)

    def pred(x):
        return float(np.interp(x, ax1, ax2, left=ax2[0] + (x - ax1[0]), right=ax2[-1] + (x - ax1[-1])))

    x1s = list(range(m_tpl['train_x0'] + BLOCK // 2, m_tpl['train_x1'] - BLOCK // 2, EVERY))
    blocks = dense(t_new, t_tpl, x1s, pred)
    bx1 = np.array([b[0] for b in blocks], float)
    bx2 = np.array([b[1] for b in blocks], float)
    bdy = np.array([b[2] for b in blocks], float)
    bncc = np.array([b[3] for b in blocks], float)
    use = np.isfinite(bncc) & np.isfinite(bx2) & np.isfinite(bdy) & (bncc > 0.5)
    keep = np.zeros(len(blocks), bool)
    if int(use.sum()) >= 2:
        ki = robust_fit(bx1[use], bx2[use])
        keep[np.where(use)[0][ki]] = True
    if int(keep.sum()) < 2:
        raise SystemExit(f'对齐第一遍有效加密块不足（仅 {int(keep.sum())} 块，至少需要 2 块），无法拟合映射')
    amap1 = AlignMap(bx2[keep], bx1[keep] - head_t, bdy[keep])
    # 第二遍：使用第一遍稳健拟合后的纵向偏移 amap1.dy_at_x，避免离群块 dy 污染精配
    dys2 = amap1.dy_at_x(amap1.u_to_x(bx1 - head_t))
    blocks = refine(t_new, t_tpl, x1s, amap1, head_t, dys2)
    bx2 = np.array([b[1] for b in blocks], float)
    bdy2 = np.array([b[2] for b in blocks], float)
    bncc = np.array([b[3] for b in blocks], float)
    use = np.isfinite(bncc) & np.isfinite(bx2) & np.isfinite(bdy2) & (bncc > 0.6)
    keep = np.zeros(len(blocks), bool)
    if int(use.sum()) >= 2:
        ki = robust_fit(bx1[use], bx2[use], w=7, tol=8.0)
        keep[np.where(use)[0][ki]] = True
    if int(keep.sum()) < 2:
        raise SystemExit(f'对齐第二遍有效加密块不足（仅 {int(keep.sum())} 块，至少需要 2 块），无法拟合映射')
    amap = AlignMap(bx2[keep], bx1[keep] - head_t, bdy2[keep])

    # 残差：用加密块建的映射去预测各定位靶（靶没参与建映射）
    res = np.array([amap.x_to_u(a['x']) - a['u'] for a in good])
    for a in am:
        a['u_fit_residual'] = round(float(amap.x_to_u(a['x']) - a['u']), 2) if a['ok'] else None

    # 裁剪与车头保护
    car1 = [a for a in am if a['car'] == 1]
    head_anchor = min(car1, key=lambda a: a['u']) if car1 else None
    head_ok = bool(head_anchor and head_anchor['ok'])
    x_trim = float(amap.u_to_x(0))
    if head_ok and x_trim > head_anchor['x']:
        x_trim = head_anchor['x'] - 1000
    if not head_ok:
        x_trim = float(m_new['head_x'])

    # 质检第二段：逐块比
    qc2 = []
    h = BLOCK // 2
    for (x1, x2, dy, ncc), k in zip(blocks, keep):
        a = t_tpl.region(x1 - h, x1 + h, 800, 3600, step=2)
        b = t_new.region(int(x2) - h, int(x2) + h, 800 + int(dy), 3600 + int(dy), step=2)
        br = blur_r(b) / max(blur_r(a), 1e-6)
        fl = []
        if ncc < 0.5:
            fl.append('mismatch')
        if br < 0.6:
            fl.append('blur')
        qc2.append({'u': round(x1 - head_t), 'x': round(x2, 1), 'ncc': round(ncc, 4), 'blur_ratio': round(br, 3),
                    'used': bool(k), 'flags': fl})
    flagged = [q for q in qc2 if q['flags']]

    g = [{'x': round(float(x), 2), 'u': round(float(u), 2), 'dy': round(float(d), 1)}
         for x, u, d in zip(amap.xs, amap.us, amap.dys)]
    scale = amap.scale_at_u(np.array([p['u'] for p in g]))
    out = {
        'trip': args.trip, 'template_ver': tpl['version'], 'template_trip': tpl['trip'],
        'anchors': [{'id': a['id'], 'car': a['car'], 'x': a['x'], 'y': a['y'], 'u': a['u'], 'v': a['v'],
                     'score': a['ncc'], 'ok': a['ok'], 'u_fit_residual': a['u_fit_residual'], **({'reject': a['reject']} if 'reject' in a else {})}
                    for a in am],
        'anchors_found': len(good), 'anchors_total': len(am),
        'g': g, 'blocks_total': len(blocks), 'blocks_used': int(keep.sum()),
        'residual_px': {'median': round(float(np.median(np.abs(res))), 2), 'p95': round(float(np.percentile(np.abs(res), 95)), 2),
                        'max': round(float(np.abs(res).max()), 2), 'n': int(len(res))},
        'scale_range': [round(float(scale.min()), 4), round(float(scale.max()), 4)],
        'offset_range_px': [round(float((amap.xs - amap.us - head_t).min()), 1), round(float((amap.xs - amap.us - head_t).max()), 1)],
        'head_protect': {'anchor': head_anchor['id'] if head_anchor else None, 'found': head_ok},
        'x_trim': round(x_trim, 1), 'u_trim0': 0,
        'qc2': qc2, 'qc2_flagged': len(flagged),
    }
    io.write_json(io.trip_dir(args.trip) / 'align.json', out)
    _overlay(args.trip, t_tpl, t_new, m_tpl, head_t, amap)
    return {'step': 'align', 'trip': args.trip, 'anchors': f"{len(good)}/{len(am)}", 'blocks_used': int(keep.sum()),
            'residual_px': out['residual_px'], 'scale_range': out['scale_range'], 'offset_range_px': out['offset_range_px'],
            'head_protect': head_ok, 'qc2_flagged': len(flagged), 'out': io.rel(io.trip_dir(args.trip) / 'align.json')}


def _overlay(trip, t_tpl, t_new, m_tpl, head_t, amap, S=16):
    """对齐检查图：每节车一行，模板为绿、对齐后新趟为品红，重合处呈灰。翻转显示。"""
    rows = []
    for p in m_tpl['train_panels']:
        a = t_tpl.region(p['x0'], p['x1'], step=S).astype(np.float32)
        us = (np.arange(a.shape[1]) * S + p['x0'] - head_t).astype(float)
        xs = amap.u_to_x(us)
        lo, hi = int(xs.min()) - S, int(xs.max()) + 2 * S
        b_strip = t_new.region(lo, hi, step=S).astype(np.float32)
        mapx = ((xs - lo) / S).astype(np.float32)[None, :].repeat(a.shape[0], 0)
        mapy = (np.arange(a.shape[0], dtype=np.float32)[:, None]
                + np.asarray(amap.dy_at_x(xs), np.float32)[None, :] / S)
        b = cv2.remap(b_strip, mapx, mapy, cv2.INTER_LINEAR)
        rgb = np.dstack([b, a, b]).astype(np.uint8)[:, ::-1]
        rows.append(rgb)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (wmax - r.shape[1], 0), (0, 0))) for r in rows]
    io.imwrite(io.trip_dir(trip) / '预览' / '对齐叠加_1-16.jpg', np.vstack(rows), quality=88)

