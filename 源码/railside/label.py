"""S4 零件预标：按提示跑 SAM 出多边形、把样板转向架搬到同型各台、出核对视图。

  railside label view --box X0 Y0 X1 Y1 [--step N] [--grid G] [--out P]
        带趟坐标网格的局部图（不翻转），读图定提示用
  railside label propagate --sample C3B1 --to C1B1 C2B1 ...
        样板转向架的提示按 calib.json 的轴端中心整台映射，再逐件局部 NCC 精修，写回 预标提示.json
  railside label sam [--ids ...] [--bogie ...] [--all]
        按 预标提示.json 跑 SAM2，生成 标注/模板T01/零件.json（只重算选中的，其余保留）
  railside label review
        每台转向架、每节车车下各一张核对图（翻转显示，车头在右）和计数表

预标提示.json 是可复现的输入：提示（类型、框、点）都用趟坐标；零件.json 由脚本从提示和原图生成。
编号：转向架序号 b = (车 − 1)·2 + 区（B1=1、B2=2），转向架上的零件 id = 1000·b + n，
搬运时保留 n；车下（BODY）零件 id = 20000 + 100·车 + n。
"""
import csv
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

from . import io
from .coords import H
from .frames import Trip

TRIP = 'T01'
SMALL = 480            # 长边不超过它的提示合并进 1024 全分辨率窗口一起跑
WIN = 1024
NCC_MIN = 0.55         # 搬运精修的 NCC 低于它不放，进复核清单
NCC_BIG_MIN = 0.45     # 大件（非螺栓）放宽到这里，放进来但标 low_ncc
MAX_PTS = {True: 16, False: 40}   # 小件 / 大件多边形点数上限
COLORS = {'BOGIE': (255, 255, 255), 'WHEEL': (255, 160, 0), 'AXLEBOX': (0, 200, 255), 'AXLE_END': (0, 255, 255),
          'SPRING1': (0, 255, 0), 'AIRSPRING': (255, 0, 255), 'DAMPER': (255, 255, 0), 'LINK': (0, 140, 255),
          'BOLT': (60, 60, 255), 'EQUIP': (200, 120, 255), 'MARK': (120, 255, 200)}


def label_dir():
    return io.ROOT / '标注' / f'模板{TRIP}'


def prompts_path():
    return label_dir() / '预标提示.json'


def parts_path():
    return label_dir() / '零件.json'


def review_dir():
    return io.trip_dir(TRIP) / '预览' / '标注核对'


def bogie_no(car, zone):
    return (car - 1) * 2 + (1 if zone == 'B1' else 2)


def bogie_key(car, zone):
    return f'C{car}{zone}'


def parse_bogie(key):
    return int(key[1]), key[2:]


def load_prompts():
    p = prompts_path()
    return io.read_json(p) if p.exists() else {'trip': TRIP, 'prompts': []}


def save_prompts(d):
    d['prompts'] = sorted(d['prompts'], key=lambda q: q['id'])
    io.write_json(prompts_path(), {k: v for k, v in d.items() if k not in ('code_ver', 'created_at')})


# ---------------------------------------------------------------- 几何

def box_of(poly):
    a = np.asarray(poly, float)
    return [int(np.floor(a[:, 0].min())), int(np.floor(a[:, 1].min())),
            int(np.ceil(a[:, 0].max())) + 1, int(np.ceil(a[:, 1].max())) + 1]


def rect_poly(b):
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def ellipse_poly(b, n=16):
    x0, y0, x1, y1 = b
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[round(cx + rx * np.cos(t), 1), round(cy + ry * np.sin(t), 1)] for t in a]


def mask_to_poly(mask, origin, max_pts):
    """二值掩膜 → 最大连通块的外轮廓（趟坐标），简化到不超过 max_pts 个点。没有像素返回 None。"""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 9:
        return None
    peri = cv2.arcLength(c, True)
    eps = 0.003 * peri
    a = cv2.approxPolyDP(c, eps, True)
    while len(a) > max_pts:
        eps *= 1.3
        a = cv2.approxPolyDP(c, eps, True)
    ox, oy = origin
    return [[float(p[0][0] + ox), float(p[0][1] + oy)] for p in a]


def poly_mask(poly, origin, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly, float) - origin).astype(np.int32)], 1)
    return m


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


# ---------------------------------------------------------------- view

def cmd_view(args):
    x0, y0, x1, y1 = args.box
    step = args.step or max(1, int(np.ceil(max(x1 - x0, y1 - y0) / 1600)))
    t = Trip.from_id(TRIP)
    im = cv2.cvtColor(t.region(x0, x1, max(0, y0), min(H, y1), step=step), cv2.COLOR_GRAY2BGR)
    y0 = max(0, y0)
    g = args.grid
    h, w = im.shape[:2]
    for gx in range((x0 // g + 1) * g, x1, g):
        c = (gx - x0) // step
        major = gx % (g * 5) == 0
        cv2.line(im, (c, 0), (c, h), (0, 200, 255) if major else (0, 110, 150), 1)
        if major:
            cv2.putText(im, str(gx), (c + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for gy in range((y0 // g + 1) * g, y1, g):
        r = (gy - y0) // step
        major = gy % (g * 5) == 0
        cv2.line(im, (0, r), (w, r), (0, 200, 255) if major else (0, 110, 150), 1)
        if major:
            cv2.putText(im, str(gy), (2, r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if args.overlay and parts_path().exists():
        for q in io.read_json(parts_path())['parts']:
            b = q['bbox']
            if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            pts = np.round((np.asarray(q['poly'], float) - [x0, y0]) / step).astype(np.int32)
            col = COLORS.get(q['type'], (255, 255, 255))
            cv2.polylines(im, [pts], True, col, 1)
            cv2.putText(im, str(q['id'] % 1000), (int(pts[:, 0].min()), int(pts[:, 1].min()) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    out = args.out or (io.trip_dir(TRIP) / '预览' / '标注工作' / f'view_{x0}_{y0}_{step}.jpg')
    out = io.imwrite(out, im, quality=90)
    return {'step': 'label view', 'out': io.rel(out), 'shape': im.shape[:2], 'scale': step}


# ---------------------------------------------------------------- SAM

_PRED = None


def predictor():
    global _PRED
    if _PRED is None:
        from ultralytics.models.sam import SAM2Predictor
        w = io.ROOT / 'models' / 'sam' / 'sam2.1_b.pt'
        _PRED = SAM2Predictor(overrides=dict(conf=0.25, task='segment', mode='predict', imgsz=1024,
                                             model=str(w if w.exists() else 'sam2.1_b.pt'), verbose=False, save=False))
    return _PRED


def plan_windows(prompts):
    """把提示分到 SAM 窗口：小件按 1024 全分辨率窗口成组，大件各自一个窗口（框外扩后整体缩到 1024）。

    返回 [(x0, y0, x1, y1, [提示])]。"""
    wins = []
    small, big = [], []
    for q in prompts:
        b = q['box']
        (small if max(b[2] - b[0], b[3] - b[1]) <= SMALL else big).append(q)
    for q in big:
        x0, y0, x1, y1 = q['box']
        mg = int(0.15 * max(x1 - x0, y1 - y0)) + 32
        wins.append((x0 - mg, max(0, y0 - mg), x1 + mg, min(H, y1 + mg), [q]))
    groups = []   # [外接框, [提示]]，按 x 扫描，能塞进已有组（外接框 ≤ WIN−128）就塞
    for q in sorted(small, key=lambda q: (q['box'][0], q['box'][1])):
        b = q['box']
        for g in groups:
            ub = g[0]
            nb = [min(ub[0], b[0]), min(ub[1], b[1]), max(ub[2], b[2]), max(ub[3], b[3])]
            if nb[2] - nb[0] <= WIN - 128 and nb[3] - nb[1] <= WIN - 128:
                g[0] = nb
                g[1].append(q)
                break
        else:
            groups.append([list(b), [q]])
    for ub, qs in groups:
        cx, cy = (ub[0] + ub[2]) // 2, (ub[1] + ub[3]) // 2
        y0 = min(max(0, cy - WIN // 2), H - WIN)
        wins.append((cx - WIN // 2, y0, cx + WIN // 2, y0 + WIN, qs))
    return wins


def sam_window(t, win):
    """一个窗口编码一次，逐个提示出掩膜。返回 {提示 id: (多边形, SAM 分数, 掩膜占框比)}。"""
    x0, y0, x1, y1, qs = win
    im = cv2.cvtColor(t.region(x0, x1, y0, y1), cv2.COLOR_GRAY2BGR)
    p = predictor()
    p.set_image(im)
    out = {}
    for q in qs:
        b = q['box']
        bb = [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]
        kw = {'bboxes': [bb]}
        pts = [(pt, 1) for pt in q.get('pos', [])] + [(pt, 0) for pt in q.get('neg', [])]
        if pts:
            kw['points'] = [[[pt[0] - x0, pt[1] - y0] for pt, _ in pts]]
            kw['labels'] = [[lab for _, lab in pts]]
        r = p(**kw)[0]
        if r.masks is None or not len(r.masks.data):
            out[q['id']] = (None, 0.0, 0.0)
            continue
        m = r.masks.data[0].cpu().numpy().astype(np.uint8)
        # 掩膜限制在提示框外扩 6% 内，防止漏到背景
        mx, my = int(0.06 * (bb[2] - bb[0])) + 4, int(0.06 * (bb[3] - bb[1])) + 4
        keep = np.zeros_like(m)
        keep[max(0, bb[1] - my):bb[3] + my, max(0, bb[0] - mx):bb[2] + mx] = 1
        m &= keep
        score = float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes.conf) else 0.0
        fill = float(m.sum()) / max(1, (bb[2] - bb[0]) * (bb[3] - bb[1]))
        small = max(b[2] - b[0], b[3] - b[1]) <= SMALL
        out[q['id']] = (mask_to_poly(m, (x0, y0), MAX_PTS[small]), round(score, 3), round(fill, 3))
    return out


def make_part(q, res):
    """提示 + SAM 结果 → 零件.json 的一条。mode=box/ellipse 的直接用框，SAM 失败也退回框。"""
    mode = q.get('mode', 'sam')
    poly, score, fill = res if res else (None, None, None)
    flags = []
    if mode == 'box':
        poly = rect_poly(q['box'])
    elif mode == 'ellipse':
        poly = ellipse_poly(q['box'])
    elif poly is None:
        poly, flags = rect_poly(q['box']), ['sam_empty']
    elif fill is not None and fill < 0.12:
        flags.append('sam_small')
    if q.get('flags'):
        flags += q['flags']
    rec = {'id': q['id'], 'type': q['type'], 'poly': [[round(x, 1), round(y, 1)] for x, y in poly],
           'bbox': box_of(poly), 'parent': None, 'source': q.get('source', 'ai'), 'verified': False,
           'bogie': q.get('bogie'), 'mode': mode}
    if score is not None and mode == 'sam':
        rec['sam_score'] = score
        rec['fill'] = fill
    if q.get('ncc') is not None:
        rec['ncc'] = q['ncc']
    if q.get('from') is not None:
        rec['from'] = q['from']
    if flags:
        rec['flags'] = flags
    return rec


def cmd_sam(args):
    d = load_prompts()
    sel = d['prompts']
    if not args.all:
        ids = set(args.ids or [])
        bg = set(args.bogie or [])
        typ = set(args.type or [])
        sel = [q for q in sel if q['id'] in ids or q.get('bogie') in bg or q['type'] in typ]
    old = io.read_json(parts_path())['parts'] if parts_path().exists() else []
    keep = {q['id'] for q in d['prompts']}
    done = {q['id'] for q in sel}
    parts = {p['id']: p for p in old if p['id'] in keep and p['id'] not in done}
    t = Trip.from_id(TRIP)
    need = [q for q in sel if q.get('mode', 'sam') == 'sam']
    wins = plan_windows(need)
    t0 = time.time()
    res = {}
    for k, w in enumerate(wins, 1):
        res.update(sam_window(t, w))
        if k % 10 == 0:
            print(f'SAM 窗口 {k}/{len(wins)}，{time.time() - t0:.0f} s', flush=True)
    for q in sel:
        parts[q['id']] = make_part(q, res.get(q['id']))
    write_parts(list(parts.values()))
    flagged = [p['id'] for p in parts.values() if p.get('flags')]
    return {'step': 'label sam', 'selected': len(sel), 'windows': len(wins), 'seconds': round(time.time() - t0, 1),
            'parts': len(parts), 'flagged': len(flagged), 'out': io.rel(parts_path())}


def write_parts(parts):
    parts = sorted(parts, key=lambda p: p['id'])
    io.write_json(parts_path(), {'trip': TRIP, 'parts': parts,
                                 'note': '由 railside label sam 从 预标提示.json 生成；verified 全为 false，等人核'})


# ---------------------------------------------------------------- 搬运

def bogie_axles(calib):
    """calib.json → {C<车><区>: (左轴端中心, 右轴端中心)}，趟坐标。"""
    return {bogie_key(b['car'], b['zone']): (tuple(b['left']), tuple(b['right'])) for b in calib['bogies']}


def axle_map(src, dst):
    """两台转向架的轴端中心 → 映射 (sx, ox, oy)：x' = ox + sx·x，y' = y + oy。"""
    (sl, sr), (dl, dr) = src, dst
    sx = (dr[0] - dl[0]) / (sr[0] - sl[0])
    ox = dl[0] - sx * sl[0]
    oy = (dl[1] + dr[1]) / 2 - (sl[1] + sr[1]) / 2
    return sx, ox, oy


def map_box(b, mp):
    sx, ox, oy = mp
    return [ox + sx * b[0], b[1] + oy, ox + sx * b[2], b[3] + oy]


def ncc_refine(t, src_box, pred_box, sx, reach, step, context=8):
    """样板上 src_box 的图块（横向缩放 sx）在目标 pred_box 周围 ±reach 内找最佳位置。

    返回 (dx, dy, ncc)，dx、dy 是相对预测框的平移（趟坐标 px）。"""
    x0, y0, x1, y1 = [int(round(v)) for v in src_box]
    mg = max(context, int(0.12 * max(x1 - x0, y1 - y0)))
    a0, b0, a1, b1 = x0 - mg, max(0, y0 - mg), x1 + mg, min(H, y1 + mg)
    tpl = t.region(a0, a1, b0, b1, step=step).astype(np.float32)
    tw = max(8, int(round(tpl.shape[1] * sx)))
    tpl = cv2.resize(tpl, (tw, tpl.shape[0]), interpolation=cv2.INTER_AREA)
    px0 = pred_box[0] - mg * sx
    py0 = pred_box[1] - (y0 - b0)
    r = reach
    c0, d0 = int(round(px0 - r)), int(round(max(0, py0 - r)))
    c1, d1 = int(round(px0 + tw * step + r)), int(round(min(H, py0 + tpl.shape[0] * step + r)))
    reg = t.region(c0, c1, d0, d1, step=step).astype(np.float32)
    if reg.shape[0] < tpl.shape[0] or reg.shape[1] < tpl.shape[1] or tpl.std() < 1:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
    _, v, _, (mx, my) = cv2.minMaxLoc(res)
    return c0 + mx * step - px0, d0 + my * step - py0, float(v)


def cmd_propagate(args):
    d = load_prompts()
    calib = io.read_json(io.trip_dir(TRIP) / 'calib.json')
    axles = bogie_axles(calib)
    src = [q for q in d['prompts'] if q.get('bogie') == args.sample and q.get('source') != 'propagated']
    if not src:
        raise SystemExit(f'{args.sample} 上没有样板提示')
    sc, sz = parse_bogie(args.sample)
    t = Trip.from_id(TRIP)
    review = d.setdefault('review', {})
    summary = {}
    for dst in args.to:
        car, zone = parse_bogie(dst)
        if zone != sz and not args.cross:
            raise SystemExit(f'{dst} 和样板 {args.sample} 不同型（{zone} ≠ {sz}）；确实要搬加 --cross')
        b = bogie_no(car, zone)
        mp = axle_map(axles[args.sample], axles[dst])
        d['prompts'] = [q for q in d['prompts'] if not (q.get('bogie') == dst and q.get('source') == 'propagated')]
        flagged = []
        # 先放大件（含 BOGIE），螺栓按最近大件的局部残差再精修
        big = [q for q in src if q['type'] != 'BOLT']
        bolts = [q for q in src if q['type'] == 'BOLT']
        placed = []   # (中心 x, 中心 y, dx, dy)
        new = []
        for q in big + bolts:
            pb = map_box(q['box'], mp)
            w = max(pb[2] - pb[0], pb[3] - pb[1])
            if q['type'] == 'BOGIE':
                dx, dy, v = 0.0, 0.0, 1.0
            else:
                if q['type'] == 'BOLT' and placed:
                    cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    near = min(placed, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                    pb = [pb[0] + near[2], pb[1] + near[3], pb[2] + near[2], pb[3] + near[3]]
                    reach, step, ctx = 48, 1, 72
                else:
                    reach = 60 + int(0.08 * w)
                    step = max(1, int(w // 400))
                    ctx = 8
                dx, dy, v = ncc_refine(t, q['box'], pb, mp[0], reach, step, ctx)
            nb = [int(round(pb[0] + dx)), int(round(pb[1] + dy)), int(round(pb[2] + dx)), int(round(pb[3] + dy))]
            if q['type'] not in ('BOLT', 'BOGIE') and v >= NCC_MIN:
                placed.append(((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2, dx, dy))
            nid = 1000 * b + q['id'] % 1000
            # 大件位置由整台映射 + 局部精修定，NCC 稍低多是外观差别（线缆、附件），放进来但打标记；螺栓不硬放
            low = v < NCC_MIN and not (q['type'] != 'BOLT' and v >= NCC_BIG_MIN)
            if low:
                flagged.append({'id': nid, 'from': q['id'], 'type': q['type'], 'box': nb, 'ncc': round(v, 3)})
                continue
            nq = {'id': nid, 'type': q['type'], 'box': nb, 'bogie': dst, 'source': 'propagated',
                  'from': q['id'], 'ncc': round(v, 3)}
            if v < NCC_MIN:
                nq['flags'] = ['low_ncc']
            if q.get('mode'):
                nq['mode'] = q['mode']
            for k in ('pos', 'neg'):
                if q.get(k):
                    nq[k] = [[round(mp[1] + mp[0] * x + dx), round(y + mp[2] + dy)] for x, y in q[k]]
            new.append(nq)
        d['prompts'] += new
        review[dst] = flagged
        summary[dst] = {'placed': len(new), 'flagged': len(flagged), 'sx': round(mp[0], 4)}
    save_prompts(d)
    return {'step': 'label propagate', 'sample': args.sample, 'to': summary, 'out': io.rel(prompts_path())}


# ---------------------------------------------------------------- review

def load_types_order():
    p = io.ROOT / '主数据' / '零件类型.csv'
    if not p.exists():
        return list(COLORS)
    with open(p, encoding='utf-8-sig', newline='') as fh:
        return [r['type'] for r in csv.DictReader(fh)]


def draw_parts(im, parts, x0, x1, y0, step, flip=True):
    for q in parts:
        pts = (np.asarray(q['poly'], float) - [x0, y0]) / step
        if flip:
            pts[:, 0] = (x1 - x0) / step - 1 - pts[:, 0]
        pts = np.round(pts).astype(np.int32)
        bad = bool(q.get('flags')) or (q.get('ncc') is not None and q['ncc'] < 0.7)
        col = (0, 0, 255) if bad else COLORS.get(q['type'], (255, 255, 255))
        cv2.polylines(im, [pts], True, col, 3 if bad else 1)
        if q['type'] != 'BOGIE':
            tx, ty = int(pts[:, 0].min()), max(10, int(pts[:, 1].min()) - 3)
            cv2.putText(im, f"{q['id'] % 1000}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)


def cmd_review(args):
    parts = io.read_json(parts_path())['parts']
    d = load_prompts()
    m = io.read_json(io.trip_dir(TRIP) / 'manifest.json')
    t = Trip.from_id(TRIP)
    out = review_dir()
    out.mkdir(parents=True, exist_ok=True)
    step = args.step
    by_bogie = defaultdict(list)
    for q in parts:
        by_bogie[q.get('bogie') or 'BODY'].append(q)
    imgs = []
    for key, qs in sorted(by_bogie.items()):
        if key == 'BODY':
            continue
        bg = [q for q in qs if q['type'] == 'BOGIE']
        b = bg[0]['bbox'] if bg else box_of([p for q in qs for p in q['poly']])
        x0, x1 = b[0] - 400, b[2] + 400
        im = cv2.cvtColor(t.region(x0, x1, 400, H, step=step)[:, ::-1], cv2.COLOR_GRAY2BGR)
        draw_parts(im, qs, x0, x1, 400, step)
        cv2.putText(im, key, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        imgs.append(io.imwrite(out / f'{key}.jpg', im, quality=88))
    body = by_bogie.get('BODY', [])
    for p in m['train_panels']:
        qs = [q for q in parts if p['x0'] <= (q['bbox'][0] + q['bbox'][2]) / 2 < p['x1']]
        if not qs:
            continue
        s = step * 2
        im = cv2.cvtColor(t.region(p['x0'], p['x1'], 0, H, step=s)[:, ::-1], cv2.COLOR_GRAY2BGR)
        draw_parts(im, [q for q in qs if q['type'] != 'BOLT'], p['x0'], p['x1'], 0, s)
        imgs.append(io.imwrite(out / f'车{p["car"]}_整节.jpg', im, quality=85))
    # 计数表
    types = load_types_order()
    keys = sorted(k for k in by_bogie if k != 'BODY')
    cnt = {k: Counter(q['type'] for q in by_bogie[k]) for k in keys}
    lines = ['# 标注计数（railside label review 生成）', '',
             '| 转向架 | ' + ' | '.join(types) + ' | 合计 | 待复核 |', '|' + '---|' * (len(types) + 3)]
    for k in keys:
        rv = len(d.get('review', {}).get(k, [])) + sum(1 for q in by_bogie[k] if q.get('flags'))
        lines.append(f'| {k} | ' + ' | '.join(str(cnt[k].get(ty, 0)) for ty in types)
                     + f' | {sum(cnt[k].values())} | {rv} |')

    bc = Counter(q['type'] for q in body)
    lines.append('| 车下 | ' + ' | '.join(str(bc.get(ty, 0)) for ty in types) + f' | {sum(bc.values())} | |')
    tot = Counter(q['type'] for q in parts)
    lines.append('| **全车** | ' + ' | '.join(str(tot.get(ty, 0)) for ty in types) + f' | {len(parts)} | |')
    rv = [r | {'bogie': k} for k, rs in d.get('review', {}).items() for r in rs]
    rv += [{'id': q['id'], 'type': q['type'], 'flags': q['flags'], 'bogie': q.get('bogie')} for q in parts if q.get('flags')]
    lines += ['', f'## 待复核 {len(rv)} 条', '']
    lines += [f"- {r['bogie']} #{r['id']} {r['type']}：" + (f"搬运 NCC {r['ncc']}，未放" if 'ncc' in r and 'flags' not in r
                                                        else '、'.join(r.get('flags', []))) for r in rv]
    (out / '计数.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'step': 'label review', 'images': len(imgs), 'parts': len(parts), 'review': len(rv),
            'out': io.rel(out)}


# ---------------------------------------------------------------- CLI

def add_parser(sub, step):
    p = sub.add_parser(step, help='S4 零件预标：SAM、搬运、核对视图')
    s = p.add_subparsers(dest='action', required=True)
    v = s.add_parser('view')
    v.add_argument('--box', type=int, nargs=4, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'))
    v.add_argument('--step', type=int, default=0)
    v.add_argument('--grid', type=int, default=200)
    v.add_argument('--overlay', action='store_true', help='叠加现有 零件.json')
    v.add_argument('--out')
    a = s.add_parser('sam')
    a.add_argument('--ids', type=int, nargs='*')
    a.add_argument('--bogie', nargs='*')
    a.add_argument('--type', nargs='*')
    a.add_argument('--all', action='store_true')
    g = s.add_parser('propagate')
    g.add_argument('--sample', required=True)
    g.add_argument('--to', nargs='+', required=True)
    g.add_argument('--cross', action='store_true', help='允许搬到不同型的转向架')
    r = s.add_parser('review')
    r.add_argument('--step', type=int, default=4)
    p.set_defaults(func=run)


def run(args):
    return {'view': cmd_view, 'sam': cmd_sam, 'propagate': cmd_propagate, 'review': cmd_review}[args.action](args)
