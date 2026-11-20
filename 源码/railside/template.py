"""S4 模板：定位靶 + 零件身份模板。

  railside template anchors T01        自动挑定位靶 → 标注/模板T01/定位靶.json，并出总览图供人工确认
  railside template build --trip T01   由定位靶和零件标注生成 数据/成品/模板/v<n>/template.json，
                                       并冻结编码表 主数据/零件实例_v<n>.csv

模板坐标 u = x − head_x（模板趟车头），v = y。
"""
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import anchors as anc
from . import io
from .frames import Trip

def label_dir(trip):
    return io.ROOT / '标注' / f'模板{trip}'


def tpl_dir(ver):
    return io.OUT / '模板' / ver


def resolve_ver(ver=None):
    """给了版本就用它；没给取 数据/成品/模板 下最新的 v<n>。一个都没有时报错退出。"""
    if ver:
        return str(ver)
    from .db import latest_template
    v = latest_template()
    if not v:
        raise SystemExit('数据/成品/模板 下没有模板，先跑 railside template build')
    return v


def load_template(ver=None):
    return io.read_json(tpl_dir(resolve_ver(ver)) / 'template.json')


def load_types():
    p = io.ROOT / '主数据' / '零件类型.csv'
    if not p.exists():
        return {}
    with open(p, encoding='utf-8-sig', newline='') as fh:
        return {r['type']: r for r in csv.DictReader(fh)}


def add_parser(sub, step):
    p = sub.add_parser(step, help='S4 模板：anchors 选定位靶 / build 生成模板')
    s = p.add_subparsers(dest='action', required=True)
    a = s.add_parser('anchors')
    a.add_argument('trip', nargs='?', default='T01')
    b = s.add_parser('build')
    b.add_argument('--trip', default='T01')
    b.add_argument('--ver', required=True, help='模板版本，如 v3；已有模板目录不可覆盖，已冻结的编码表必须重建一致')
    p.set_defaults(func=run)


def run(args):
    return cmd_anchors(args) if args.action == 'anchors' else cmd_build(args)


def cmd_anchors(args):
    a = anc.select(args.trip)
    out = label_dir(args.trip) / '定位靶.json'
    io.write_json(out, {'trip': args.trip, 'patch': anc.PATCH, 'anchors': a, 'source': 'auto', 'verified': False})
    review = _anchor_review(args.trip, a)
    per_car = defaultdict(int)
    for x in a:
        per_car[x['car']] += 1
    return {'step': 'template anchors', 'trip': args.trip, 'n': len(a), 'per_car': dict(per_car),
            'out': io.rel(out), 'review': io.rel(review)}


def _anchor_review(trip, a):
    """每节车一行的总览图（翻转显示），框出靶位并标编号。"""
    m = io.read_json(io.trip_dir(trip) / 'manifest.json')
    t = Trip.from_id(trip)
    rows = []
    for p in m['train_panels']:
        im = cv2.cvtColor(t.region(p['x0'], p['x1'], step=16)[:, ::-1], cv2.COLOR_GRAY2BGR)
        for x in a:
            if x['car'] != p['car']:
                continue
            c0 = (p['x1'] - 1 - (x['x'] + x['w'])) // 16
            r0 = x['y'] // 16
            cv2.rectangle(im, (c0, r0), (c0 + x['w'] // 16, r0 + x['h'] // 16), (0, 220, 255), 2)
            cv2.putText(im, x['id'], (c0, max(12, r0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        rows.append(im)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (wmax - r.shape[1], 0), (0, 0))) for r in rows]
    return io.imwrite(io.trip_dir(trip) / '预览' / '定位靶_1-16.jpg', np.vstack(rows), quality=88)


# ---------------------------------------------------------------- build

def cmd_build(args):
    """生成模板 ver。

    模板目录已存在：拒绝，不覆盖。
    编码表 主数据/零件实例_<ver>.csv 已冻结（进 git，另一台机器从原图复现模板时就是这样）：照样生成，
    但生成出的编码表必须和冻结的逐行一致，否则删掉这次生成的目录并报错——编码定了不改。
    中途失败也删掉这次生成的目录，不留半成品。
    """
    trip, ver = args.trip, args.ver
    d = tpl_dir(ver)
    codes_path = io.ROOT / '主数据' / f'零件实例_{ver}.csv'
    if d.exists():
        raise SystemExit(f'模板版本 {ver} 已存在（{io.rel(d)}），禁止覆盖；请指定新的 --ver')
    frozen = _read_codes(codes_path) if codes_path.exists() else None
    try:
        out = _build(trip, ver, d, codes_path, frozen)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    out['codes'] = 'verified' if frozen is not None else 'frozen'
    return out


def _build(trip, ver, d, codes_path, frozen):
    m = io.read_json(io.trip_dir(trip) / 'manifest.json')
    head = m['head_x']
    a = io.read_json(label_dir(trip) / '定位靶.json')['anchors']
    t = Trip.from_id(trip)
    anchors_out = []
    for x in a:
        patch = d / 'anchors' / f"{x['id']}.png"
        io.imwrite(patch, t.region(x['x'], x['x'] + x['w'], x['y'], x['y'] + x['h']))
        anchors_out.append({'id': x['id'], 'car': x['car'], 'u': x['x'] - head, 'v': x['y'], 'w': x['w'], 'h': x['h'],
                            'patch': io.rel(patch)})
    calib = _maybe(io.trip_dir(trip) / 'calib.json')
    parts, codes = [], []
    lab_p = label_dir(trip) / '零件.json'
    if lab_p.exists():
        parts = build_parts(t, m, io.read_json(lab_p)['parts'], head, d, calib)
        codes = [{k: p[k] for k in ('code', 'type', 'parent', 'car', 'zone')} | {
            'u': round(p['centroid_uv'][0]), 'v': round(p['centroid_uv'][1])} for p in parts]
    if frozen is not None:
        diff = _codes_diff(frozen, codes)
        if diff:
            raise SystemExit(f'重建的编码表和冻结的 {io.rel(codes_path)} 不一致（{diff}）；'
                             '标注或标定变了就该出新版本，请指定新的 --ver')
    elif codes:
        _write_codes(codes_path, codes)
    io.write_json(d / 'template.json', {
        'version': ver, 'trip': trip, 'head_x': head, 'train_panels': m['train_panels'],
        'anchors': anchors_out, 'parts': parts, 'calib': calib,
        'labels_sha256': _sha(lab_p) if lab_p.exists() else None,
    })
    return {'step': 'template build', 'ver': ver, 'trip': trip, 'anchors': len(anchors_out), 'parts': len(parts),
            'out': io.rel(d / 'template.json')}


CODE_FIELDS = ['code', 'type', 'parent', 'car', 'zone', 'u', 'v']


def _read_codes(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return [{k: r.get(k) or '' for k in CODE_FIELDS} for r in csv.DictReader(fh)]


def _codes_diff(frozen, codes):
    """两份编码表逐行比；一致返回空串，否则返回第一处不同的简述。"""
    new = [{k: '' if r.get(k) is None else str(r[k]) for k in CODE_FIELDS} for r in codes]
    if len(new) != len(frozen):
        return f'行数 {len(new)} ≠ 冻结的 {len(frozen)}'
    for i, (a, b) in enumerate(zip(frozen, new), 1):
        if a != b:
            k = next(k for k in CODE_FIELDS if a[k] != b[k])
            return f'第 {i} 行 {a["code"]} 的 {k}：冻结 {a[k]!r}，重建 {b[k]!r}'
    return ''


def _maybe(p):
    return io.read_json(p) if Path(p).exists() else None


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_codes(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CODE_FIELDS)
        w.writeheader()
        w.writerows(rows)


def crop_margin(w, h):
    """零件局部图在外框四周留的边：长宽的 1/5，至少 16、至多 256 px（大件不必留几千 px 的背景）。"""
    return min(256, max(16, int(w) // 5)), min(256, max(16, int(h) // 5))


def assign_codes(labels, m):
    """给标注分配稳定编码 C<车>-<区>-<类型>-<序号>。返回 {标注 id: (code, car, zone)}。

    区：落在某台转向架（type=BOGIE）外框里的零件归该转向架，同车两台按 x 先后为 B1、B2；其余为 BODY。
    序号：同车、同区、同类型内按 x 再按 y（各取 50 px 网格）排序。
    """
    def car_of(x):
        for p in m['train_panels']:
            if p['x0'] <= x < p['x1']:
                return p['car']
        return 0

    def cx(l):
        b = l['bbox']
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    bogies = sorted([l for l in labels if l['type'] == 'BOGIE'], key=lambda l: l['bbox'][0])
    bzone = {}
    per_car = defaultdict(list)
    for b in bogies:
        per_car[car_of(cx(b)[0])].append(b)
    for car, bs in per_car.items():
        for i, b in enumerate(sorted(bs, key=lambda l: l['bbox'][0]), 1):
            bzone[b['id']] = (car, f'B{i}')
    groups = defaultdict(list)
    meta = {}
    for l in labels:
        x, y = cx(l)
        car = car_of(x)
        if l['type'] == 'BOGIE':
            car, zone = bzone[l['id']]
        else:
            zone = 'BODY'
            for b in bogies:
                bb = b['bbox']
                if bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]:
                    zone = bzone[b['id']][1]
                    break
        meta[l['id']] = (car, zone)
        groups[(car, zone, l['type'])].append((round(x / 50), round(y / 50), l['id']))
    out = {}
    for (car, zone, typ), items in groups.items():
        for n, (_, _, lid) in enumerate(sorted(items), 1):
            out[lid] = (f'C{car}-{zone}-{typ}-{n:02d}', car, zone)
    return out


def build_parts(t, m, labels, head, d, calib):
    codes = assign_codes(labels, m)
    types = load_types()
    by_id = {l['id']: l for l in labels}
    sx = _sx_at(calib)
    sy = (calib or {}).get('sy_mm_per_px')
    out = []
    for l in labels:
        code, car, zone = codes[l['id']]
        poly = np.asarray(l['poly'], float)
        x0, y0, x1, y1 = [int(v) for v in l['bbox']]
        mx, my = crop_margin(x1 - x0, y1 - y0)
        cx0, cy0, cx1, cy1 = x0 - mx, max(0, y0 - my), x1 + mx, min(4096, y1 + my)
        crop = t.region(cx0, cx1, cy0, cy1)
        mask = np.zeros(crop.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(poly - [cx0, cy0]).astype(np.int32)], 255)
        cp = io.imwrite(d / 'crops' / f'{code}.png', crop)
        mp = io.imwrite(d / 'masks' / f'{code}.png', mask)
        ys, xs = np.nonzero(mask)
        area = int(len(xs))
        cxy = (float(xs.mean() + cx0), float(ys.mean() + cy0)) if area else ((x0 + x1) / 2, (y0 + y1) / 2)
        angle = _orientation(xs, ys)
        pid = l.get('parent')
        if pid is None:
            pid = _auto_parent(l, labels, types)
        inside = crop[mask > 0]
        out.append({
            'code': code, 'type': l['type'], 'parent': codes[pid][0] if pid in codes else None,
            'car': car, 'zone': zone,
            'bbox_uv': [x0 - head, y0, x1 - head, y1],
            'poly_uv': [[round(px - head, 1), round(py, 1)] for px, py in poly],
            'centroid_uv': [round(cxy[0] - head, 1), round(cxy[1], 1)],
            'area_px': area, 'angle_deg': angle,
            'size_mm': [round((x1 - x0) * sx(cxy[0]), 1), round((y1 - y0) * sy, 1)] if (sx and sy) else None,
            'crop': io.rel(cp), 'crop_origin_uv': [cx0 - head, cy0], 'mask': io.rel(mp),
            'feat': {'mean': round(float(inside.mean()), 2) if area else None,
                     'std': round(float(inside.std()), 2) if area else None,
                     'hu': [round(float(h), 6) for h in cv2.HuMoments(cv2.moments(mask, True)).ravel()]},
            'source': l.get('source'), 'verified': l.get('verified', False), 'label_id': l['id'],
        })
    return out


def _orientation(xs, ys):
    if len(xs) < 5:
        return None
    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    w, v = np.linalg.eigh(c)
    vx, vy = v[:, 1]
    return round(float(np.degrees(np.arctan2(vy, vx))) % 180, 2)


def _auto_parent(l, labels, types):
    want = (types.get(l['type']) or {}).get('parent_type')
    if not want:
        return None
    b = l['bbox']
    best = None
    for o in labels:
        if o['type'] != want or o['id'] == l['id']:
            continue
        ob = o['bbox']
        if ob[0] <= (b[0] + b[2]) / 2 <= ob[2] and ob[1] <= (b[1] + b[3]) / 2 <= ob[3]:
            area = (ob[2] - ob[0]) * (ob[3] - ob[1])
            if best is None or area < best[0]:
                best = (area, o['id'])
    return best[1] if best else None


def _sx_at(calib):
    if not calib or not calib.get('sx_curve'):
        return None
    xs = np.array([p['x'] for p in calib['sx_curve']], float)
    ss = np.array([p['mm_per_px'] for p in calib['sx_curve']], float)
    return lambda x: float(np.interp(x, xs, ss))

