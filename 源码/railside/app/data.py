"""应用的数据层：趟列表、零件框、告警、零件详情、概况、局部图。

只读各步输出的 JSON（设计.md §4.1）。哪个文件缺了就退化为空，并给出提示文字。
框的坐标：`xy` 是趟坐标 [x0, y0, x1, y1]（不翻转），`rect` 是翻转显示图里的 [xd, y, w, h]。
"""
import csv
import functools
import math
import re
from pathlib import Path

import cv2
import numpy as np

from .. import io
from ..coords import AlignMap, H, display_x
from . import dzi

TRIP_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')

# 流水线各步的输出文件（概况面板的“文件”一栏）
PIPE_FILES = [
    ('qc.json', 'S1 质检'), ('manifest.json', 'S2 拼接'), ('calib.json', 'S3 标定'),
    ('align.json', 'S5 对齐'), ('seg.json', 'S6a 分割'), ('parts.json', 'S6b 身份'),
    ('compare.json', 'S8 对比'),
]
BIG_KEYS = {'poly_uv', 'poly_xy', 'poly', 'mask', 'feat'}  # 详情里不回传的大字段
STATUS_COLOR = {'ok': (139, 189, 62), 'alarm': (58, 163, 242), 'missing': (90, 90, 239)}  # BGR
TEMPLATE_COLOR = (211, 161, 78)


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


# ---------- 读文件 ----------

def load(path):
    """读 JSON；文件不存在返回 None。按修改时间缓存（返回的对象只读，不要改）。"""
    path = Path(path)
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return _read(str(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=64)
def _read(path, _mtime, _size):
    return io.read_json(path)


def trip_file(trip, name):
    return load(io.trip_dir(trip) / name)


def check_trip(trip):
    if not TRIP_RE.match(trip or ''):
        raise BadRequest(f'趟次号不合法：{trip!r}（只许字母、数字、- 和 _）')
    if not (io.trip_dir(trip) / 'manifest.json').exists():
        raise NotFound(f'没有趟次 {trip}：缺 manifest.json，先跑 stitch')
    return trip


def files_present(trip):
    d = io.trip_dir(trip)
    return [{'file': f, 'step': s, 'exists': (d / f).exists()} for f, s in PIPE_FILES]


def list_trips():
    out = []
    for p in sorted(io.MID.glob('*/manifest.json')):
        trip = p.parent.name
        if not TRIP_RE.match(trip):
            continue
        m = load(p)
        qc = trip_file(trip, 'qc.json')
        out.append({
            'trip': trip, 'n_frames': m.get('n_frames'),
            'train_x0': m.get('train_x0'), 'train_x1': m.get('train_x1'),
            'n_cars': len(m.get('train_panels') or []),
            'verdict': qc.get('verdict') if qc else None,
            'files': files_present(trip),
        })
    return out


def trip_info(trip):
    m = trip_file(check_trip(trip), 'manifest.json')
    p = dzi.pyramid(trip)
    cars = []
    for c in m.get('train_panels') or []:
        xd = int(display_x(c['x1'] - 1, p.x0, p.x1))
        cars.append({'car': c['car'], 'x0': c['x0'], 'x1': c['x1'], 'width_px': c['width_px'], 'xd': xd, 'w': c['x1'] - c['x0']})
    return {
        'trip': trip, 'train_x0': p.x0, 'train_x1': p.x1, 'head_x': m.get('head_x', p.x0),
        'width': p.width, 'height': p.height, 'n_frames': m.get('n_frames'), 'frame_w': m.get('frame_w', 2000),
        'dzi': {'url': f'/api/trips/{trip}/dzi.dzi', 'tile': dzi.TILE, 'overlap': dzi.OVERLAP, 'levels': p.max_level + 1},
        'cars': cars, 'bad_seams': m.get('bad_seams'),
    }


# ---------- 模板与主数据 ----------

def _ver_dir(ver):
    ver = str(ver)
    return io.OUT / '模板' / (ver if ver.startswith('v') else f'v{ver}')


def latest_template():
    best = None
    for p in (io.OUT / '模板').glob('v*/template.json'):
        m = re.fullmatch(r'v(\d+)', p.parent.name)
        if m and (best is None or int(m[1]) > best[0]):
            best = (int(m[1]), p)
    return load(best[1]) if best else None


def template_for(trip):
    """这一趟对齐所用的模板版本；没有对齐结果时取最新版。"""
    align = trip_file(trip, 'align.json')
    if align and align.get('template_ver') is not None:
        tpl = load(_ver_dir(align['template_ver']) / 'template.json')
        if tpl:
            return tpl
    return latest_template()


def type_names():
    p = io.ROOT / '主数据' / '零件类型.csv'
    if not p.exists():
        return {}
    with open(p, encoding='utf-8-sig', newline='') as f:
        return {r['type']: r.get('name') or r['type'] for r in csv.DictReader(f) if r.get('type')}


def annotation(trip):
    """模板趟的人工标注（`标注/模板<趟>/零件.json`）；模板还没生成时用它叠框。"""
    return load(io.ROOT / '标注' / f'模板{trip}' / '零件.json')


# ---------- 零件框 ----------

def alarm_types(item):
    """compare.json 的 alarms[] 可能是字符串或带 type 的对象，统一成字符串。"""
    out = []
    for a in (item or {}).get('alarms') or []:
        if isinstance(a, dict):
            a = a.get('type') or a.get('kind') or a.get('name') or 'other'
        out.append(str(a))
    return out


def status_of(inst, cmp_item):
    if inst and inst.get('found_by') == 'missing':
        return 'missing'
    if alarm_types(cmp_item):
        return 'alarm'
    return 'ok'


class Context:
    """一趟的零件相关文件，一次读齐。"""

    def __init__(self, trip):
        self.trip = check_trip(trip)
        self.m = trip_file(trip, 'manifest.json')
        self.x0, self.x1 = int(self.m['train_x0']), int(self.m['train_x1'])
        self.tpl = template_for(trip)
        self.tpl_parts = {p['code']: p for p in (self.tpl or {}).get('parts') or []}
        self.is_tpl_trip = bool(self.tpl) and self.tpl.get('trip') == trip
        parts = trip_file(trip, 'parts.json')
        self.has_parts = parts is not None
        self.insts = {p['code']: p for p in (parts or {}).get('instances') or []}
        cmp = trip_file(trip, 'compare.json')
        self.cmp = cmp
        self.cmp_items = {c['code']: c for c in (cmp or {}).get('items') or []}
        align = trip_file(trip, 'align.json')
        self.align = align
        self.amap = AlignMap.from_json(align) if align and len(align.get('g') or []) >= 2 else None
        self.names = type_names()

    def uv_to_xy(self, bbox_uv, tpl_trip=False):
        """模板坐标框 → 本趟趟坐标框。模板趟：x = u + head_x；其他趟用 align.json 的映射。"""
        u0, v0, u1, v1 = bbox_uv
        if tpl_trip or self.is_tpl_trip:
            hx = self.tpl['head_x']
            return [u0 + hx, v0, u1 + hx, v1]
        if self.amap is None:
            return None
        (xa, xb), (ya, yb) = self.amap.to_trip([u0, u1], [v0, v1])
        return [float(xa), float(ya), float(xb), float(yb)]

    def rect(self, xy):
        x0, y0, x1, y1 = xy
        return [float(display_x(x1 - 1, self.x0, self.x1)), y0, x1 - x0, y1 - y0]

    def record(self, code, tp=None, inst=None):
        tp = tp if tp is not None else self.tpl_parts.get(code)
        inst = inst if inst is not None else self.insts.get(code)
        item = self.cmp_items.get(code)
        xy = None
        if self.is_tpl_trip and tp and tp.get('bbox_uv'):
            xy = self.uv_to_xy(tp['bbox_uv'])
        elif inst and inst.get('bbox_xy'):
            xy = list(inst['bbox_xy'])
        elif inst and inst.get('bbox_uv'):
            xy = self.uv_to_xy(inst['bbox_uv'])
        src = tp or inst or {}
        status = status_of(inst, item)
        alarms = alarm_types(item)
        if status == 'missing' and 'missing' not in alarms:
            alarms = ['missing', *alarms]
        typ = src.get('type')
        return {
            'code': code, 'type': typ, 'type_name': self.names.get(typ, typ), 'parent': src.get('parent'),
            'car': src.get('car'), 'zone': src.get('zone'), 'status': status, 'alarms': alarms,
            'xy': xy, 'rect': self.rect(xy) if xy else None,
        }

    def records(self):
        """(来源, 零件记录列表)。模板趟叠模板零件，其他趟叠 parts.json。"""
        if self.is_tpl_trip and self.tpl_parts:
            recs = [self.record(c) for c in self.tpl_parts]
            recs += [self.record(c) for c in self.insts if c not in self.tpl_parts]
            return 'template', recs
        if self.has_parts:
            return 'parts', [self.record(c) for c in self.insts]
        ann = annotation(self.trip)
        if ann:
            recs = []
            for p in ann.get('parts') or []:
                xy = list(p['bbox'])
                recs.append({'code': f"#{p['id']}", 'type': p.get('type'), 'type_name': self.names.get(p.get('type'), p.get('type')),
                             'parent': p.get('parent'), 'car': None, 'zone': None, 'status': 'ok', 'alarms': [],
                             'xy': xy, 'rect': self.rect(xy), 'verified': p.get('verified')})
            return 'annotation', recs
        return None, []


def anchor_boxes(ctx):
    """定位靶：模板趟取模板里的靶（u, v 是左上角），其他趟取 align.json 找到的位置（尺寸取模板）。"""
    tpl_a = {a['id']: a for a in (ctx.tpl or {}).get('anchors') or []}
    out = []
    if ctx.is_tpl_trip:
        for a in tpl_a.values():
            xy = ctx.uv_to_xy([a['u'], a['v'], a['u'] + a['w'], a['v'] + a['h']])
            out.append({'id': a['id'], 'car': a.get('car'), 'ok': True, 'score': None, 'xy': xy, 'rect': ctx.rect(xy)})
    elif ctx.align:
        for a in ctx.align.get('anchors') or []:
            t = tpl_a.get(a['id'], {})
            w, h = a.get('w', t.get('w', 512)), a.get('h', t.get('h', 512))
            xy = [a['x'], a['y'], a['x'] + w, a['y'] + h]
            out.append({'id': a['id'], 'car': a.get('car'), 'ok': a.get('ok', True), 'score': a.get('score'),
                        'xy': xy, 'rect': ctx.rect(xy)})
    return out


def overlay(trip):
    ctx = Context(trip)
    source, recs = ctx.records()
    boxes = [r for r in recs if r['rect']]
    anchors = anchor_boxes(ctx)
    counts = {k: sum(r['status'] == k for r in boxes) for k in ('ok', 'alarm', 'missing')}
    msg = None
    if source is None:
        if ctx.is_tpl_trip:
            msg = f"模板 {ctx.tpl.get('version')} 里还没有零件（标注未导入），暂无零件框"
        elif ctx.tpl:
            msg = '本趟还没有零件识别结果（parts.json 不存在），先跑 match'
        else:
            msg = '还没有模板（数据/成品/模板/v*/template.json）和本趟零件识别结果（parts.json），暂无零件框'
    elif source == 'annotation':
        msg = '模板还没生成，显示的是人工标注（标注/模板%s/零件.json），没有编码和对比状态' % trip
    elif len(boxes) < len(recs):
        msg = f'{len(recs) - len(boxes)} 个零件没有位置（缺 bbox 且无对齐映射），未画出'
    return {'trip': trip, 'source': source, 'message': msg, 'counts': counts,
            'template_ver': (ctx.tpl or {}).get('version'), 'template_trip': (ctx.tpl or {}).get('trip'),
            'boxes': boxes, 'anchors': anchors}


def _metrics(item):
    return {k: item.get(k) for k in ('ncc', 'diff_area', 'shift_mm', 'dtheta_deg')} if item else {}


def alarms(trip):
    ctx = Context(trip)
    source, recs = ctx.records()
    rows = [{**r, 'kind': 'part', 'metrics': _metrics(ctx.cmp_items.get(r['code']))} for r in recs if r['status'] != 'ok']
    for i, f in enumerate((ctx.cmp or {}).get('foreign') or [], 1):
        xy = ctx.uv_to_xy(f['bbox_uv']) if f.get('bbox_uv') and (ctx.is_tpl_trip or ctx.amap) else None
        rows.append({'code': f'异物-{i:02d}', 'kind': 'foreign', 'type': 'FOREIGN', 'type_name': '异物', 'parent': None,
                     'status': 'alarm', 'alarms': ['foreign'], 'xy': xy, 'rect': ctx.rect(xy) if xy else None,
                     'metrics': {'area': f.get('area')}})
    types = {}
    for r in rows:
        for a in r['alarms']:
            types[a] = types.get(a, 0) + 1
    msg = None
    if ctx.cmp is None and not ctx.has_parts:
        msg = '本趟还没有对比结果（compare.json 不存在），跑完 S8 对比后这里列出告警'
    elif ctx.cmp is None:
        msg = '还没有对比结果（compare.json 不存在），下面只列缺失零件'
    return {'trip': trip, 'message': msg, 'types': types, 'rows': rows}


# ---------- 零件详情 ----------

def _strip(d):
    return {k: v for k, v in d.items() if k not in BIG_KEYS} if d else None


def find_record(ctx, code):
    _, recs = ctx.records()
    for r in recs:
        if r['code'] == code:
            return r
    raise NotFound(f'{ctx.trip} 没有零件 {code}')


def part_detail(trip, code):
    ctx = Context(trip)
    rec = find_record(ctx, code)
    tp = ctx.tpl_parts.get(code)
    return {
        'trip': trip, 'record': rec,
        'template': {'version': ctx.tpl.get('version'), 'trip': ctx.tpl.get('trip'), 'part': _strip(tp)} if tp else None,
        'instance': _strip(ctx.insts.get(code)),
        'compare': ctx.cmp_items.get(code),
        'crops': {'template': bool(tp and tp.get('bbox_uv')), 'trip': rec['xy'] is not None},
        'history': history(code),
    }


def history(code):
    """跨趟历史。优先调 S7 的 railside.query.part_history；导入失败、库不存在或库里没有这个编码时读各趟 JSON。"""
    note = None
    try:
        from .. import query
        res = query.part_history(code)
        rows = _rows_from_history(res)
        if rows:
            return {'source': '库（railside.query）', 'note': None, 'rows': rows}
        note = '库里没有这个编码的实例（库没重建？），改为直接读各趟 JSON'
    except ImportError:
        note = 'railside.query 还没有，直接读各趟 JSON'
    except Exception as e:  # 库没建、查询出错：照样给出 JSON 里的历史
        note = f'railside.query 出错（{type(e).__name__}: {e}），改为直接读各趟 JSON'
    rows = []
    for t in list_trips():
        ctx = Context(t['trip'])
        if not (ctx.is_tpl_trip and code in ctx.tpl_parts) and code not in ctx.insts:
            continue  # 这一趟还没识别过这个零件
        inst = ctx.insts.get(code) or {}
        item = ctx.cmp_items.get(code)
        rows.append({'trip': t['trip'], 'status': status_of(inst, item),
                     'found_by': inst.get('found_by') or ('template' if ctx.is_tpl_trip else None),
                     'match_score': inst.get('match_score'), **_metrics(item), 'alarms': alarm_types(item)})
    return {'source': '各趟 JSON', 'note': note, 'rows': rows}


def _rows_from_history(res):
    """query.part_history 的 trips[]（trip_id、instances、comparisons、alarms）→ 网页要的行。"""
    rows = []
    for t in res.get('trips') or []:
        inst = (t.get('instances') or [{}])[0]
        cmp_ = (t.get('comparisons') or [{}])[0]
        alarms = [a.get('type') for a in t.get('alarms') or [] if a.get('type')]
        status = 'missing' if inst.get('found_by') == 'missing' else ('alarm' if alarms else 'ok')
        rows.append({'trip': t.get('trip_id'), 'status': status, 'found_by': inst.get('found_by'),
                     'match_score': inst.get('match_score'), 'ncc': cmp_.get('ncc'),
                     'diff_area': cmp_.get('diff_area'), 'shift_mm': cmp_.get('shift_mm'),
                     'dtheta_deg': cmp_.get('dtheta_deg'), 'alarms': alarms})
    return rows


def crop(trip, code, side):
    """零件局部图 JPEG（翻转显示，框出零件）。side = template | trip。"""
    ctx = Context(trip)
    rec = find_record(ctx, code)
    if side == 'template':
        tp = ctx.tpl_parts.get(code)
        if not (tp and tp.get('bbox_uv')):
            raise NotFound(f'模板里没有 {code}')
        src_trip, xy, color = ctx.tpl['trip'], ctx.uv_to_xy(tp['bbox_uv'], tpl_trip=True), TEMPLATE_COLOR
    elif side == 'trip':
        if rec['xy'] is None:
            raise NotFound(f'{trip} 的 {code} 没有位置')
        src_trip, xy, color = trip, rec['xy'], STATUS_COLOR[rec['status']]
    else:
        raise BadRequest(f'side 只能是 template 或 trip：{side}')
    check_trip(src_trip)
    return _crop(str(io.trip_dir(src_trip)), src_trip, tuple(int(round(v)) for v in xy), color)


@functools.lru_cache(maxsize=256)
def _crop(_trip_dir, trip, xy, color, max_side=560):
    bx0, by0, bx1, by1 = xy
    bw, bh = max(bx1 - bx0, 1), max(by1 - by0, 1)
    pad = max(48, int(0.6 * max(bw, bh)))
    t = dzi.reader(trip)
    cx0, cx1 = max(0, bx0 - pad), min(t.width, bx1 + pad)
    cy0, cy1 = max(0, by0 - pad), min(H, by1 + pad)
    step = max(1, math.ceil(max(cx1 - cx0, cy1 - cy0) / max_side))
    im = t.region(cx0, cx1, cy0, cy1, step=step)[:, ::-1]
    im = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_GRAY2BGR)
    # 翻转后：显示列 k ↔ 趟坐标 x = cx1 - 1 - k*step
    p0 = (int((cx1 - bx1) / step), int((by0 - cy0) / step))
    p1 = (int((cx1 - bx0) / step), int((by1 - cy0) / step))
    cv2.rectangle(im, p0, p1, color, 2 if step > 1 else 3, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


# ---------- 概况（质检、拼接、对齐……） ----------

def summary(trip):
    check_trip(trip)
    qc = trip_file(trip, 'qc.json')
    m = trip_file(trip, 'manifest.json')
    calib = trip_file(trip, 'calib.json')
    align = trip_file(trip, 'align.json')
    seg = trip_file(trip, 'seg.json')
    parts = trip_file(trip, 'parts.json')
    cmp = trip_file(trip, 'compare.json')
    out = {'trip': trip, 'files': files_present(trip)}
    if qc:
        flagged = [{'f': p['f'], 'flags': p['flags']} for p in qc.get('per_frame') or [] if p.get('flags')]
        out['qc'] = {'verdict': qc.get('verdict'), 'reasons': qc.get('reasons') or [], 'frames': qc.get('frames'),
                     'missing': qc.get('missing') or [], 'train_frames': qc.get('train_frames'), 'flagged': flagged,
                     'readonly': qc.get('readonly'), 'created_at': qc.get('created_at'), 'code_ver': qc.get('code_ver')}
    if m:
        out['stitch'] = {'train_x0': m.get('train_x0'), 'train_x1': m.get('train_x1'), 'head_x': m.get('head_x'),
                         'train_frames': m.get('train_frames'), 'bad_seams': m.get('bad_seams'),
                         'n_seams': len(m.get('seams') or []),
                         'cars': [{k: c.get(k) for k in ('car', 'x0', 'x1', 'width_px')} for c in m.get('train_panels') or []]}
    if calib:
        sx = [p.get('mm_per_px') for p in calib.get('sx_curve') or [] if p.get('mm_per_px') is not None]
        out['calib'] = {'source': calib.get('source'), 'n_refs': len(calib.get('refs') or []),
                        'sx_min': min(sx) if sx else None, 'sx_max': max(sx) if sx else None,
                        'sy_mm_per_px': calib.get('sy_mm_per_px')}
    if align:
        out['align'] = {'template_ver': align.get('template_ver'), 'n_anchors': len(align.get('anchors') or []),
                        'n_g': len(align.get('g') or []), 'residual_px': align.get('residual_px'), 'u_trim0': align.get('u_trim0'),
                        **{k: align.get(k) for k in ('anchors_found', 'anchors_total', 'scale_range', 'offset_range_px',
                                                     'blocks_used', 'blocks_total', 'qc2_flagged') if k in align}}
    if seg:
        out['seg'] = {'model': seg.get('model'), 'n_dets': len(seg.get('dets') or [])}
    if parts:
        by = {}
        for p in parts.get('instances') or []:
            by[p.get('found_by')] = by.get(p.get('found_by'), 0) + 1
        out['parts'] = {'n': len(parts.get('instances') or []), 'found_by': by}
    if cmp:
        items = cmp.get('items') or []
        out['compare'] = {'n_items': len(items), 'n_alarm_parts': sum(bool(alarm_types(i)) for i in items),
                          'n_foreign': len(cmp.get('foreign') or [])}
    return out
