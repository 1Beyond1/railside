"""S7 库：把各步的 JSON 输出装进 SQLite（设计.md §4.2，表结构见 schema.sql）。

库由文件重建，不是真源：`railside db build` 删库重建。智能体日志的真源是 数据/中间/<趟>/agent_log.jsonl，
`append_agent_log` 追加到它，库存在时同时写进库。
"""
import csv
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from pathlib import Path

import numpy as np

from .. import io
from ..coords import trip_to_raw

SCHEMA = Path(__file__).with_name('schema.sql')
SCHEMA_VER = '1'
TRIP_FILES = ('trip', 'qc', 'manifest', 'calib', 'align', 'seg', 'parts', 'compare')
CODE_RE = re.compile(r'^C(\d+)-([A-Z0-9]+)-([A-Za-z0-9_]+)-(\d+)$')
_db_lock = threading.Lock()


# ---------- 路径 ----------

def rel(path):
    """相对项目根的 posix 路径。先按未解析的路径算：worktree 里 数据/ 是 junction，io.rel 会解析到别处变成绝对路径。"""
    p = Path(path)
    try:
        return p.relative_to(io.ROOT).as_posix()
    except ValueError:
        return io.rel(p)


def default_db():
    return io.OUT / '库.sqlite'


def template_root():
    return io.OUT / '模板'


def norm_ver(v):
    if v is None:
        return None
    s = str(v)
    return s if s.startswith('v') else f'v{s}'


def latest_template():
    """数据/成品/模板 下编号最大的 v<n>（有 template.json 的）；没有时 None。"""
    root = template_root()
    if not root.is_dir():
        return None
    vers = [p.name for p in root.iterdir() if re.fullmatch(r'v\d+', p.name) and (p / 'template.json').is_file()]
    return max(vers, key=lambda s: int(s[1:])) if vers else None


def discover_trips():
    """数据/中间 下有 manifest.json 的趟。"""
    mid = io.MID
    if not mid.is_dir():
        return []
    return sorted(p.name for p in mid.iterdir() if io.is_valid_trip(p.name) and (p / 'manifest.json').is_file())


def resolve_path(p, *bases):
    """JSON 里记的路径：绝对路径原样；相对路径依次在 bases 下找，找到即返回，都没有返回 None。"""
    if not p:
        return None
    p = Path(p)
    if p.is_absolute():
        return p if p.exists() else None
    for b in bases:
        q = Path(b) / p
        if q.exists():
            return q
    return None


def load_trip_files(trip):
    """读一趟的各步 JSON：{名字: dict 或 None}。"""
    d = io.trip_dir(trip)
    return {k: (io.read_json(d / f'{k}.json') if (d / f'{k}.json').is_file() else None) for k in TRIP_FILES}


def load_template(ver):
    if not ver or not io.is_valid_trip(str(ver)):
        return None
    p = template_root() / str(ver) / 'template.json'
    return io.read_json(p) if p.is_file() else None


# ---------- 解析小工具（report 也用） ----------

def parse_code(code):
    """零件编码 C<车>-<区>-<类型>-<序号> → (car, zone, type, seq)；不合规返回 None。"""
    m = CODE_RE.match(code or '')
    return (int(m[1]), m[2], m[3], int(m[4])) if m else None


def box4(b):
    """[x0, y0, x1, y1] → 四个数；缺失返回 4 个 None。"""
    if b is None or len(b) != 4:
        return (None,) * 4
    return tuple(float(v) for v in b)


def frames_src(src_frames=None, x0=None, x1=None):
    """原图帧范围写成 'frames:f0-f1'。src_frames 可以是 [f0, f1]、[f, ...] 或 'f0-f1' 字符串。"""
    if src_frames is not None:
        if isinstance(src_frames, str):
            return f'frames:{src_frames}'
        fs = [int(f) for f in src_frames]
        if fs:
            return f'frames:{min(fs)}-{max(fs)}'
    if x0 is not None and x1 is not None:
        f0 = int(trip_to_raw(int(x0))[0])
        f1 = int(trip_to_raw(max(int(x0), int(x1) - 1))[0])
        return f'frames:{f0}-{f1}'
    return None


def alarms_from_compare(compare):
    """compare.json → 告警列表。每条 {item, code, type, value, threshold, bbox_uv, detail}。

    item 是 items[] 的下标（foreign 为 None）。alarms[] 元素可为字符串或对象。
    """
    out = []
    for i, it in enumerate(compare.get('items') or []):
        for a in it.get('alarms') or []:
            if isinstance(a, dict):
                rest = {k: v for k, v in a.items() if k not in ('type', 'kind', 'value', 'threshold')}
                out.append({'item': i, 'code': it.get('code'), 'type': a.get('type') or a.get('kind'),
                            'value': a.get('value'), 'threshold': a.get('threshold'),
                            'bbox_uv': a.get('bbox_uv'), 'detail': rest or None})
            else:
                out.append({'item': i, 'code': it.get('code'), 'type': str(a), 'value': None,
                            'threshold': None, 'bbox_uv': None, 'detail': None})
    for f in compare.get('foreign') or []:
        rest = {k: v for k, v in f.items() if k not in ('bbox_uv', 'area')}
        out.append({'item': None, 'code': None, 'type': 'foreign', 'value': f.get('area'), 'threshold': None,
                    'bbox_uv': f.get('bbox_uv'), 'detail': rest or None})
    return out


def _j(v):
    return None if v is None else json.dumps(v, ensure_ascii=False, sort_keys=True)


def _num_or_json(v):
    if v is None or isinstance(v, (int, float)):
        return v
    return _j(v)


def _unit(name):
    n = name.lower()
    for key, unit in (('mm2', 'mm2'), ('mm', 'mm'), ('deg', 'deg'), ('_px', 'px'), ('_uv', 'px'), ('_xy', 'px')):
        if key in n:
            return unit
    return None


def flatten_features(d, keys):
    """把 d 里 keys 列出的字段展开成 [(name, value, value_text)]。"""
    out = []

    def walk(name, v):
        if isinstance(v, bool) or v is None:
            out.append((name, None, _j(v)))
        elif isinstance(v, (int, float)):
            out.append((name, float(v), None))
        elif isinstance(v, (list, tuple)) and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v):
            for i, e in enumerate(v):
                out.append((f'{name}[{i}]', float(e), None))
        elif isinstance(v, dict):
            for k in sorted(v):
                walk(f'{name}.{k}', v[k])
        else:
            out.append((name, None, _j(v)))

    for k in keys:
        if k in d and d[k] is not None:
            walk(k, d[k])
    return out


# ---------- 建库 ----------

class _Builder:
    def __init__(self, con):
        self.con = con
        self.skipped = []

    def ins(self, table, row):
        cols = list(row)
        cur = self.con.execute(f'INSERT INTO {table} ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})',
                               [row[c] for c in cols])
        return cur.lastrowid

    def source(self, path, kind, trip=None, obj=None):
        data = Path(path).read_bytes()
        self.ins('source_file', {
            'path': rel(path), 'trip_id': trip, 'kind': kind, 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'code_ver': (obj or {}).get('code_ver'), 'created_at': (obj or {}).get('created_at')})

    def miss(self, path):
        self.skipped.append(rel(path))

    # 主数据
    def part_types(self):
        p = io.ROOT / '主数据' / '零件类型.csv'
        if not p.is_file():
            return self.miss(p)
        self.source(p, 'part_type')
        num = ('ncc_min', 'shift_mm_max', 'dtheta_max_deg', 'area_change_max')
        with open(p, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                row = {k: (r.get(k) or None) for k in ('type', 'name', 'parent_type', 'desc')}
                for k in num:
                    row[k] = float(r[k]) if r.get(k) not in (None, '') else None
                row['src'] = 'file:' + rel(p)
                self.ins('part_type', row)

    # 模板
    def template(self, ver):
        p = template_root() / ver / 'template.json'
        if not p.is_file():
            return self.miss(p)
        t = io.read_json(p)
        self.source(p, 'template', obj=t)
        ttrip, head_x = t.get('trip'), t.get('head_x')
        prov = {'template_ver': ver, 'src': 'file:' + rel(p), 'code_ver': t.get('code_ver'), 'created_at': t.get('created_at')}
        self.ins('template_version', {'template_ver': ver, 'trip_id': ttrip, 'head_x': head_x,
                                      'n_anchors': len(t.get('anchors') or []), 'n_parts': len(t.get('parts') or []),
                                      'calib_json': _j(t.get('calib')), 'src': prov['src'],
                                      'code_ver': prov['code_ver'], 'created_at': prov['created_at']})
        for a in t.get('anchors') or []:
            u, v = a.get('u'), a.get('v')
            self.ins('anchor', {'kind': 'template', 'trip_id': ttrip, 'anchor_id': str(a.get('id')), 'car': a.get('car'),
                                'x': None if u is None or head_x is None else u + head_x, 'y': v, 'u': u, 'v': v,
                                'w': a.get('w'), 'h': a.get('h'), 'score': None, 'patch': a.get('patch'), **prov})
        for part in t.get('parts') or []:
            u0, v0, u1, v1 = box4(part.get('bbox_uv'))
            pid = self.ins('part_instance', {
                'kind': 'template', 'trip_id': ttrip, 'code': part.get('code'), 'type': part.get('type'),
                'parent': part.get('parent'), 'car': part.get('car'), 'zone': part.get('zone'), 'found_by': None,
                'match_score': None, 'u0': u0, 'v0': v0, 'u1': u1, 'v1': v1, 'x0': None, 'y0': None, 'x1': None, 'y1': None,
                'crop': part.get('crop'), 'mask': part.get('mask'), **prov})
            self.features(pid, ttrip, part, ('centroid_uv', 'area_px', 'angle_deg', 'size_mm', 'feat'), prov)

    def features(self, pid, trip, d, keys, prov):
        for name, val, txt in flatten_features(d, keys):
            self.ins('part_feature', {'part_instance_id': pid, 'trip_id': trip, 'code': d.get('code'), 'name': name,
                                      'value': val, 'value_text': txt, 'unit': _unit(name), **prov})

    # 一趟
    def trip(self, trip, build_ver):
        d = io.trip_dir(trip)
        js = {}
        for k in TRIP_FILES:
            p = d / f'{k}.json'
            if p.is_file():
                js[k] = io.read_json(p)
                self.source(p, k, trip, js[k])
            else:
                js[k] = None
                self.miss(p)
        tj, qc, man, cal, al, seg, parts, cmp_ = (js[k] for k in TRIP_FILES)
        tv = norm_ver((al or {}).get('template_ver')) or build_ver
        first = tj or man or qc or {}
        trow = {
            'trip_id': trip, 'template_ver': tv,
            'src_dir': (tj or man or {}).get('src_dir'), 'camera': (tj or qc or {}).get('camera'),
            'n_frames': (tj or {}).get('n_frames') or (man or {}).get('n_frames') or (qc or {}).get('frames'),
            'src': 'file:' + rel(d / ('trip.json' if tj else 'manifest.json' if man else 'qc.json')),
            'code_ver': first.get('code_ver'), 'created_at': first.get('created_at'),
        }
        if man:
            tf = man.get('train_frames') or [None, None]
            trow.update(frame_w=man.get('frame_w'), frame_h=man.get('frame_h'), train_x0=man.get('train_x0'),
                        train_x1=man.get('train_x1'), head_x=man.get('head_x'), train_frame0=tf[0], train_frame1=tf[-1],
                        bad_seams=man.get('bad_seams'))
        if qc:
            trow.update(qc_verdict=qc.get('verdict'), qc_reasons_json=_j(qc.get('reasons')), qc_missing_json=_j(qc.get('missing')))
        if cal:
            sx = [p['mm_per_px'] for p in cal.get('sx_curve') or [] if p.get('mm_per_px') is not None]
            trow.update(calib_source=_j(cal.get('source')) if not isinstance(cal.get('source'), str) else cal['source'],
                        sy_mm_per_px=cal.get('sy_mm_per_px'),
                        sx_mm_per_px_median=float(np.median(sx)) if sx else None, calib_refs_json=_j(cal.get('refs')))
        if al:
            trow.update(align_residual_json=_j(al.get('residual_px')), align_u_trim0=al.get('u_trim0'),
                        n_align_anchors=len(al.get('anchors') or []))
        if seg:
            trow.update(seg_model_json=_j(seg.get('model')), n_seg_dets=len(seg.get('dets') or []))
        self.ins('trip', trow)

        def prov(obj, name, src=None):
            return {'trip_id': trip, 'template_ver': tv, 'src': src or 'file:' + rel(d / f'{name}.json'),
                    'code_ver': obj.get('code_ver'), 'created_at': obj.get('created_at')}

        if man:
            for p in man.get('train_panels') or []:
                f0 = int(trip_to_raw(p['x0'])[0])
                f1 = int(trip_to_raw(p['x1'] - 1)[0])
                self.ins('panel', {'panel': p.get('panel', p.get('car')), 'car': p.get('car'), 'x0': p['x0'], 'x1': p['x1'],
                                   'width_px': p.get('width_px'), 'height_px': p.get('height_px'), 'frame0': f0, 'frame1': f1,
                                   **prov(man, 'manifest', f'frames:{f0}-{f1}')})
        if al:
            for a in al.get('anchors') or []:
                x = a.get('x')
                self.ins('anchor', {'kind': 'align', 'anchor_id': str(a.get('id')), 'car': a.get('car'), 'x': x, 'y': a.get('y'),
                                    'u': a.get('u'), 'v': a.get('v'), 'w': a.get('w'), 'h': a.get('h'), 'score': a.get('score'),
                                    'patch': None, **prov(al, 'align', frames_src(x0=x, x1=x + 1) if x is not None else None)})
        part_src = {}
        if parts:
            for it in parts.get('instances') or []:
                code = it.get('code')
                cz = parse_code(code)
                u0, v0, u1, v1 = box4(it.get('bbox_uv'))
                x0, y0, x1, y1 = box4(it.get('bbox_xy'))
                src = frames_src(it.get('src_frames'), x0, x1)
                part_src[code] = src
                pv = prov(parts, 'parts', src)
                pid = self.ins('part_instance', {
                    'kind': 'trip', 'code': code, 'type': it.get('type'), 'parent': it.get('parent'),
                    'car': cz[0] if cz else None, 'zone': cz[1] if cz else None, 'found_by': it.get('found_by'),
                    'match_score': it.get('match_score'), 'u0': u0, 'v0': v0, 'u1': u1, 'v1': v1,
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'crop': it.get('crop'), 'mask': it.get('mask'), **pv})
                self.features(pid, trip, it, ('centroid_mm', 'area_mm2', 'angle_deg', 'match_score'), pv)
        if cmp_:
            alarms = alarms_from_compare(cmp_)
            cids = {}
            for i, it in enumerate(cmp_.get('items') or []):
                code = it.get('code')
                cids[i] = self.ins('comparison', {
                    'code': code, 'ncc': it.get('ncc'), 'diff_area': it.get('diff_area'),
                    'shift_mm': _num_or_json(it.get('shift_mm')), 'dtheta_deg': it.get('dtheta_deg'),
                    'n_alarms': len(it.get('alarms') or []), 'item_json': _j(it),
                    **prov(cmp_, 'compare', part_src.get(code))})
            for a in alarms:
                u0, v0, u1, v1 = box4(a['bbox_uv'])
                self.ins('alarm', {'comparison_id': cids.get(a['item']), 'code': a['code'], 'type': a['type'],
                                   'value': _num_or_json(a['value']), 'threshold': _num_or_json(a['threshold']),
                                   'u0': u0, 'v0': v0, 'u1': u1, 'v1': v1, 'detail_json': _j(a['detail']),
                                   **prov(cmp_, 'compare', part_src.get(a['code']))})
        lp = d / 'agent_log.jsonl'
        if lp.is_file():
            self.source(lp, 'agent_log', trip)
            for line in lp.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    _insert_log(self.con, json.loads(line), 'file:' + rel(lp))
        else:
            self.miss(lp)


def _insert_log(con, rec, src):
    con.execute(
        'INSERT INTO agent_log (trip_id, ts, step, tool, args_json, result_json, status, template_ver, src, code_ver, created_at)'
        ' VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (rec.get('trip'), rec.get('ts'), rec.get('step'), rec.get('tool'), _j(rec.get('args')), _j(rec.get('result')),
         rec.get('status'), rec.get('template_ver'), src, rec.get('code_ver'), rec.get('ts')))


def build(db_path=None, trips=None, template_ver=None):
    """删库重建。trips 默认 数据/中间 下有 manifest.json 的趟；template_ver 默认最新的 v<n>。返回摘要。"""
    db_path = Path(db_path) if db_path else default_db()
    trips = list(trips) if trips else discover_trips()
    tv = norm_ver(template_ver) or latest_template()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(f'{db_path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp')
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA.read_text(encoding='utf-8'))
        b = _Builder(con)
        b.part_types()
        if tv:
            b.template(tv)
        else:
            b.skipped.append(rel(template_root() / 'v*' / 'template.json'))
        for t in trips:
            b.trip(t, tv)
        meta = {'schema': SCHEMA_VER, 'built_at': io.now(), 'code_ver': io.code_ver(), 'template_ver': tv,
                'trips_json': _j(trips), 'skipped_json': _j(b.skipped)}
        con.executemany('INSERT INTO meta VALUES (?, ?)', list(meta.items()))
        con.commit()
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {t: con.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in tables}
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    else:
        con.close()
    try:
        with _db_lock:
            io._atomic_replace(tmp, db_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {'step': 'db', 'db': rel(db_path), 'template_ver': tv, 'trips': trips, 'rows': counts, 'skipped': b.skipped}


def append_agent_log(trip, step, tool, args, result, status, db_path=None):
    """追加一条智能体日志到 数据/中间/<趟>/agent_log.jsonl；库存在时同时写进库。返回这条记录。"""
    rec = {'ts': io.now(), 'trip': trip, 'step': step, 'tool': tool, 'args': args, 'result': result,
           'status': status, 'code_ver': io.code_ver()}
    p = io.trip_dir(trip) / 'agent_log.jsonl'
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=io._jsonable) + '\n'
    with _db_lock:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(line)
        db_path = Path(db_path) if db_path else default_db()
        if db_path.is_file():
            rec_db = json.loads(line)
            con = sqlite3.connect(db_path, timeout=10.0)
            try:
                with con:
                    _insert_log(con, rec_db, 'file:' + rel(p))
            finally:
                con.close()
    return rec


# ---------- 命令行 ----------

def _path_arg(s):
    p = Path(s)
    return p if p.is_absolute() else io.ROOT / p


def add_parser(sub, step):
    p = sub.add_parser(step, help='S7 库：从各步 JSON 删库重建 SQLite')
    sp = p.add_subparsers(dest='action', required=True)
    b = sp.add_parser('build', help='删库重建')
    b.add_argument('--trips', nargs='+', default=None, help='默认 数据/中间 下所有有 manifest.json 的趟')
    b.add_argument('--template', default=None, help='模板版本，如 v1；默认 数据/成品/模板 下最新的')
    b.add_argument('--db', default=None, help='默认 数据/成品/库.sqlite')
    b.set_defaults(func=run)


def run(args):
    return build(_path_arg(args.db) if args.db else None, args.trips, args.template)
