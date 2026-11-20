"""S7 查询：按趟、按零件编码、按告警查库，或跑只读 SQL。输出 JSON。

函数可直接调用（应用要用）：trip_info、part_history、alarms、sql。库只读打开。
"""
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from . import io
from .db import _path_arg, default_db, rel


def connect(db_path=None):
    """只读打开库。库不存在时报 FileNotFoundError。"""
    p = Path(db_path) if db_path else default_db()
    if not p.is_file():
        raise FileNotFoundError(f'库不存在: {rel(p)}，先跑 railside db build')
    con = sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only = ON')
    return con


def _row(r):
    d = dict(r)
    for k in list(d):
        if k.endswith('_json') and isinstance(d[k], str):
            d[k[:-5]] = json.loads(d.pop(k))
    return d


def _all(con, q, args=()):
    return [_row(r) for r in con.execute(q, args)]


def trip_info(trip, db_path=None):
    """一趟的概况：趟次行、面板、零件按 found_by 计数、告警按类型计数、智能体日志条数。"""
    with closing(connect(db_path)) as con:
        t = _all(con, 'SELECT * FROM trip WHERE trip_id = ?', (trip,))
        if not t:
            return {'trip': trip, 'found': False}
        return {
            'trip': trip, 'found': True, 'info': t[0],
            'panels': _all(con, 'SELECT * FROM panel WHERE trip_id = ? ORDER BY panel', (trip,)),
            'parts_by_found_by': {r['found_by']: r['n'] for r in con.execute(
                "SELECT found_by, count(*) n FROM part_instance WHERE kind = 'trip' AND trip_id = ? GROUP BY found_by", (trip,))},
            'alarms_by_type': {r['type']: r['n'] for r in con.execute(
                'SELECT type, count(*) n FROM alarm WHERE trip_id = ? GROUP BY type', (trip,))},
            'n_comparisons': con.execute('SELECT count(*) FROM comparison WHERE trip_id = ?', (trip,)).fetchone()[0],
            'n_agent_log': con.execute('SELECT count(*) FROM agent_log WHERE trip_id = ?', (trip,)).fetchone()[0],
        }


def _features(con, pid):
    return {r['name']: (r['value'] if r['value'] is not None else r['value_text'])
            for r in con.execute('SELECT name, value, value_text FROM part_feature WHERE part_instance_id = ? ORDER BY id', (pid,))}


def part_history(code, trip=None, db_path=None):
    """一个零件编码：类型主数据、模板里的定义、各趟的实例（含特征）、对比结果和告警。"""
    with closing(connect(db_path)) as con:
        tpl = _all(con, "SELECT * FROM part_instance WHERE kind = 'template' AND code = ? ORDER BY template_ver", (code,))
        for r in tpl:
            r['features'] = _features(con, r['id'])
        q = "SELECT * FROM part_instance WHERE kind = 'trip' AND code = ?" + (' AND trip_id = ?' if trip else '') + ' ORDER BY trip_id'
        trips = []
        for r in _all(con, q, (code, trip) if trip else (code,)):
            r['features'] = _features(con, r['id'])
            trips.append(r)
        cq = 'SELECT * FROM comparison WHERE code = ?' + (' AND trip_id = ?' if trip else '') + ' ORDER BY trip_id, id'
        comps = _all(con, cq, (code, trip) if trip else (code,))
        aq = 'SELECT * FROM alarm WHERE code = ?' + (' AND trip_id = ?' if trip else '') + ' ORDER BY trip_id, id'
        alarm_rows = _all(con, aq, (code, trip) if trip else (code,))
        types = {r['type'] for r in tpl} | {r['type'] for r in trips}
        ptype = _all(con, f'SELECT * FROM part_type WHERE type IN ({",".join("?" * len(types))})', sorted(t for t in types if t)) if types else []
        by_trip = {}
        for r in trips:
            by_trip.setdefault(r['trip_id'], {'instances': [], 'comparisons': [], 'alarms': []})['instances'].append(r)
        for r in comps:
            by_trip.setdefault(r['trip_id'], {'instances': [], 'comparisons': [], 'alarms': []})['comparisons'].append(r)
        for r in alarm_rows:
            by_trip.setdefault(r['trip_id'], {'instances': [], 'comparisons': [], 'alarms': []})['alarms'].append(r)
        return {'code': code, 'part_type': ptype, 'template': tpl,
                'trips': [{'trip_id': k, **v} for k, v in sorted(by_trip.items())]}


def alarms(trip=None, type=None, db_path=None):
    """告警列表，可按趟、按类型过滤；附对比指标。"""
    where, args = [], []
    if trip:
        where.append('a.trip_id = ?')
        args.append(trip)
    if type:
        where.append('a.type = ?')
        args.append(type)
    q = ('SELECT a.*, c.ncc, c.diff_area, c.shift_mm, c.dtheta_deg FROM alarm a LEFT JOIN comparison c ON c.id = a.comparison_id'
         + (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY a.trip_id, a.id')
    with closing(connect(db_path)) as con:
        return _all(con, q, args)


_READ_ONLY = re.compile(r'^\s*(select|with)\b', re.I)


def sql(query, db_path=None):
    """只读 SQL：只许 SELECT / WITH 开头的单条语句；库以只读方式打开，写操作会被 SQLite 拒绝。"""
    if not _READ_ONLY.match(query):
        raise ValueError('只许 SELECT 或 WITH 开头的查询')
    with closing(connect(db_path)) as con:
        try:
            cur = con.execute(query)
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
            raise ValueError(f'查询被拒绝或出错: {e}') from e
        cols = [c[0] for c in cur.description or []]
        return {'columns': cols, 'rows': [list(r) for r in cur.fetchall()]}


def add_parser(sub, step):
    p = sub.add_parser(step, help='S7 查询：按趟、零件编码、告警查库，输出 JSON')
    p.add_argument('--db', default=None, help='默认 数据/成品/库.sqlite')
    sp = p.add_subparsers(dest='what', required=True)
    t = sp.add_parser('trip', help='一趟的概况')
    t.add_argument('trip')
    pa = sp.add_parser('part', help='零件编码：模板信息 + 各趟实例和对比历史')
    pa.add_argument('code')
    pa.add_argument('--trip', default=None)
    a = sp.add_parser('alarms', help='告警列表')
    a.add_argument('--trip', default=None)
    a.add_argument('--type', default=None)
    s = sp.add_parser('sql', help='只读 SQL（SELECT / WITH）')
    s.add_argument('query')
    p.set_defaults(func=run)


def run(args):
    db = _path_arg(args.db) if args.db else None
    try:
        if args.what == 'trip':
            return trip_info(args.trip, db)
        if args.what == 'part':
            return part_history(args.code, args.trip, db)
        if args.what == 'alarms':
            rows = alarms(args.trip, args.type, db)
            return {'n': len(rows), 'alarms': rows}
        return sql(args.query, db)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f'query: {e}')
