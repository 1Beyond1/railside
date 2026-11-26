"""评分：拿各步输出和 标注/评测/ 的真值算 设计.md §5 表里的格子。缺输入的格子报“缺”和原因，不崩。

  python -m railside.eval.score                 # 打印表，并写 实验/E5/评分.json
  python -m railside.eval.score --json-only

各格口径：
  质检   坏帧真值 vs <坏帧趟>/qc.json：missing[] 和逐帧 flags；检出率按类型，误报 = 真值外被标记的帧
  标定   T01、T02 calib.json：12 处 sx 离散 sd（%）、两趟 sx 均值差（%）
  对齐   标注/评测/对应点.csv（手标）+ T02 align.json：误差中位数、p95（没有手标点就报缺）
  身份   T02 parts.json：找到率（按 found_by 分）
  对比   合成缺陷真值 vs <合成趟>/compare.json：
           “检出” = 目标零件 status 为 alarm/missing，或零件外框之外有 foreign 框压到缺陷外框
           “类型对” = 告警类型和缺陷对得上（missing→missing，rotate→dtheta，shift→shift，
                      occlude→任何告警，foreign→foreign 框）
         误报：真实 T02 的告警数（两趟同一列车、没有真缺陷）；合成趟里离所有缺陷都远的告警数
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .. import io

KIND_ALARM = {'missing': {'missing'}, 'rotate': {'dtheta'}, 'shift': {'shift'}, 'occlude': None, 'foreign': None}
NEAR_PX = 300


def _load(p):
    p = Path(p)
    return io.read_json(p) if p.is_file() else None


def _overlap(a, b, pad=0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


# ------------------------------------------------------------------ 质检

def score_qc(truth, qc):
    if truth is None:
        return {'status': '缺', 'why': '没有 标注/评测/坏帧.json，先跑 python -m railside.eval.badframes'}
    if qc is None:
        return {'status': '缺', 'why': f"没有 数据/中间/{truth['trip']}/qc.json，先跑 railside ingest"}
    flagged = {p['f']: set(p.get('flags') or []) for p in qc.get('per_frame') or [] if p.get('flags')}
    missing = set(qc.get('missing') or [])
    want = {'dark': {'dark'}, 'over': {'over'}, 'blur': {'blur'}, 'occlusion': {'occlusion'}}
    rows, bad_f = {}, set()
    for b in truth['bad']:
        f, k = b['f'], b['kind']
        bad_f.add(f)
        r = rows.setdefault(k, {'n': 0, 'hit': 0, 'hit_any': 0, 'miss': []})
        r['n'] += 1
        if k == 'missing':
            ok = any_ok = f in missing
        else:
            got = flagged.get(f, set())
            ok, any_ok = bool(got & want[k]), bool(got)
        r['hit'] += ok
        r['hit_any'] += any_ok
        if not ok:
            r['miss'].append({'f': f, 'got': sorted(flagged.get(f, set())), 'params': b['params']})
    fp = sorted(f for f in flagged if f not in bad_f)
    n = sum(r['n'] for r in rows.values())
    hit = sum(r['hit'] for r in rows.values())
    return {'status': 'ok', 'trip': truth['trip'], 'n': n, 'hit': hit, 'recall': round(hit / n, 3) if n else None,
            'by_kind': rows, 'false_flags': fp, 'n_false': len(fp), 'verdict': qc.get('verdict')}


# ------------------------------------------------------------------ 标定

def score_calib(c1, c2):
    if not c1 or not c2:
        return {'status': '缺', 'why': '缺 T01 或 T02 的 calib.json'}

    def sd(c):
        s = np.array([p['mm_per_px'] for p in c['sx_curve']], float)
        return float(s.mean()), float(s.std() / s.mean() * 100), len(s)   # 总体 sd，与 calib.json、E1 同口径
    m1, s1, n1 = sd(c1)
    m2, s2, n2 = sd(c2)
    return {'status': 'ok', 'sx_mean': [round(m1, 4), round(m2, 4)], 'sd_pct': [round(s1, 2), round(s2, 2)],
            'n': [n1, n2], 'trip_diff_pct': round((m2 - m1) / m1 * 100, 2)}


# ------------------------------------------------------------------ 对齐

def score_align(pts_path, align):
    if not Path(pts_path).is_file():
        return {'status': '缺', 'why': f'没有手标对应点 {io.rel(pts_path)}（要人看图标，不能用算法代替）'}
    if align is None:
        return {'status': '缺', 'why': '没有 T02 align.json'}
    from ..coords import AlignMap
    m = AlignMap.from_json(align)
    with open(pts_path, encoding='utf-8-sig', newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get('x_t02') or '').strip()]   # 跳过的点（看不清）不算
    # T02 的点映射到模板坐标，再加模板趟车头 = T01 趟坐标，与手标 T01 位置比
    man = _load(io.trip_dir(align.get('template_trip') or 'T01') / 'manifest.json')
    if not man:
        return {'status': '缺', 'why': '没有模板趟的 manifest.json（要它的 head_x）'}
    head = man['head_x']
    err = []
    for r in rows:
        u, v = m.to_template(float(r['x_t02']), float(r['y_t02']))
        err.append(float(np.hypot(float(u) + head - float(r['x_t01']), float(v) - float(r['y_t01']))))
    e = np.array(err)
    return {'status': 'ok', 'n': len(e), 'median_px': round(float(np.median(e)), 1), 'p95_px': round(float(np.percentile(e, 95)), 1)}


# ------------------------------------------------------------------ 身份

def score_identity(parts):
    if parts is None:
        return {'status': '缺', 'why': '没有 T02 parts.json'}
    items = parts['instances']
    by = {}
    for i in items:
        by[i.get('found_by')] = by.get(i.get('found_by'), 0) + 1
    found = sum(v for k, v in by.items() if k not in ('missing', None))
    return {'status': 'ok', 'n': len(items), 'found': found, 'found_rate': round(found / len(items), 4) if items else None,
            'by_found_by': by, 'code_check': '缺：编码正确率要人工抽查 20 个，未做'}


# ------------------------------------------------------------------ 对比

def _alarms(item):
    return {a['type'] for a in item.get('alarms') or []}


def score_compare(truth, cmp, base, parts=None):
    if truth is None:
        return {'status': '缺', 'why': '没有 标注/评测/合成缺陷.json，先跑 python -m railside.eval.synth'}
    if cmp is None:
        return {'status': '缺', 'why': f"没有 数据/中间/{truth['trip']}/compare.json，先跑 railside run 合成趟"}
    if parts is None and io.is_valid_trip(truth.get('trip')):
        parts = _load(io.trip_dir(truth['trip']) / 'parts.json')
    part_boxes = {p['code']: p['bbox_xy'] for p in (parts or {}).get('instances') or []
                  if p.get('code') and p.get('bbox_xy')}

    def box_of(item):
        return item.get('bbox_xy') or part_boxes.get(item.get('code'))

    items = {i['code']: i for i in cmp['items']}
    foreign = cmp.get('foreign') or []
    rows, detail = {}, []
    for d in truth['defects']:
        k = d['kind']
        r = rows.setdefault(k, {'n': 0, 'hit': 0, 'hit_type': 0})
        r['n'] += 1
        it = items.get(d['code']) if k != 'foreign' else None
        fhit = [f for f in foreign if f.get('bbox_xy') and _overlap(f['bbox_xy'], d['bbox_xy'])]
        if k == 'foreign':
            hit = hit_type = bool(fhit)
            # 落在某个子件上的异物，子件告警也算检出（类型不算对）
            for i in cmp['items']:
                ib = box_of(i)
                if ib and i.get('status') in ('alarm', 'missing') and _overlap(ib, d['bbox_xy']):
                    hit = True
        else:
            st = (it or {}).get('status')
            hit = st in ('alarm', 'missing') or bool(fhit)
            want = KIND_ALARM[k]
            got = _alarms(it or {})
            hit_type = hit if want is None else bool(got & want)
        r['hit'] += hit
        r['hit_type'] += hit_type
        detail.append({'id': d['id'], 'kind': k, 'code': d['code'], 'params': d['params'], 'hit': hit, 'hit_type': hit_type,
                       'status': (it or {}).get('status'), 'alarms': sorted(_alarms(it or {})),
                       'ncc': (it or {}).get('ncc'), 'shift_mm': (it or {}).get('shift_mm'),
                       'dtheta_deg': (it or {}).get('dtheta_deg'), 'foreign_hits': len(fhit)})
    for r in rows.values():
        r['recall'] = round(r['hit'] / r['n'], 3)
        r['recall_type'] = round(r['hit_type'] / r['n'], 3)
    n = sum(r['n'] for r in rows.values())
    boxes = [d['bbox_xy'] for d in truth['defects']] + [d['part_bbox_xy'] for d in truth['defects']]
    far = [i for i in cmp['items'] if i.get('status') in ('alarm', 'missing') and box_of(i)
           and not any(_overlap(box_of(i), b, NEAR_PX) for b in boxes)]
    far_f = [f for f in foreign if not any(_overlap(f['bbox_xy'], b, NEAR_PX) for b in boxes)]
    out = {'status': 'ok', 'trip': truth['trip'], 'n': n, 'hit': sum(r['hit'] for r in rows.values()),
           'hit_type': sum(r['hit_type'] for r in rows.values()), 'by_kind': rows, 'detail': detail,
           'synth_far_alarms': len(far) + len(far_f)}
    out['recall'] = round(out['hit'] / n, 3)
    out['recall_type'] = round(out['hit_type'] / n, 3)
    if base is not None:
        out['base_trip_alarms'] = sum(1 for i in base['items'] if i.get('status') in ('alarm', 'missing')) + len(base.get('foreign') or [])
        out['base_trip_status'] = _count(i.get('status') for i in base['items'])
        out['base_trip_alarm_types'] = _count(a['type'] for i in base['items'] for a in i.get('alarms') or [])
    return out


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# ------------------------------------------------------------------ 汇总

def score_all(synth_trip='T02S', bad_trip='T01B'):
    ev = io.ROOT / '标注' / '评测'
    mid = io.MID
    return {
        '质检': score_qc(_load(ev / '坏帧.json'), _load(mid / bad_trip / 'qc.json')),
        '标定': score_calib(_load(mid / 'T01' / 'calib.json'), _load(mid / 'T02' / 'calib.json')),
        '对齐': score_align(ev / '对应点.csv', _load(mid / 'T02' / 'align.json')),
        '分割': {'status': '缺', 'why': '没有训练好的分割权重和 T02 验证标注'},
        '身份': score_identity(_load(mid / 'T02' / 'parts.json')),
        '对比': score_compare(_load(ev / '合成缺陷.json'), _load(mid / synth_trip / 'compare.json'),
                            _load(mid / 'T02' / 'compare.json'), _load(mid / synth_trip / 'parts.json')),
    }


def table(s):
    def cell(k):
        r = s[k]
        if r['status'] != 'ok':
            return f"缺：{r['why']}"
        if k == '质检':
            by = '、'.join(f"{kk} {v['hit']}/{v['n']}" for kk, v in r['by_kind'].items())
            return f"检出 {r['hit']}/{r['n']}（{by}）；误报 {r['n_false']} 帧"
        if k == '标定':
            return f"sx {r['sx_mean'][0]}/{r['sx_mean'][1]} mm/px，sd {r['sd_pct'][0]}%/{r['sd_pct'][1]}%，两趟差 {r['trip_diff_pct']}%"
        if k == '对齐':
            return f"{r['n']} 对，中位 {r['median_px']} px，p95 {r['p95_px']} px"
        if k == '身份':
            return f"找到 {r['found']}/{r['n']}（{r['found_rate']:.1%}），{r['by_found_by']}；{r['code_check']}"
        if k == '对比':
            by = '、'.join(f"{kk} {v['hit']}/{v['n']}" for kk, v in r['by_kind'].items())
            base = f"；真实 T02 告警 {r['base_trip_alarms']}" if 'base_trip_alarms' in r else ''
            return (f"召回 {r['hit']}/{r['n']}（{by}），类型对 {r['hit_type']}/{r['n']}{base}"
                    f"；合成趟远离缺陷的告警 {r['synth_far_alarms']}")
        return json.dumps(r, ensure_ascii=False)
    lines = ['| 能力 | 结果 |', '|---|---|']
    for k in s:
        lines.append(f'| {k} | {cell(k)} |')
    return '\n'.join(lines)


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description='评测评分（设计.md §5）')
    ap.add_argument('--synth-trip', default='T02S')
    ap.add_argument('--bad-trip', default='T01B')
    ap.add_argument('--out', default=io.ROOT / '实验' / 'E5' / '评分.json')
    ap.add_argument('--json-only', action='store_true')
    a = ap.parse_args(argv)
    s = score_all(a.synth_trip, a.bad_trip)
    io.write_json(Path(a.out), s)
    print(json.dumps(s, ensure_ascii=False) if a.json_only else table(s))


if __name__ == '__main__':
    main()
