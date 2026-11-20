"""S9 智能体：一问一答、写巡检报告、MCP 服务。

工具（两种模式共用，每次调用都写一条 railside.db.append_agent_log）：
  qc_summary / align_summary / calib_summary   各步摘要
  alarms          告警列表（railside.query.alarms；库没建时退回 compare.json）
  part_history    零件编码的跨趟历史（railside.query.part_history）
  sql             只读 SQL（railside.query.sql）
  run_step        跑某一步（railside.pipeline）
  run_trip        跑整趟，原图 → 报告（railside.pipeline）
  report          出 HTML 报告（railside.report）

大模型后端可选：OpenAI 兼容 chat completions + tool calling。环境变量
RAILSIDE_LLM_BASE_URL、RAILSIDE_LLM_API_KEY、RAILSIDE_LLM_MODEL；没配走离线模式
（按关键词和编码正则把常见问法映射到工具，用模板拼回答）。本文件不直接调任何付费 API，
测试用 monkeypatch 假的 OpenAI 兼容响应。

命令：
  uv run railside agent --trip T02 --message "这趟有哪些告警"
  uv run railside agent --trip T02 --report
  uv run railside mcp           （stdio，接入方法见 agent_mcp.md）
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from . import db, io, pipeline, query, report

VERDICT_CN = {'pass': '通过', 'warn': '警告', 'fail': '失败'}
STATUS_CN = {'ok': '成功', 'partial': '部分完成', 'skipped': '跳过', 'failed': '失败', 'error': '出错'}
# 告警严重度：缺失和异物最要紧，其次是几何/外观，qc 只表示这一处不判。
SEVERITY = {'missing': 5, 'foreign': 4, 'dtheta': 3, 'shift': 3, 'area': 2, 'ncc': 2, 'qc': 1}
CODE_RE = re.compile(r'C\d+-[A-Za-z0-9]+-[A-Za-z0-9_]+-\d+')
ALARM_WORDS = ('告警', '报警', '异常', '问题', '缺陷', '缺失', '异物')
RUN_WORDS = ('跑', '运行', '重跑', '处理', '执行')
STEP_WORDS = {'质检': 'ingest', '收录': 'ingest', '拼接': 'stitch', '标定': 'calib', '对齐': 'align',
              '分割': 'segment', '身份': 'match', '匹配': 'match', '对比': 'compare', '入库': 'db',
              '报告': 'report'}
ALARM_TYPES = {'缺失': 'missing', '异物': 'foreign', '转角': 'dtheta', '位移': 'shift', '面积': 'area'}

SYSTEM = """你是地铁列车侧面视觉检测系统的助手。可以用工具查一趟的质检、对齐、标定、告警、零件历史，或跑某一步、跑整趟、出报告。
规矩：
- 只谈这趟数据的事，用中文，简短（3 句以内）；数字一律来自工具结果，不要编。
- 先调工具再回答，一次可以调多个；工具报错就说缺什么、该跑哪一步，不要假装有结果。
- 不自己写文件、不改数据；出报告和跑流程用对应工具。
"""

# ---------- 工具清单（名字、说明、参数 JSON Schema） ----------

TOOLS = {
    'qc_summary': {
        'desc': '一趟的质检结论（S1）：verdict、帧数、缺帧、车内帧范围、有标记的帧。',
        'params': {'type': 'object', 'properties': {
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'}}, 'required': []}},
    'align_summary': {
        'desc': '一趟的对齐摘要（S5）：定位靶找到数、残差中位/p95、局部横向比例、质检第二段标记。',
        'params': {'type': 'object', 'properties': {
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'}}, 'required': []}},
    'calib_summary': {
        'desc': '一趟的标定摘要（S3）：横向 mm/px 范围、纵向 mm/px、尺寸来源。',
        'params': {'type': 'object', 'properties': {
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'}}, 'required': []}},
    'alarms': {
        'desc': '告警列表，按严重度排序；可按趟、按类型（missing/foreign/ncc/shift/dtheta/area/qc）过滤。',
        'params': {'type': 'object', 'properties': {
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'},
            'type': {'type': 'string', 'description': '只看这一种告警'}}, 'required': []}},
    'part_history': {
        'desc': '一个零件编码在各趟的识别方式、匹配分、对比指标（NCC、差异面积、位移、转角）和告警。',
        'params': {'type': 'object', 'properties': {
            'code': {'type': 'string', 'description': '零件编码，如 C3-B1-BOLT-04'},
            'trip': {'type': 'string', 'description': '只看这一趟；不给就列各趟'}},
            'required': ['code']}},
    'sql': {
        'desc': '只读 SQL（SELECT 或 WITH 开头），查 数据/成品/库.sqlite，返回 columns 和 rows。',
        'params': {'type': 'object', 'properties': {
            'text': {'type': 'string', 'description': '一条 SELECT/WITH 语句'}}, 'required': ['text']}},
    'run_step': {
        'desc': '跑流程里的一步（ingest/stitch/calib/align/match/compare/db/report；segment 要权重）。',
        'params': {'type': 'object', 'properties': {
            'step': {'type': 'string', 'description': '步骤名'},
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'},
            'src': {'type': 'string', 'description': '原图目录（ingest 需要；默认取 trip.json）'}},
            'required': ['step']}},
    'run_trip': {
        'desc': '跑整趟：从原图质检跑到入库、出报告（一条命令，人不插手）。',
        'params': {'type': 'object', 'properties': {
            'src': {'type': 'string', 'description': '原图目录；默认取 trip.json 里记的'},
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'},
            'weights': {'type': 'string', 'description': '分割权重，可不给'}}, 'required': []}},
    'report': {
        'desc': '出这一趟的 HTML 报告（数据/成品/报告/<趟>/index.html）。',
        'params': {'type': 'object', 'properties': {
            'trip': {'type': 'string', 'description': '趟次号，默认当前趟'}}, 'required': []}},
}


def tool_specs():
    """OpenAI 兼容 tools 参数。"""
    return [{'type': 'function', 'function': {'name': k, 'description': v['desc'], 'parameters': v['params']}}
            for k, v in TOOLS.items()]


def _brief(obj, max_rows=20):
    """写日志用：长列表截断，记总数。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list) and len(v) > max_rows:
                out[k] = [_brief(x, max_rows) for x in v[:max_rows]]
                out[k + '_total'] = len(v)
            else:
                out[k] = _brief(v, max_rows) if isinstance(v, (dict, list)) else v
        return out
    if isinstance(obj, list):
        return [_brief(x, max_rows) if isinstance(x, (dict, list)) else x for x in obj[:max_rows]]
    return obj


def _num(v, fmt='{:.3g}'):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return fmt.format(v)
    return str(v)


class Tools:
    """一趟的工具集。call() 统一执行、留痕、把异常变成 {'error': ...}。"""

    def __init__(self, trip=None, weights=None, template=None):
        self.trip = trip
        self.weights = weights
        self.template = template
        self.calls = []

    # ---------- 小工具 ----------
    def _trip(self, trip=None):
        t = trip or self.trip
        if not t:
            raise ValueError('没有指定趟次')
        if not io.trip_dir(t).is_dir():
            raise FileNotFoundError(f'没有趟次 {t}（缺 数据/中间/{t}/），先跑 railside run 或 ingest')
        return t

    def _json(self, trip, name):
        p = io.trip_dir(trip) / f'{name}.json'
        return io.read_json(p) if p.is_file() else None

    def _src_of(self, trip):
        meta = self._json(trip, 'trip') or {}
        src = meta.get('src_dir')
        if not src:
            return None
        p = Path(src)
        return str(p if p.is_absolute() else io.ROOT / p)

    # ---------- 工具 ----------
    def qc_summary(self, trip=None):
        t = self._trip(trip)
        qc = self._json(t, 'qc')
        if qc is None:
            return {'trip': t, 'found': False, 'message': '还没有 qc.json，先跑 ingest'}
        flagged = [{'f': p.get('f'), 'flags': p.get('flags')} for p in qc.get('per_frame') or [] if p.get('flags')]
        return {'trip': t, 'found': True, 'verdict': qc.get('verdict'), 'frames': qc.get('frames'),
                'missing': qc.get('missing') or [], 'train_frames': qc.get('train_frames'),
                'reasons': qc.get('reasons') or [], 'n_flagged': len(flagged), 'flagged': flagged[:20]}

    def align_summary(self, trip=None):
        t = self._trip(trip)
        al = self._json(t, 'align')
        if al is None:
            return {'trip': t, 'found': False, 'message': '还没有 align.json，先跑 align'}
        flags = [{'u': q.get('u'), 'ncc': q.get('ncc'), 'blur_ratio': q.get('blur_ratio'), 'flags': q.get('flags')}
                 for q in al.get('qc2') or [] if q.get('flags')]
        return {'trip': t, 'found': True, 'template_ver': al.get('template_ver'),
                'anchors_found': al.get('anchors_found'), 'anchors_total': al.get('anchors_total'),
                'residual_px': al.get('residual_px'), 'scale_range': al.get('scale_range'),
                'offset_range_px': al.get('offset_range_px'), 'head_protect': al.get('head_protect'),
                'qc2_flagged': al.get('qc2_flagged', len(flags)), 'qc2_flags': flags[:20]}

    def calib_summary(self, trip=None):
        t = self._trip(trip)
        cal = self._json(t, 'calib')
        if cal is None:
            return {'trip': t, 'found': False, 'message': '还没有 calib.json（S3 标定还没跑或模块还没实现）'}
        sx = sorted(p['mm_per_px'] for p in cal.get('sx_curve') or [] if p.get('mm_per_px') is not None)
        return {'trip': t, 'found': True, 'source': cal.get('source'), 'sy_mm_per_px': cal.get('sy_mm_per_px'),
                'n_samples': len(sx), 'sx_min': sx[0] if sx else None,
                'sx_median': sx[len(sx) // 2] if sx else None, 'sx_max': sx[-1] if sx else None,
                'n_refs': len(cal.get('refs') or []), 'refs': (cal.get('refs') or [])[:10]}

    def alarms(self, trip=None, type=None):
        t = self._trip(trip)
        try:
            rows = query.alarms(t, type)
            source = '库（railside.query）'
        except FileNotFoundError:
            rows, source = self._alarms_from_json(t, type), 'compare.json'
            if rows is None:
                return {'trip': t, 'source': None, 'n': 0, 'alarms': [],
                        'message': '还没有告警来源：库没建（railside db build）、compare.json 也没有'}
        if not rows and self._json(t, 'compare') is None:
            return {'trip': t, 'source': source, 'n': 0, 'alarms': [],
                    'message': '还没有对比结果（compare.json 不存在），不能断定没有告警'}
        rows = sorted(rows, key=lambda r: (-SEVERITY.get(r.get('type'), 2), str(r.get('code') or '')))
        brief = [{'trip': r.get('trip_id'), 'code': r.get('code'), 'type': r.get('type'),
                  'value': r.get('value'), 'threshold': r.get('threshold'), 'ncc': r.get('ncc'),
                  'diff_area': r.get('diff_area'), 'shift_mm': r.get('shift_mm'), 'dtheta_deg': r.get('dtheta_deg'),
                  'u0': r.get('u0'), 'v0': r.get('v0')} for r in rows]
        return {'trip': t, 'source': source, 'n': len(brief), 'alarms': brief}

    def _alarms_from_json(self, trip, type=None):
        cmp_ = self._json(trip, 'compare')
        if cmp_ is None:
            return None
        items = cmp_.get('items') or []
        out = []
        for a in db.alarms_from_compare(cmp_):
            if type and a['type'] != type:
                continue
            m = items[a['item']] if a['item'] is not None and a['item'] < len(items) else {}
            b = a['bbox_uv'] or [None] * 4
            out.append({'trip_id': trip, 'code': a['code'], 'type': a['type'], 'value': a['value'],
                        'threshold': a['threshold'], 'ncc': m.get('ncc'), 'diff_area': m.get('diff_area'),
                        'shift_mm': m.get('shift_mm'), 'dtheta_deg': m.get('dtheta_deg'),
                        'u0': b[0], 'v0': b[1]})
        return out

    def part_history(self, code, trip=None):
        code = (code or '').strip()
        h = query.part_history(code, trip=trip)
        tpl = (h.get('template') or [None])[0] or {}
        trips = []
        for t in h.get('trips') or []:
            inst = (t.get('instances') or [{}])[0]
            cmp_ = (t.get('comparisons') or [{}])[0]
            trips.append({'trip': t.get('trip_id'), 'found_by': inst.get('found_by'),
                          'match_score': inst.get('match_score'), 'ncc': cmp_.get('ncc'),
                          'diff_area': cmp_.get('diff_area'), 'shift_mm': cmp_.get('shift_mm'),
                          'dtheta_deg': cmp_.get('dtheta_deg'),
                          'alarms': [a.get('type') for a in t.get('alarms') or []]})
        names = [r.get('name') for r in h.get('part_type') or [] if r.get('name')]
        return {'code': code, 'type': tpl.get('type'), 'type_name': names[0] if names else None,
                'car': tpl.get('car'), 'zone': tpl.get('zone'), 'parent': tpl.get('parent'),
                'in_template': bool(h.get('template')), 'trips': trips}

    def sql(self, text):
        return query.sql(text)

    def run_step(self, step, trip=None, src=None):
        t = self._trip(trip)
        if step == 'ingest' and not src:
            src = self._src_of(t)
        r = pipeline.run_step(step, pipeline.step_argv(step, t, src=src, weights=self.weights, template=self.template))
        head = {k: r.get(k) for k in ('step', 'status', 'seconds', 'reason')}
        return {**head, 'summary': r.get('summary')}

    def run_trip(self, src=None, trip=None, weights=None):
        t = self._trip(trip)
        src = src or self._src_of(t)
        if not src:
            raise ValueError(f'没给原图目录，{t} 的 trip.json 里也没有 src_dir')
        return pipeline.run_trip(Path(src), t, weights or self.weights, self.template)

    def report(self, trip=None):
        return report.write_report(self._trip(trip))

    # ---------- 统一执行与留痕 ----------
    def call(self, name, args=None):
        spec = TOOLS.get(name)
        if spec is None:
            raise ValueError(f'没有这个工具：{name}')
        args = dict(args or {})
        allowed = set(spec['params']['properties'])
        unknown = set(args) - allowed
        args = {k: v for k, v in args.items() if k in allowed}
        t0 = time.time()
        try:
            if unknown:
                raise ValueError(f'参数不认识：{sorted(unknown)}（可用：{sorted(allowed)}）')
            result = getattr(self, name)(**args)
            status = result.get('status', 'ok') if name in ('run_trip', 'run_step') else 'ok'
        except Exception as e:  # 工具出错也要接着问，把错误交给模型/模板
            result = {'error': f'{type(e).__name__}: {e}'}
            status = 'error'
        rec = {'tool': name, 'args': args, 'status': status, 'seconds': round(time.time() - t0, 2)}
        self.calls.append(rec)
        log_trip = self.trip or args.get('trip') or '_mcp'
        db.append_agent_log(log_trip, 'agent', name, args, _brief(result), status)
        return result


# ---------- 离线意图路由 ----------

def route(message, trip=None):
    """把一句话映射成 [(工具, 参数)]。离线模式用，也可单独测。"""
    m = (message or '').strip()
    low = m.lower()
    if re.match(r'^\s*(select|with)\b', low):
        return [('sql', {'text': m})]
    code = CODE_RE.search(m)
    if code:
        whole = any(w in m for w in ('两趟', '历史', '各趟', '两次', '跨趟'))
        return [('part_history', {'code': code.group(0), **({} if whole else {'trip': trip})})]
    if any(w in m for w in RUN_WORDS):
        if any(w in m for w in ('整趟', '全流程', '一条命令', '从头', '整个流程')):
            return [('run_trip', {'trip': trip})]
        for w, s in STEP_WORDS.items():
            if w in m:
                return [('run_step', {'step': s, 'trip': trip})]
    if '报告' in m and any(w in m for w in ('写', '生成', '出', '做', '看看')):
        return [('report', {'trip': trip})]
    if any(w in m for w in ALARM_WORDS):
        args = {'trip': trip}
        for w, tp in ALARM_TYPES.items():
            if w in m:
                args['type'] = tp
                break
        return [('alarms', args)]
    if '质检' in m or '质量' in m:
        return [('qc_summary', {'trip': trip})]
    if '对齐' in m:
        return [('align_summary', {'trip': trip})]
    if '标定' in m or 'mm/px' in low or '比例' in m:
        return [('calib_summary', {'trip': trip})]
    return [('qc_summary', {'trip': trip}), ('align_summary', {'trip': trip}), ('alarms', {'trip': trip})]


# ---------- 离线回答 ----------

def _ans_qc(r):
    if not r.get('found'):
        return f"{r['trip']} 还没有质检结果：{r.get('message', '先跑 ingest')}。"
    s = f"{r['trip']} 质检：{VERDICT_CN.get(r['verdict'], r['verdict'])}，共 {r.get('frames')} 帧"
    tf = r.get('train_frames') or []
    if len(tf) == 2 and tf[0]:
        s += f"，列车在第 {tf[0]}–{tf[1]} 帧"
    s += '；' + ('无缺帧' if not r.get('missing') else f"缺帧 {r['missing']}")
    if r.get('reasons'):
        s += '；理由：' + '；'.join(str(x) for x in r['reasons'][:5])
    if r.get('n_flagged'):
        s += f"；{r['n_flagged']} 帧有异常标记"
    return s + '。'


def _ans_align(r):
    if not r.get('found'):
        return f"{r['trip']} 还没有对齐结果：{r.get('message', '先跑 align')}。"
    res = r.get('residual_px') or {}
    s = f"{r['trip']} 对齐到模板 {r.get('template_ver')}：定位靶 {r.get('anchors_found')}/{r.get('anchors_total')}"
    if res:
        s += f"，残差中位 {res.get('median')} px、p95 {res.get('p95')} px（n={res.get('n')}）"
    sr = r.get('scale_range')
    if sr:
        s += f"，局部横向比例 {sr[0]}–{sr[1]}"
    if r.get('qc2_flagged'):
        s += f"；质检第二段标记 {r['qc2_flagged']} 处"
    return s + '。'


def _ans_calib(r):
    if not r.get('found'):
        return f"{r['trip']} 还没有标定结果：{r.get('message', '先跑 calib')}。"
    s = f"{r['trip']} 标定：横向 {r.get('sx_min')}–{r.get('sx_max')} mm/px"
    if r.get('n_samples'):
        s += f"（中位 {r.get('sx_median')}，{r['n_samples']} 个采样点）"
    if r.get('sy_mm_per_px') is not None:
        s += f"，纵向 {r.get('sy_mm_per_px')} mm/px"
    if r.get('source'):
        s += f"；来源：{r.get('source') if isinstance(r['source'], str) else json.dumps(r['source'], ensure_ascii=False)}"
    return s + '。'


def _ans_alarms(r):
    if not r.get('n'):
        if r.get('message'):
            return f"{r['trip']}：{r['message']}。"
        return f"{r['trip']} 没有告警。"
    parts = []
    for a in r['alarms'][:12]:
        who = a.get('code') or '异物'
        bits = [str(a.get('type'))]
        if a.get('value') is not None:
            bits.append(f"值 {a['value']}")
        if a.get('threshold') is not None:
            bits.append(f"阈值 {a['threshold']}")
        parts.append(f"{who}（{'，'.join(bits)}）")
    s = f"{r['trip']} 共 {r['n']} 条告警：" + '；'.join(parts)
    if r['n'] > 12:
        s += f"……还有 {r['n'] - 12} 条"
    return s + '。'


def _ans_part(r):
    if not r.get('trips'):
        return f"{r['code']} 在各趟里都没有记录（模板和本趟识别里都没有）。"
    head = r['code'] + (f"（{r['type_name']}）" if r.get('type_name') else '')
    lines = [f'{head}：']
    for t in r['trips']:
        seg = f"- {t['trip']}：{t.get('found_by') or '—'}"
        ms = _num(t.get('match_score'))
        if ms:
            seg += f"，匹配 {ms}"
        ncc = _num(t.get('ncc'))
        if ncc:
            seg += f"，NCC {ncc}"
        sm = _num(t.get('shift_mm'))
        if sm:
            seg += f"，位移 {sm} mm"
        dt = _num(t.get('dtheta_deg'))
        if dt:
            seg += f"，转角 {dt}°"
        seg += '，告警 ' + ('无' if not t['alarms'] else '、'.join(str(a) for a in t['alarms']))
        lines.append(seg)
    return '\n'.join(lines)


def _ans_sql(r):
    rows, cols = r.get('rows') or [], r.get('columns') or []
    if not rows:
        return f"查询返回 0 行（列：{'、'.join(cols) or '—'}）。"
    head = '；'.join('，'.join(str(c) for c in row) for row in rows[:5])
    return f"查询返回 {len(rows)} 行（列：{'、'.join(cols)}）：{head}" + ('……' if len(rows) > 5 else '') + '。'


def _ans_run(r):
    n = {k: sum(1 for s in r['steps'] if s['status'] == k) for k in ('ok', 'skipped', 'failed')}
    s = (f"{r['trip']} 处理{STATUS_CN.get(r.get('status'), r.get('status', ''))}："
         f"{len(r['steps'])} 步里成功 {n['ok']}、跳过 {n['skipped']}、"
         f"失败 {n['failed']}，用时 {r.get('elapsed_s')} s")
    if r.get('report'):
        s += f"；报告 {r['report']}"
    if r.get('reason'):
        s += f"；原因：{r['reason']}"
    skip = [x for x in r['steps'] if x['status'] == 'skipped']
    if skip:
        s += '；跳过 ' + '、'.join(f"{x['step']}（{x.get('reason')}）" for x in skip)
    return s + '。'


def _ans_step(r):
    s = f"{r.get('step')}：{STATUS_CN.get(r.get('status'), r.get('status'))}，用时 {r.get('seconds')} s"
    if r.get('reason'):
        s += f"（{r['reason']}）"
    return s + '。'


def _ans_report(r):
    return f"报告已生成：{r.get('out')}（{r.get('alarms')} 处告警）。"


_ANSWERS = {'qc_summary': _ans_qc, 'align_summary': _ans_align, 'calib_summary': _ans_calib,
            'alarms': _ans_alarms, 'part_history': _ans_part, 'sql': _ans_sql,
            'run_step': _ans_step, 'run_trip': _ans_run, 'report': _ans_report}


def answer(name, result):
    """工具结果 → 一句话（离线模式；在线模式由模型写）。"""
    if isinstance(result, dict) and result.get('error'):
        return f"{name} 出错：{result['error']}"
    fn = _ANSWERS.get(name)
    return fn(result) if fn else json.dumps(result, ensure_ascii=False, default=str)[:800]


def offline_answer(tools, message):
    return '\n'.join(answer(name, tools.call(name, args)) for name, args in route(message, tools.trip))


# ---------- 在线模式（OpenAI 兼容） ----------

def llm_config():
    base = (os.environ.get('RAILSIDE_LLM_BASE_URL') or '').rstrip('/')
    model = os.environ.get('RAILSIDE_LLM_MODEL') or ''
    if not base or not model:
        return None
    return {'base': base, 'model': model, 'key': os.environ.get('RAILSIDE_LLM_API_KEY') or ''}


def _post(url, payload, key, timeout=60):
    """POST JSON，返回解析后的响应。测试里 monkeypatch 它（不发真实请求）。"""
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def _chat(cfg, messages):
    payload = {'model': cfg['model'], 'messages': messages, 'temperature': 0, 'tools': tool_specs()}
    res = _post(cfg['base'] + '/chat/completions', payload, cfg['key'])
    msg = ((res.get('choices') or [{}])[0] or {}).get('message') or {}
    return {'content': msg.get('content') or '', 'tool_calls': msg.get('tool_calls') or []}


def chat_loop(tools, message, max_rounds=6):
    """tool calling 循环：模型要工具就跑、结果回给它，直到给出文字回答。"""
    cfg = llm_config()
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': message}]
    try:
        for _ in range(max_rounds):
            msg = _chat(cfg, messages)
            if not msg['tool_calls']:
                return msg['content'] or '（模型没有给出文字回答）'
            messages.append({'role': 'assistant', 'content': msg['content'], 'tool_calls': msg['tool_calls']})
            for tc in msg['tool_calls']:
                fn = tc.get('function') or {}
                try:
                    args = json.loads(fn.get('arguments') or '{}')
                except ValueError:
                    args = {}
                result = tools.call(fn.get('name'), args if isinstance(args, dict) else {})
                messages.append({'role': 'tool', 'tool_call_id': tc.get('id') or fn.get('name') or 'call',
                                 'content': json.dumps(result, ensure_ascii=False, default=io._jsonable)[:4000]})
        return '（工具调用轮次用完，没有得出最终回答）'
    except Exception as e:  # 网络、鉴权、响应格式：退回离线，不让人干等
        note = f'（大模型调用失败：{type(e).__name__}: {e}；改用离线回答）'
        if not tools.calls:
            return note + '\n' + offline_answer(tools, message)
        return note + '\n上次工具结果：' + '；'.join(f"{c['tool']} {c['status']}" for c in tools.calls)


def ask(trip, message, weights=None, template=None):
    """一问一答。返回 {reply, tools, mode}。"""
    tools = Tools(trip, weights, template)
    if llm_config():
        reply = chat_loop(tools, message)
        mode = '在线'
    else:
        reply = offline_answer(tools, message)
        mode = '离线'
    return {'reply': reply, 'tools': tools.calls, 'mode': mode}


# ---------- 巡检报告 ----------

def _report_facts(tools, trip):
    return {'qc': tools.call('qc_summary', {'trip': trip}),
            'align': tools.call('align_summary', {'trip': trip}),
            'calib': tools.call('calib_summary', {'trip': trip}),
            'alarms': tools.call('alarms', {'trip': trip})}


def _md_report(trip, facts, mode='离线'):
    qc, al, cal, alarms = facts['qc'], facts['align'], facts['calib'], facts['alarms']
    L = [f'# {trip} 智能体巡检报告', '',
         f"- 生成时间：{io.now()}",
         f"- 代码版本：{io.code_ver()}",
         f"- 模式：{mode}（按规则汇总；配了 RAILSIDE_LLM_* 时由模型撰写）", '',
         '## 1 质检（S1）']
    if qc.get('found'):
        L.append(f"- 结论：{VERDICT_CN.get(qc['verdict'], qc['verdict'])}，{qc['frames']} 帧，"
                 f"车内第 {(qc.get('train_frames') or [None, None])[0]}–{(qc.get('train_frames') or [None, None])[1]} 帧")
        L.append(f"- 缺帧：{qc['missing'] or '无'}；有标记的帧：{qc['n_flagged'] or '无'}")
        for r in qc.get('reasons') or []:
            L.append(f"- 理由：{r}")
    else:
        L.append('- 没有 qc.json（S1 未运行）')
    L += ['', '## 2 对齐质量（S5）']
    if al.get('found'):
        res = al.get('residual_px') or {}
        L.append(f"- 模板 {al.get('template_ver')}；定位靶 {al.get('anchors_found')}/{al.get('anchors_total')}")
        if res:
            L.append(f"- 残差：中位 {res.get('median')} px、p95 {res.get('p95')} px、最大 {res.get('max')} px（n={res.get('n')}）")
        if al.get('scale_range'):
            L.append(f"- 局部横向比例 {al['scale_range'][0]}–{al['scale_range'][1]}；"
                     f"横向偏移 {al.get('offset_range_px')}")
        L.append(f"- 质检第二段标记 {al.get('qc2_flagged', 0)} 处")
    else:
        L.append('- 没有 align.json（S5 未运行）')
    L += ['', '## 3 标定（S3）']
    if cal.get('found'):
        L.append(f"- 横向 {cal.get('sx_min')}–{cal.get('sx_max')} mm/px（中位 {cal.get('sx_median')}），"
                 f"纵向 {cal.get('sy_mm_per_px')} mm/px")
        L.append(f"- 来源：{cal.get('source')}；用了 {cal.get('n_refs')} 个已知尺寸")
    else:
        L.append('- 没有 calib.json（S3 未运行）：毫米结论只能按粗估比例算，见「局限」')
    L += ['', '## 4 告警（按严重度）']
    rows = alarms.get('alarms') or []
    if rows:
        L.append('| # | 编码 | 类型 | 数值 | 阈值 | 位置 u0,v0 | NCC |')
        L.append('|---|---|---|---|---|---|---|')
        for i, a in enumerate(rows, 1):
            L.append(f"| {i} | {a.get('code') or '异物'} | {a.get('type')} | {a.get('value')} | "
                     f"{a.get('threshold')} | {a.get('u0')},{a.get('v0')} | {a.get('ncc')} |")
    else:
        L.append(f"- {alarms.get('message') or '无告警'}")
    L += ['', '## 5 需人工复核']
    todo = [a for a in rows if a.get('type') == 'missing']
    if todo:
        L.append('- 缺失候选：' + '、'.join(str(a.get('code')) for a in todo) + '（先看该处遮挡和 qc 标记）')
    if al.get('found') and al.get('qc2_flags'):
        L.append('- 对齐质检第二段标记的 u 位置：' +
                 '、'.join(f"{q.get('u')}（{'/'.join(q.get('flags') or [])}）" for q in al['qc2_flags'][:10]))
    strong = [a for a in rows if SEVERITY.get(a.get('type'), 0) >= 3]
    if strong:
        L.append('- 严重度较高的：' + '、'.join(f"{a.get('code') or '异物'}/{a.get('type')}" for a in strong[:10]))
    if not (todo or (al.get('found') and al.get('qc2_flags')) or strong):
        L.append('- 没有需要复核的位置。')
    L += ['', '## 6 局限']
    if not cal.get('found'):
        L.append('- 没有标定（S3）：位移、尺寸的毫米数按粗估的横向比例折算，绝对值不可信。')
    if not alarms.get('source'):
        L.append('- 没有对比结果（S8）：这里只做了质检和对齐，没有零件级告警。')
    L.append('- 真实两趟没有已知故障，列出的告警都要当误报候选看，得靠人工复核或合成缺陷评测判定。')
    L.append('- 只看相机这一侧的走行部，车内侧、顶部和另一侧不在范围。')
    return '\n'.join(L) + '\n'


def write_agent_report(trip, template=None):
    """写 数据/成品/报告/<趟>/智能体报告.md。返回 {path, tools, mode}。"""
    tools = Tools(trip, None, template)
    facts = _report_facts(tools, trip)
    cfg = llm_config()
    text = None
    if cfg:
        try:
            prompt = ('下面是一趟地铁列车侧面检测的事实（JSON）。请写一份中文 Markdown 巡检报告，'
                      '分节：质检、对齐质量、标定、告警（按严重度排序，列成表）、需人工复核的位置、局限。'
                      '只用给定数字，不要编；没有的步骤写明未运行。\n\n'
                      + json.dumps(facts, ensure_ascii=False, default=io._jsonable))
            msg = _chat(cfg, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}])
            text = msg['content'] or None
        except Exception:
            text = None
    mode = '在线' if text else '离线'
    if not text:
        text = _md_report(trip, facts, mode)
    out = io.OUT / '报告' / trip / '智能体报告.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    return {'path': out, 'tools': tools.calls, 'mode': mode}


# ---------- MCP ----------

def mcp_server():
    """MCP 服务：把同一套工具暴露给 Claude Code 这类客户端（stdio）。"""
    from mcp.server.mcpserver import MCPServer

    srv = MCPServer('railside', instructions='地铁列车侧面视觉检测：查质检、对齐、标定、告警、零件历史，跑步骤、跑整趟、出报告。'
                                             '坐标用趟坐标 (x, y)，不翻转；模板坐标 u 从车头起。')
    tools = Tools()

    def call(name, **args):
        return json.dumps(tools.call(name, args), ensure_ascii=False, default=io._jsonable)

    @srv.tool(description=TOOLS['qc_summary']['desc'])
    def qc_summary(trip: str | None = None) -> str:
        return call('qc_summary', trip=trip)

    @srv.tool(description=TOOLS['align_summary']['desc'])
    def align_summary(trip: str | None = None) -> str:
        return call('align_summary', trip=trip)

    @srv.tool(description=TOOLS['calib_summary']['desc'])
    def calib_summary(trip: str | None = None) -> str:
        return call('calib_summary', trip=trip)

    @srv.tool(description=TOOLS['alarms']['desc'])
    def alarms(trip: str | None = None, type: str | None = None) -> str:
        return call('alarms', trip=trip, type=type)

    @srv.tool(description=TOOLS['part_history']['desc'])
    def part_history(code: str, trip: str | None = None) -> str:
        return call('part_history', code=code, trip=trip)

    @srv.tool(description=TOOLS['sql']['desc'])
    def sql(text: str) -> str:
        return call('sql', text=text)

    @srv.tool(description=TOOLS['run_step']['desc'])
    def run_step(step: str, trip: str | None = None, src: str | None = None) -> str:
        return call('run_step', step=step, trip=trip, src=src)

    @srv.tool(description=TOOLS['run_trip']['desc'])
    def run_trip(src: str | None = None, trip: str | None = None, weights: str | None = None) -> str:
        return call('run_trip', src=src, trip=trip, weights=weights)

    @srv.tool(name='report', description=TOOLS['report']['desc'])
    def report_tool(trip: str | None = None) -> str:
        return call('report', trip=trip)

    return srv


def run_mcp(args):
    mcp_server().run()
    return None


# ---------- 命令行 ----------

def add_parser(sub, step):
    if step == 'mcp':
        p = sub.add_parser(step, help='MCP 服务（stdio）：把智能体的工具暴露给 Claude Code 等客户端')
        p.set_defaults(func=run_mcp)
        return
    p = sub.add_parser(step, help='S9 智能体：一问一答；--report 写巡检报告')
    p.add_argument('--trip', required=True, help='趟次号，如 T02')
    p.add_argument('--message', '-m', default=None, help='问题，如「这趟有哪些告警」')
    p.add_argument('--report', action='store_true', help='写 数据/成品/报告/<趟>/智能体报告.md')
    p.add_argument('--weights', default=None, help='分割权重（跑 segment 或整趟时用）')
    p.add_argument('--template', default=None, help='模板版本；默认 数据/成品/模板 下最新的')
    p.set_defaults(func=run)


def run(args):
    trip = args.trip
    if args.report:
        rep = write_agent_report(trip, args.template)
        path = io.rel(rep['path'])
        return {'reply': f'巡检报告已生成：{path}（{rep["mode"]}模式）', 'tools': rep['tools'], 'report': path}
    if not args.message:
        raise SystemExit('agent: 要给 --message，或用 --report')
    r = ask(trip, args.message, args.weights, args.template)
    return {'reply': r['reply'], 'tools': r['tools'], 'mode': r['mode']}
