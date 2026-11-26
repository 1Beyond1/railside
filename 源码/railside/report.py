"""S7 每趟 HTML 报告：index.html + img/，用相对路径，浏览器直接打开。

直接读 数据/中间/<趟>/ 下各步 JSON 和模板（不依赖库）。缺哪步的 JSON，那一节写“未运行”。
给人看的图一律水平翻转，车头在右边（设计.md §2）。
"""
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import io
from .coords import H, AlignMap
from .db import rel, alarms_from_compare, box4, latest_template, load_template, load_trip_files, norm_ver, resolve_path, template_root
from .frames import Trip

PV_STEP = 8        # 整车_1-8.jpg 的缩小倍数
PV_DOWN = 4        # 报告里再缩小的倍数（合计 1/32）
CROP_H = 200       # 局部图显示高度上限
CROP_W = 360
CROP_MARGIN = 48   # 从原图取局部图时外框四周留的边（px）
RED = (0, 0, 255)
NOT_RUN = '<p class="nr">未运行</p>'


def esc(v):
    return html.escape(str(v))


def fmt(v):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.4g}'
    if isinstance(v, (list, dict)):
        return esc(json.dumps(v, ensure_ascii=False))
    return esc(v)


def table(rows, cols, heads=None):
    if not rows:
        return '<p>（空）</p>'
    th = ''.join(f'<th>{esc(h)}</th>' for h in (heads or cols))
    trs = ''.join('<tr>' + ''.join(f'<td>{fmt(r.get(c))}</td>' for c in cols) + '</tr>' for r in rows)
    return f'<table><tr>{th}</tr>{trs}</table>'


def kv(pairs):
    return '<table class="kv">' + ''.join(f'<tr><th>{esc(k)}</th><td>{fmt(v)}</td></tr>' for k, v in pairs) + '</table>'


# ---------- 局部图 ----------

class _Raw:
    """按趟取原图区域；原图不在时返回 None。"""

    def __init__(self):
        self.trips = {}

    def region(self, trip, box):
        if trip not in self.trips:
            try:
                self.trips[trip] = Trip.from_id(trip)
            except (OSError, KeyError, ValueError):
                self.trips[trip] = None
        t = self.trips[trip]
        if t is None or box is None:
            return None
        x0, y0, x1, y1 = box
        m = CROP_MARGIN
        ya, yb = max(0, int(y0) - m), min(H, int(y1) + m)
        xa = int(x0) - m
        try:
            im = t.region(xa, int(x1) + m, ya, yb)
        except (OSError, ValueError):
            return None
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im, (int(x0) - xa, int(y0) - ya), (int(x1) - xa - 1, int(y1) - ya - 1), RED, 2)
        return im


def _shown(im):
    """缩到显示尺寸并水平翻转。"""
    h, w = im.shape[:2]
    s = min(1.0, CROP_H / h, CROP_W / w)
    if s < 1:
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return im[:, ::-1]


def _crop_image(file_path, raw, trip, box):
    """优先用零件的局部图文件，没有再从原图按外框取。返回 BGR/灰度图或 None。"""
    if file_path is not None:
        try:
            return io.imread(file_path, cv2.IMREAD_COLOR)
        except IOError:
            pass
    return raw.region(trip, box)


# ---------- 整车预览 ----------

def preview_with_marks(man, preview, marks):
    """把 整车_1-8.jpg 按 train_panels 切成每节一行（车 1 在上，各行翻转、右对齐），标出 marks。

    marks：[(编号, (x0, y0, x1, y1) 趟坐标)]。返回 BGR 图。
    """
    x1t = man['train_x1']

    def col(x):  # 趟坐标 x → 整车_1-8.jpg 里的列（翻转后）
        return (x1t // PV_STEP - 1) - (int(x) // PV_STEP)

    panels = man.get('train_panels') or [{'car': 1, 'x0': man['train_x0'], 'x1': x1t}]
    segs = []
    for p in panels:
        ja, jb = max(0, col(p['x1'] - 1)), min(preview.shape[1] - 1, col(p['x0']))
        seg = preview[:, ja:jb + 1]
        seg = cv2.resize(seg, (max(1, seg.shape[1] // PV_DOWN), max(1, seg.shape[0] // PV_DOWN)), interpolation=cv2.INTER_AREA)
        segs.append((p, ja, seg))
    wmax = max(s.shape[1] for _, _, s in segs)
    bar = 26
    rows = []
    for p, ja, seg in segs:
        h, w = seg.shape[:2]
        off = wmax - w
        row = np.zeros((h + bar, wmax, 3), np.uint8)
        row[bar:, off:] = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) if seg.ndim == 2 else seg
        cv2.putText(row, f"car {p.get('car')}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for n, (bx0, by0, bx1, by1) in marks:
            if bx1 <= p['x0'] or bx0 >= p['x1']:
                continue
            ca = (col(min(bx1, p['x1']) - 1) - ja) // PV_DOWN + off
            cb = (col(max(bx0, p['x0'])) - ja) // PV_DOWN + off
            ya = int(by0) // (PV_STEP * PV_DOWN) + bar
            yb = int(by1) // (PV_STEP * PV_DOWN) + bar
            cv2.rectangle(row, (ca - 2, ya - 2), (max(cb, ca) + 2, yb + 2), RED, 2)
            cv2.putText(row, str(n), (max(0, ca - 4), 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        rows.append(row)
    return np.vstack(rows)


# ---------- 报告 ----------

CSS = """body{font-family:"Microsoft YaHei",sans-serif;margin:24px;max-width:1400px;color:#222}
h1{font-size:22px}h2{font-size:18px;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:28px}
table{border-collapse:collapse;margin:6px 0;font-size:13px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:left;vertical-align:top}
th{background:#f3f3f3}.kv th{width:160px}.nr{color:#a00;font-weight:bold}.note{color:#666;font-size:13px}
.alarm{border:1px solid #e0b0b0;margin:10px 0;padding:8px}.pair{display:flex;gap:16px;flex-wrap:wrap}
.pair figure{margin:0}.pair figcaption{font-size:12px;color:#555}img{border:1px solid #999}
.wide{overflow-x:auto}.wide img{max-width:100%}"""


def _alarm_entries(cmp_, parts_by_code, tpl_by_code):
    """把告警按零件归组：每个有告警的 item 一条，每块异物一条。"""
    groups = {}
    for a in alarms_from_compare(cmp_):
        key = ('item', a['item']) if a['item'] is not None else ('foreign', len(groups))
        bxy = (a.get('detail') or {}).get('bbox_xy')
        g = groups.setdefault(key, {'code': a['code'], 'alarms': [], 'bbox_uv': a['bbox_uv'], 'bbox_xy': bxy})
        g['alarms'].append(a)
    items = cmp_.get('items') or []
    out = []
    for (kind, i), g in groups.items():
        g['metrics'] = items[i] if kind == 'item' else {}
        g['part'] = parts_by_code.get(g['code'])
        g['tpl'] = tpl_by_code.get(g['code'])
        if g['bbox_uv'] is None:
            g['bbox_uv'] = (g['part'] or {}).get('bbox_uv') or (g['tpl'] or {}).get('bbox_uv')
        out.append(g)
    return out


def write_report(trip, out_dir=None, template_ver=None):
    io.validate_trip(trip)
    if out_dir:
        out = Path(out_dir)
    else:
        rep_root = (io.OUT / '报告').resolve()
        out = io.OUT / '报告' / trip
        try:
            out.resolve().relative_to(rep_root)
        except ValueError:
            raise ValueError(f'报告路径越界：{trip!r}') from None
        if out.resolve() == rep_root:
            raise ValueError(f'报告路径越界：{trip!r}')
    img = out / 'img'
    if img.exists():
        if img.is_symlink() or img.resolve().parent != out.resolve():
            raise ValueError(f'拒绝清理越界或符号链接目录：{img}')
        shutil.rmtree(img)
    img.mkdir(parents=True)
    js = load_trip_files(trip)
    tj, qc, man, cal, al, seg, parts, cmp_ = (js[k] for k in ('trip', 'qc', 'manifest', 'calib', 'align', 'seg', 'parts', 'compare'))
    tv = norm_ver((al or {}).get('template_ver')) or norm_ver(template_ver) or latest_template()
    tpl = load_template(tv)
    tdir = io.trip_dir(trip)
    raw = _Raw()
    amap = AlignMap.from_json(al) if al and len(al.get('g') or []) >= 2 else None
    head_x = (man or {}).get('head_x')

    def uv_to_xy(b):
        """模板外框 → 本趟外框。有 align 用映射，否则按车头平移估算（x = u + head_x）。"""
        if b is None:
            return None, None
        u0, v0, u1, v1 = box4(b)
        if amap is not None:
            xa, ya = amap.to_trip(u0, v0)
            xb, yb = amap.to_trip(u1, v1)
            return (float(xa), float(ya), float(xb), float(yb)), 'align'
        if head_x is not None:
            return (u0 + head_x, v0, u1 + head_x, v1), 'head_x'
        return None, None

    s = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{esc(trip)} 检测报告</title>'
         f'<style>{CSS}</style></head><body><h1>{esc(trip)} 检测报告</h1>']

    # 1 趟次与代码版本
    s.append('<h2>1 趟次与代码版本</h2>')
    s.append(kv([('趟次', trip), ('原图目录', (tj or man or {}).get('src_dir')), ('相机', (tj or qc or {}).get('camera')),
                 ('帧数', (tj or man or {}).get('n_frames')), ('模板版本', tv or '（无）'),
                 ('报告代码版本', io.code_ver()), ('报告生成时间', io.now())]))
    s.append('<p class="note">各步输出的来源（来龙去脉）：</p>')
    s.append(table([{'步骤': k, '文件': f'{k}.json', 'code_ver': (v or {}).get('code_ver') if v else '未运行',
                     'created_at': (v or {}).get('created_at')} for k, v in js.items()],
                   ['步骤', '文件', 'code_ver', 'created_at']))

    # 2 质检
    s.append('<h2>2 质检结论（S1）</h2>')
    if qc:
        s.append(kv([('结论', qc.get('verdict')), ('帧数', qc.get('frames')), ('缺帧', qc.get('missing')),
                     ('车内帧', qc.get('train_frames')), ('理由', '；'.join(qc.get('reasons') or []) or '无')]))
    else:
        s.append(NOT_RUN)

    # 3 train_panels
    s.append('<h2>3 train_panels（S2）</h2>')
    if man:
        s.append(kv([('车内范围 train_x0–train_x1', [man.get('train_x0'), man.get('train_x1')]), ('车头 head_x', head_x),
                     ('错缝数', man.get('bad_seams'))]))
        s.append(table(man.get('train_panels') or [], ['panel', 'car', 'x0', 'x1', 'width_px', 'height_px']))
    else:
        s.append(NOT_RUN)

    # 4 标定
    s.append('<h2>4 标定摘要（S3）</h2>')
    if cal:
        sx = [p['mm_per_px'] for p in cal.get('sx_curve') or [] if p.get('mm_per_px') is not None]
        s.append(kv([('尺寸来源', cal.get('source')), ('纵向 mm/px', cal.get('sy_mm_per_px')),
                     ('横向 mm/px 采样点数', len(sx)),
                     ('横向 mm/px 最小/中位/最大', [round(float(np.min(sx)), 5), round(float(np.median(sx)), 5),
                                                round(float(np.max(sx)), 5)] if sx else None)]))
        refs = cal.get('refs') or []
        if refs and all(isinstance(r, dict) for r in refs):
            cols = sorted({k for r in refs for k in r})
            s.append(table(refs, cols))
    else:
        s.append(NOT_RUN)

    # 5 对齐
    s.append('<h2>5 对齐摘要（S5）</h2>')
    if al:
        s.append(kv([('模板版本', al.get('template_ver')), ('定位靶数', len(al.get('anchors') or [])),
                     ('映射采样点数', len(al.get('g') or [])), ('残差 residual_px', al.get('residual_px')),
                     ('车头裁剪 u_trim0', al.get('u_trim0'))]))
    else:
        s.append(NOT_RUN)

    # 6 零件统计
    s.append('<h2>6 零件统计（S6b）</h2>')
    parts_by_code = {}
    if parts:
        inst = parts.get('instances') or []
        parts_by_code = {p.get('code'): p for p in inst}
        cnt = Counter(p.get('found_by') for p in inst)
        rows = [{'found_by': k, '个数': v} for k, v in sorted(cnt.items(), key=lambda kv_: str(kv_[0]))]
        rows.append({'found_by': '合计', '个数': len(inst)})
        s.append(table(rows, ['found_by', '个数']))
        if tpl:
            s.append(f'<p class="note">模板 {esc(tv)} 共 {len(tpl.get("parts") or [])} 个零件。</p>')
    else:
        s.append(NOT_RUN)

    # 7 告警
    s.append('<h2>7 告警（S8）</h2>')
    marks, n_alarm = [], None
    if cmp_:
        tpl_by_code = {p.get('code'): p for p in (tpl or {}).get('parts') or []}
        entries = _alarm_entries(cmp_, parts_by_code, tpl_by_code)
        n_alarm = len(entries)
        s.append(f'<p>对比 {len(cmp_.get("items") or [])} 个零件，告警 {len(entries)} 处'
                 f'（{sum(len(e["alarms"]) for e in entries)} 条）。左为模板局部图，右为本趟局部图，均已水平翻转。</p>')
        tdir_t = template_root() / tv if tv else None
        for n, e in enumerate(entries, 1):
            part, tp = e['part'] or {}, e['tpl'] or {}
            if part.get('bbox_xy'):
                bxy, how = box4(part['bbox_xy']), 'parts'
            elif e.get('bbox_xy') and len(e['bbox_xy']) == 4:
                bxy, how = box4(e['bbox_xy']), 'compare'
            else:
                bxy, how = uv_to_xy(e['bbox_uv'])
            if bxy is not None:
                marks.append((n, bxy))
            # 模板局部图
            t_box = None
            if tpl and e['bbox_uv'] is not None and tpl.get('head_x') is not None:
                u0, v0, u1, v1 = box4(e['bbox_uv'])
                t_box = (u0 + tpl['head_x'], v0, u1 + tpl['head_x'], v1)
            t_img = _crop_image(resolve_path(tp.get('crop'), *(b for b in (tdir_t, io.ROOT) if b)), raw,
                                tpl.get('trip') if tpl else None, t_box) if tpl else None
            c_img = _crop_image(resolve_path(part.get('crop'), tdir, io.ROOT), raw, trip, bxy)
            figs = []
            for tag, im, cap in (('tpl', t_img, f'模板 {tv or ""}（{(tpl or {}).get("trip", "")}）'), ('trip', c_img, f'本趟 {trip}')):
                if im is None:
                    figs.append(f'<figure><p class="note">无图</p><figcaption>{esc(cap)}</figcaption></figure>')
                    continue
                name = f'alarm{n:03d}_{tag}.jpg'
                io.imwrite(img / name, _shown(im))
                figs.append(f'<figure><img src="img/{name}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>')
            m = e['metrics']
            types = '、'.join(str(a['type']) for a in e['alarms'])
            vals = [f"{a['type']}={fmt(a['value'])}" + (f"（阈值 {fmt(a['threshold'])}）" if a['threshold'] is not None else '')
                    for a in e['alarms'] if a['value'] is not None]
            pos = f'趟坐标 x {bxy[0]:.0f}–{bxy[2]:.0f}' + ('（按车头估算）' if how == 'head_x' else '') if bxy else '位置未知'
            s.append(f'<div class="alarm"><b>#{n} {esc(e["code"] or "异物")}</b> 类型：{esc(types)}；'
                     f'{pos}；找到方式：{fmt(part.get("found_by"))}<br>'
                     f'NCC {fmt(m.get("ncc"))}，差异面积 {fmt(m.get("diff_area"))}，位移 mm {fmt(m.get("shift_mm"))}，'
                     f'转角 Δθ° {fmt(m.get("dtheta_deg"))}' + (f'；{esc("；".join(vals))}' if vals else '') +
                     f'<div class="pair">{"".join(figs)}</div></div>')
        if not entries:
            s.append('<p>无告警。</p>')
    else:
        s.append(NOT_RUN)

    # 8 整车预览
    s.append('<h2>8 整车预览与告警位置</h2>')
    pv = tdir / '预览' / f'整车_1-{PV_STEP}.jpg'
    if man and pv.is_file():
        im = preview_with_marks(man, io.imread(pv), marks)
        io.imwrite(img / 'preview.jpg', im, quality=85)
        s.append(f'<p class="note">每节车一行（车 1 在上），水平翻转，车头在右；1/{PV_STEP * PV_DOWN} 缩小。'
                 f'红框是告警位置，编号同第 7 节。</p><div class="wide"><img src="img/preview.jpg" alt="整车预览"></div>')
    else:
        s.append(NOT_RUN)

    s.append('</body></html>')
    (out / 'index.html').write_text('\n'.join(s), encoding='utf-8')
    return {'step': 'report', 'trip': trip, 'template_ver': tv, 'alarms': n_alarm,
            'missing': [f'{k}.json' for k, v in js.items() if v is None], 'out': rel(out / 'index.html')}


def add_parser(sub, step):
    p = sub.add_parser(step, help='S7 每趟 HTML 报告')
    p.add_argument('trip')
    p.add_argument('--out', default=None, help='默认 数据/成品/报告/<趟>')
    p.add_argument('--template', default=None, help='模板版本；默认取 align.json 的，没有时取最新的')
    p.set_defaults(func=run)


def run(args):
    out = None
    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else io.ROOT / out
    return write_report(args.trip, out, args.template)
