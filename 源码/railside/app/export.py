"""离线浏览包：把应用要用的接口结果、长图瓦片、零件局部图导成静态文件。

导出后不需要 Python、不需要原图：双击包里的 index.html，浏览器直接打开（file://）。
只读：长图浏览、零件框、零件详情（两趟局部图并排）、告警、质检与对齐；不能处理新趟、不能对话。

浏览器在 file:// 下不让 fetch 读本地 JSON，所以接口结果写成一个脚本 data/data.js（window.RS_STATIC），
前端检测到它就不走网络（static/app.js 的 OFF）。瓦片和局部图是普通 <img>，file:// 下能读。

长图只导到 --max-level 层（默认比原分辨率低一层 = 半分辨率）：全分辨率约 8000 张瓦片、几百 MB；
零件细节看详情里的局部图（按原分辨率裁）。前端用自定义瓦片源，把瓦片边长按倍数放大，坐标不变。

用法：uv run railside app --export <输出目录> [--trip T02 ...] [--max-level 18]
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import io
from . import data, dzi

STATIC = Path(__file__).resolve().parent / 'static'
QUALITY = 80


def export(out, trips=None, max_level=None, log=None):
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'输出目录不是空的：{out}（换个目录，或先删掉）')
    all_trips = data.list_trips()
    names = trips or [t['trip'] for t in all_trips if data.trip_file(t['trip'], 'parts.json')]
    if not names:
        raise SystemExit('没有可导出的趟次：先跑完 match（要有 parts.json）')
    api, tiles, n_tiles, n_crops = {}, {}, 0, 0
    api['/api/trips'] = {'trips': [t for t in all_trips if t['trip'] in names]}
    for trip in names:
        info = data.trip_info(trip)
        api[f'/api/trips/{trip}'] = info
        api[f'/api/trips/{trip}/overlay'] = ov = data.overlay(trip)
        api[f'/api/trips/{trip}/alarms'] = data.alarms(trip)
        api[f'/api/trips/{trip}/summary'] = data.summary(trip)
        d = out / 'data' / trip
        # 零件详情和两侧局部图
        codes = sorted({b['code'] for b in ov['boxes'] if b.get('code')})
        for code in codes:
            det = data.part_detail(trip, code)
            api[f'/api/trips/{trip}/parts/{code}'] = det
            for side in ('template', 'trip'):
                if det['crops'][side]:
                    _write(d / 'crops' / f'{code}_{side}.jpg', data.crop(trip, code, side))
                    n_crops += 1
        log(f'{trip}: {len(codes)} 个零件详情、{n_crops} 张局部图')
        # 长图瓦片 0..max_level 层
        p = dzi.pyramid(trip)
        top = max(0, p.max_level - 1) if max_level is None else min(max_level, p.max_level)
        k = 1 << (p.max_level - top)       # 前端瓦片边长、重叠都放大 k 倍，坐标仍按原分辨率
        for level in range(top + 1):
            _, wl, hl = p.level(level)
            cols, rows = -(-wl // dzi.TILE), -(-hl // dzi.TILE)
            for c in range(cols):                     # 按列走：同一批原图帧连续读，帧缓存命中
                for r in range(rows):
                    im = dzi.render(p, level, c, r)
                    ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(im), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                    if not ok:
                        raise IOError(f'瓦片编码失败：{trip} {level}/{c}_{r}')
                    _write(d / 'tiles' / str(level) / f'{c}_{r}.jpg', buf.tobytes())
                    n_tiles += 1
            log(f'{trip}: 第 {level} 层 {cols}×{rows} 张瓦片')
        tiles[trip] = {'width': p.width, 'height': p.height, 'maxLevel': top,
                       'tileSize': dzi.TILE * k, 'overlap': dzi.OVERLAP * k}
        ovw = dzi.overview_bytes(trip, 4000, 112)
        _write(d / 'overview.jpg', ovw)
    # 前端：页面、样式、脚本、OpenSeadragon；路径改成相对路径，先载入 data.js
    shutil.copytree(STATIC, out / 'static')
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    html = html.replace('"/static/', '"static/').replace(
        '<script src="static/app.js"></script>', '<script src="data/data.js"></script>\n<script src="static/app.js"></script>')
    (out / 'index.html').write_text(html, encoding='utf-8')
    (out / 'static' / 'index.html').unlink()
    blob = json.dumps({'api': api, 'tiles': tiles, 'trips': names}, ensure_ascii=False, separators=(',', ':'),
                      default=_json_default)
    _write(out / 'data' / 'data.js', ('window.RS_STATIC = ' + blob + ';\n').encode('utf-8'))
    (out / '使用说明.txt').write_text(
        '双击 index.html，用浏览器（Edge、Chrome、Firefox）打开即可，不用安装任何东西、不用联网。\n'
        '这是离线浏览包（只读）：整列长图、零件框、零件详情（模板与本趟局部图并排）、告警、质检与对齐。\n'
        '长图是半分辨率；零件细节点零件框，看详情里的原分辨率局部图。\n'
        '处理新趟、智能体对话要用完整应用：代码包里双击“启动应用.cmd”。\n', encoding='utf-8')
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    return {'step': 'app export', 'out': str(out), 'trips': names, 'tiles': n_tiles, 'crops': n_crops,
            'size_mb': round(size / 2 ** 20, 1)}


def _write(path, data_bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data_bytes)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return io.rel(o)
    raise TypeError(f'导出时遇到不能转 JSON 的值：{type(o).__name__}')
