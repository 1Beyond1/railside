"""S7 测试夹具：在临时目录里造一个项目（两趟 + 模板 + 告警 + 一趟只有 manifest），字段按 设计.md §4.1。

用法：root = make_project(tmp_path, monkeypatch)。会把 railside.io 的 ROOT/DATA/MID/OUT 指到 tmp_path 下。
"""
import json

import numpy as np

from railside import io

HEAD_X = 1000
X1 = 17000
PANELS = [{'panel': 1, 'car': 1, 'x0': 1000, 'x1': 9000, 'width_px': 8000, 'height_px': 4096},
          {'panel': 2, 'car': 2, 'x0': 9000, 'x1': 17000, 'width_px': 8000, 'height_px': 4096}]
# 模板零件：编码 → (类型, 父件, 车, 区, bbox_uv)
TPL_PARTS = {
    'C1-B1-BOLT-01': ('BOLT', 'C1-B1-FRAME-01', 1, 'B1', [500, 2000, 560, 2060]),
    'C1-B1-BOLT-02': ('BOLT', 'C1-B1-FRAME-01', 1, 'B1', [900, 2100, 960, 2160]),
    'C2-BODY-BOX-01': ('BOX', None, 2, 'BODY', [9000, 1000, 10000, 1800]),
}
SHIFT = {'T01': 0, 'T02': 100}  # 本趟 x = u + HEAD_X + SHIFT
ALARMS = {
    'T01': {'C1-B1-BOLT-01': ['ncc']},
    'T02': {'C1-B1-BOLT-02': ['rotate'], 'C2-BODY-BOX-01': [{'type': 'missing', 'value': 1, 'threshold': 0}]},
}


def _manifest(trip):
    return {'trip': trip, 'src_dir': f'数据/原始/{trip}/camImg', 'n_frames': 10, 'frame_w': 2000, 'frame_h': 4096,
            'display_flip': True, 'train_x0': HEAD_X, 'train_x1': X1, 'head_x': HEAD_X, 'train_frames': [1, 9],
            'blank': [[0, HEAD_X], [X1, 20000]], 'head_protect': {'x0': 0, 'x1': 9000}, 'train_panels': PANELS,
            'seams': [], 'bad_seams': 0}


def make_project(tmp_path, monkeypatch):
    root = tmp_path / 'proj'
    monkeypatch.setattr(io, 'ROOT', root)
    monkeypatch.setattr(io, 'DATA', root / '数据')
    monkeypatch.setattr(io, 'MID', root / '数据' / '中间')
    monkeypatch.setattr(io, 'OUT', root / '数据' / '成品')
    rng = np.random.default_rng(0)

    (root / '主数据').mkdir(parents=True)
    (root / '主数据' / '零件类型.csv').write_text(
        'type,name,parent_type,desc,ncc_min,shift_mm_max,dtheta_max_deg,area_change_max\n'
        'BOLT,螺栓,FRAME,紧固件,0.6,3,5,0.3\nBOX,设备箱,,车下设备,0.5,10,3,0.2\n', encoding='utf-8')

    tdir = io.OUT / '模板' / 'v1'
    tparts = []
    for code, (typ, parent, car, zone, b) in TPL_PARTS.items():
        crop = f'crops/{code}.png'
        io.imwrite(tdir / crop, rng.integers(0, 255, (b[3] - b[1] + 40, b[2] - b[0] + 40), np.uint8))
        tparts.append({'code': code, 'type': typ, 'parent': parent, 'car': car, 'zone': zone, 'bbox_uv': b,
                       'poly_uv': [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                       'centroid_uv': [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], 'area_px': (b[2] - b[0]) * (b[3] - b[1]),
                       'angle_deg': 0.0, 'size_mm': [12.0, 12.0], 'crop': crop, 'mask': f'masks/{code}.png',
                       'feat': {'mean': 120.5, 'hu': [0.1, 0.2]}})
    io.write_json(tdir / 'template.json', {
        'version': 'v1', 'trip': 'T01', 'head_x': HEAD_X,
        'anchors': [{'id': 'A1', 'car': 1, 'u': 300, 'v': 1500, 'w': 128, 'h': 128, 'patch': 'anchors/A1.png'},
                    {'id': 'A2', 'car': 2, 'u': 9500, 'v': 1500, 'w': 128, 'h': 128, 'patch': 'anchors/A2.png'}],
        'parts': tparts, 'calib': {'sx_mm_per_px': 0.5}})

    for trip in ('T01', 'T02'):
        d = io.trip_dir(trip)
        sh = SHIFT[trip]
        io.write_json(d / 'trip.json', {'trip': trip, 'src_dir': f'数据/原始/{trip}/camImg', 'camera': 'camImg', 'n_frames': 10})
        io.write_json(d / 'qc.json', {'trip': trip, 'frames': 10, 'missing': [], 'per_frame': [], 'verdict': 'pass', 'reasons': []})
        io.write_json(d / 'manifest.json', _manifest(trip))
        io.write_json(d / 'calib.json', {'sx_curve': [{'x': 2000, 'mm_per_px': 0.5}, {'x': 15000, 'mm_per_px': 0.52}],
                                         'sy_mm_per_px': 0.6, 'refs': [{'name': '轴距', 'mm': 2500, 'px': 5000}],
                                         'source': '公开车型资料'})
        io.write_json(d / 'align.json', {
            'template_ver': 'v1',
            'anchors': [{'id': 'A1', 'x': 300 + HEAD_X + sh, 'y': 1500, 'u': 300, 'v': 1500, 'score': 0.9}],
            'g': [{'x': HEAD_X + sh, 'u': 0}, {'x': X1 + sh, 'u': X1 - HEAD_X}],
            'residual_px': {'median': 1.5, 'p95': 4.0}, 'u_trim0': 0})
        io.write_json(d / 'seg.json', {'model': {'name': 'none', 'sha256': None}, 'dets': []})
        inst = []
        for code, (typ, parent, car, zone, b) in TPL_PARTS.items():
            missing = trip == 'T02' and code == 'C2-BODY-BOX-01'
            bxy = [b[0] + HEAD_X + sh, b[1], b[2] + HEAD_X + sh, b[3]]
            crop = f'crops/{code}.png' if trip == 'T02' and not missing else None  # T01 不给局部图，走原图兜底（原图也没有 → 无图）
            inst.append({'code': code, 'type': typ, 'parent': parent, 'bbox_uv': b, 'bbox_xy': bxy, 'mask': None, 'crop': crop,
                         'centroid_mm': [1.0, 2.0], 'area_mm2': 36.0,
                         'angle_deg': 12.0 if (trip, code) == ('T02', 'C1-B1-BOLT-02') else 0.0,
                         'match_score': 0.0 if missing else 0.9,
                         'found_by': 'missing' if missing else ('parent' if typ == 'BOX' else 'pixel'),
                         'src_frames': [bxy[0] // 2000 + 1, (bxy[2] - 1) // 2000 + 1]})
            if crop:
                io.imwrite(d / crop, rng.integers(0, 255, (100, 100), np.uint8))
        io.write_json(d / 'parts.json', {'trip': trip, 'instances': inst})
        items = [{'code': code, 'ncc': 0.95, 'diff_area': 10, 'shift_mm': [0.1, 0.2], 'dtheta_deg': 0.5,
                  'alarms': ALARMS[trip].get(code, [])} for code in TPL_PARTS]
        foreign = [{'bbox_uv': [3000, 500, 3200, 700], 'area': 4000}] if trip == 'T02' else []
        io.write_json(d / 'compare.json', {'trip': trip, 'template_ver': 'v1', 'items': items, 'foreign': foreign})
        io.imwrite(d / '预览' / '整车_1-8.jpg', rng.integers(0, 255, (512, X1 // 8 - HEAD_X // 8), np.uint8))
    (io.trip_dir('T02') / 'agent_log.jsonl').write_text(json.dumps(
        {'ts': '2026-11-17T20:00:00+08:00', 'trip': 'T02', 'step': 'compare', 'tool': 'railside compare',
         'args': {'trip': 'T02'}, 'result': {'alarms': 3}, 'status': 'ok', 'code_ver': 'abc'}, ensure_ascii=False) + '\n',
        encoding='utf-8')

    # T03：只有 trip.json 和 manifest.json，其余各步都没跑
    io.write_json(io.trip_dir('T03') / 'trip.json', {'trip': 'T03', 'src_dir': '数据/原始/T03/camImg', 'camera': 'camImg', 'n_frames': 10})
    io.write_json(io.trip_dir('T03') / 'manifest.json', _manifest('T03'))
    return root
