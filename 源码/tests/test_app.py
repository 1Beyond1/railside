"""S10 应用接口：合成夹具下的趟列表与瓦片、零件框/告警/详情、离线导出、run/agent 任务与错误。"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from railside import cli, db, io
from railside.app import data, dzi, jobs
from railside.app.server import create_app
from railside.coords import H, W

JPEG = b'\xff\xd8'


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope='module')
def synth_frames(tmp_path_factory):
    """一次性生成轻量 3 帧（6000×4096）合成原图与 1/8 预览图，供所有 Web 测试复用。"""
    base = tmp_path_factory.mktemp('app_frames')
    n_frames = 3
    total_w = n_frames * W
    rng = np.random.default_rng(42)
    small = rng.integers(30, 225, (H // 8, total_w // 8), dtype=np.uint8)
    full = cv2.resize(small, (total_w, H), interpolation=cv2.INTER_LINEAR)
    out = {}
    for idx, trip in enumerate(('T01', 'T02')):
        src = base / trip / 'camImg'
        src.mkdir(parents=True)
        canvas = np.roll(full, idx * 100, axis=1)
        for f in range(n_frames):
            ok, buf = cv2.imencode('.jpg', canvas[:, f * W:(f + 1) * W], [cv2.IMWRITE_JPEG_QUALITY, 80])
            assert ok
            (src / f'{f + 1}.jpg').write_bytes(buf.tobytes())
        pv = cv2.resize(canvas[:, ::-1], (total_w // 8, H // 8), interpolation=cv2.INTER_AREA)
        pv_path = base / trip / '整车_1-8.jpg'
        ok, buf = cv2.imencode('.jpg', pv, [cv2.IMWRITE_JPEG_QUALITY, 80])
        assert ok
        pv_path.write_bytes(buf.tobytes())
        out[trip] = {'src': src, 'preview': pv_path, 'n_frames': n_frames, 'width': total_w}
    return out


# ---------- 合成夹具：两趟 + 模板 + 告警 + 退化趟 ----------

@pytest.fixture
def fake(tmp_path, monkeypatch, synth_frames):
    mid, out = tmp_path / '中间', tmp_path / '成品'
    heads = {}
    for trip in ('T01', 'T02'):
        d = mid / trip
        sf = synth_frames[trip]
        tw = sf['width']
        m = {
            'trip': trip, 'src_dir': str(sf['src']), 'n_frames': sf['n_frames'],
            'frame_w': W, 'frame_h': H, 'display_flip': True,
            'train_x0': 0, 'train_x1': tw, 'head_x': 0,
            'train_frames': [1, sf['n_frames']], 'bad_seams': 0,
            'train_panels': [
                {'panel': 1, 'car': 1, 'x0': 0, 'x1': tw // 2, 'width_px': tw // 2, 'height_px': H},
                {'panel': 2, 'car': 2, 'x0': tw // 2, 'x1': tw, 'width_px': tw - tw // 2, 'height_px': H},
            ],
        }
        heads[trip] = m
        dump(d / 'manifest.json', m)
        dump(d / 'trip.json', {'trip': trip, 'src_dir': str(sf['src']), 'camera': 'camImg', 'n_frames': sf['n_frames']})
        (d / '预览').mkdir(parents=True)
        (d / '预览' / '整车_1-8.jpg').write_bytes(sf['preview'].read_bytes())
    dump(mid / 'T03' / 'manifest.json', heads['T02'])  # 只有 manifest 的趟：各接口都要退化
    hx = heads['T01']['head_x']
    parts = [
        {'code': 'C1-B1-BOLT-01', 'type': 'BOLT', 'parent': 'C1-B1-BOGIE-01', 'car': 1, 'zone': 'B1',
         'bbox_uv': [1000, 3000, 1200, 3150], 'centroid_uv': [1100, 3075], 'area_px': 21000, 'angle_deg': 3.0,
         'size_mm': [40.0, 30.0], 'poly_uv': [[1000, 3000], [1200, 3000], [1200, 3150], [1000, 3150]], 'feat': {'mean': 1}},
        {'code': 'C1-B1-BOGIE-01', 'type': 'BOGIE', 'parent': None, 'car': 1, 'zone': 'B1',
         'bbox_uv': [800, 2000, 2400, 4000]},
        {'code': 'C2-BODY-BOX-01', 'type': 'BOX', 'parent': None, 'car': 2, 'zone': 'BODY',
         'bbox_uv': [3500, 1000, 4500, 1800]},
    ]
    dump(out / '模板' / 'v1' / 'template.json', {
        'version': 'v1', 'trip': 'T01', 'head_x': hx, 'parts': parts,
        'anchors': [{'id': 'A101', 'car': 1, 'u': 500, 'v': 744, 'w': 512, 'h': 512}]})
    off = heads['T02']['head_x'] + 100  # T02：x = u + off
    dump(mid / 'T02' / 'align.json', {'template_ver': 'v1', 'g': [{'x': off, 'u': 0}, {'x': off + 5500, 'u': 5500}],
                                      'anchors': [{'id': 'A101', 'car': 1, 'x': off + 500, 'y': 744, 'u': 500, 'v': 744, 'score': 0.9, 'ok': True}],
                                      'residual_px': {'median': 1.2, 'p95': 3.4, 'max': 5.0, 'n': 60}})
    dump(mid / 'T02' / 'parts.json', {'instances': [
        {'code': 'C1-B1-BOLT-01', 'type': 'BOLT', 'parent': 'C1-B1-BOGIE-01', 'bbox_xy': [off + 1000, 3000, off + 1200, 3150],
         'found_by': 'pixel', 'match_score': 0.81},
        {'code': 'C1-B1-BOGIE-01', 'type': 'BOGIE', 'parent': None, 'bbox_xy': [off + 800, 2000, off + 2400, 4000],
         'found_by': 'parent', 'match_score': 0.95},
        {'code': 'C2-BODY-BOX-01', 'type': 'BOX', 'parent': None, 'bbox_xy': [off + 3500, 1000, off + 4500, 1800],
         'found_by': 'missing', 'match_score': None},
    ]})
    dump(mid / 'T02' / 'compare.json', {
        'items': [{'code': 'C1-B1-BOLT-01', 'ncc': 0.42, 'diff_area': 0.15, 'shift_mm': 1.5, 'dtheta_deg': 12.0,
                   'bbox_xy': [off + 1000, 3000, off + 1200, 3150], 'alarms': ['ncc', {'type': 'dtheta'}]},
                  {'code': 'C1-B1-BOGIE-01', 'ncc': 0.97, 'diff_area': 0.0, 'shift_mm': 0.2, 'dtheta_deg': 0.1,
                   'bbox_xy': [off + 800, 2000, off + 2400, 4000], 'alarms': []}],
        'foreign': [{'bbox_uv': [4600, 500, 4900, 700], 'area': 4200}]})
    (tmp_path / '主数据').mkdir()
    (tmp_path / '主数据' / '零件类型.csv').write_text('type,name\nBOLT,螺栓\nBOGIE,转向架\nBOX,设备箱\n', encoding='utf-8')
    monkeypatch.setattr(io, 'DATA', tmp_path)
    monkeypatch.setattr(io, 'MID', mid)
    monkeypatch.setattr(io, 'OUT', out)
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    data._crop.cache_clear()
    dzi._reader.cache_clear()
    dzi._pyramid.cache_clear()
    return {'mid': mid, 'T01': heads['T01'], 'T02': heads['T02'], 'off': off}


def test_trips_lists_synthetic(client, fake):
    r = client.get('/api/trips')
    assert r.status_code == 200
    names = {t['trip'] for t in r.json()['trips']}
    assert {'T01', 'T02', 'T03'} <= names


def test_index_and_vendor(client):
    r = client.get('/')
    assert r.status_code == 200 and 'openseadragon.min.js' in r.text
    assert client.get('/static/vendor/openseadragon/openseadragon.min.js').status_code == 200


def test_dzi_descriptor_and_tiles(client, fake):
    m = fake['T01']
    r = client.get('/api/trips/T01/dzi.dzi')
    assert r.status_code == 200
    assert f'Width="{m["train_x1"] - m["train_x0"]}"' in r.text and 'TileSize="512"' in r.text
    info = client.get('/api/trips/T01').json()
    top = info['dzi']['levels'] - 1
    cars = sorted(info['cars'], key=lambda c: c['car'])
    assert cars[0]['xd'] + cars[0]['w'] == info['width']  # 1 号车在显示图最右
    col = (info['width'] // 2) // 512
    for level, c, row in [(top, col, 3), (top - 2, col // 4, 1), (top - 3, col // 8, 0), (0, 0, 0)]:
        t = client.get(f'/api/trips/T01/dzi_files/{level}/{c}_{row}.jpg')
        assert t.status_code == 200, (level, t.text)
        assert t.headers['content-type'] == 'image/jpeg' and t.content[:2] == JPEG
    cached = fake['mid'] / 'T01' / 'dzi' / f'{m["train_x0"]}-{m["train_x1"]}' / str(top) / f'{col}_3.jpg'
    assert cached.exists()
    assert client.get(f'/api/trips/T01/dzi_files/{top}/99999_0.jpg').status_code == 404
    assert client.get('/api/trips/T01/overview.jpg?w=800&h=32').content[:2] == JPEG


def test_overlay_template_trip(client, fake):
    ov = client.get('/api/trips/T01/overlay').json()
    assert ov['source'] == 'template' and len(ov['boxes']) == 3 and len(ov['anchors']) == 1
    b = next(b for b in ov['boxes'] if b['code'] == 'C1-B1-BOLT-01')
    hx, x1 = fake['T01']['head_x'], fake['T01']['train_x1']
    assert b['xy'] == [1000 + hx, 3000, 1200 + hx, 3150]
    assert b['rect'] == [x1 - (1200 + hx), 3000, 200, 150]  # 翻转显示
    assert b['type_name'] == '螺栓' and b['status'] == 'ok'


def test_overlay_and_alarms_other_trip(client, fake):
    ov = client.get('/api/trips/T02/overlay').json()
    assert ov['source'] == 'parts'
    st = {b['code']: b['status'] for b in ov['boxes']}
    assert st == {'C1-B1-BOLT-01': 'alarm', 'C1-B1-BOGIE-01': 'ok', 'C2-BODY-BOX-01': 'missing'}
    assert ov['counts'] == {'ok': 1, 'alarm': 1, 'missing': 1}
    al = client.get('/api/trips/T02/alarms').json()
    codes = [r['code'] for r in al['rows']]
    assert set(codes) == {'C1-B1-BOLT-01', 'C2-BODY-BOX-01', '异物-01'}
    assert al['types'] == {'ncc': 1, 'dtheta': 1, 'missing': 1, 'foreign': 1}
    fo = next(r for r in al['rows'] if r['kind'] == 'foreign')
    assert fo['xy'][0] == pytest.approx(4600 + fake['off'])  # bbox_uv 经 align 映射回趟坐标
    assert fo['idx'] == 1 and fo['crops'] == {'template': True, 'trip': True}
    # 标准模板对比映射：compare_ref 按 xd_trip 升序
    ref = ov['compare_ref']
    assert ref['template_trip'] == 'T01' and ref['is_template_trip'] is False
    assert len(ref['map']) == 2 and ref['map'][0][0] < ref['map'][1][0] and ref['map'][0][1] < ref['map'][1][1]
    # 异物局部对比图接口
    for side in ('template', 'trip'):
        r = client.get(f'/api/trips/T02/foreign/1/crop.jpg?side={side}')
        assert r.status_code == 200 and r.content[:2] == JPEG
    assert client.get('/api/trips/T02/foreign/99/crop.jpg?side=trip').status_code == 404


def test_part_detail_and_crops(client, fake):
    d = client.get('/api/trips/T02/parts/C1-B1-BOLT-01').json()
    assert d['record']['status'] == 'alarm' and d['record']['alarms'] == ['ncc', 'dtheta']
    assert d['template']['part']['size_mm'] == [40.0, 30.0] and 'poly_uv' not in d['template']['part']
    assert d['compare']['ncc'] == 0.42 and d['instance']['found_by'] == 'pixel'
    hist = {r['trip']: r for r in d['history']['rows']}
    assert set(hist) == {'T01', 'T02'} and hist['T02']['ncc'] == 0.42
    assert d['history']['source'] == '各趟 JSON' and '库不存在' in d['history']['note']  # 没建库时退回读 JSON
    for side in ('template', 'trip'):
        r = client.get(f'/api/trips/T02/parts/C1-B1-BOLT-01/crop.jpg?side={side}')
        assert r.status_code == 200 and r.content[:2] == JPEG
    assert client.get('/api/trips/T02/parts/NOPE').status_code == 404
    s = client.get('/api/trips/T02/summary').json()
    assert s['align']['residual_px']['p95'] == 3.4 and s['compare']['n_alarm_parts'] == 1


def test_history_reads_query_part_history(client, fake):
    """零件历史接 S7：建库后走 query.part_history，把 trips[] 转成网页的行。"""
    db.build(io.OUT / '库.sqlite', ['T01', 'T02'], 'v1')
    d = client.get('/api/trips/T02/parts/C1-B1-BOLT-01').json()
    assert d['history']['source'].startswith('库') and d['history']['note'] is None
    rows = {r['trip']: r for r in d['history']['rows']}
    assert set(rows) == {'T02'}  # T01 没有 parts.json，库里的本趟实例只有 T02
    r = rows['T02']
    assert r['status'] == 'alarm' and r['found_by'] == 'pixel' and r['match_score'] == 0.81
    assert r['ncc'] == 0.42 and r['shift_mm'] == 1.5 and r['dtheta_deg'] == 12.0
    assert r['alarms'] == ['ncc', 'dtheta']


def test_missing_files_degrade(client, fake):
    ov = client.get('/api/trips/T03/overlay').json()
    assert ov['boxes'] == [] and ov['message']
    al = client.get('/api/trips/T03/alarms').json()
    assert al['rows'] == [] and al['message']
    s = client.get('/api/trips/T03/summary').json()
    assert 'qc' not in s and 'align' not in s
    r = client.get('/api/trips/T99/overlay')
    assert r.status_code == 404 and 'T99' in r.json()['error']
    assert client.get('/api/trips/T%2099/summary').status_code == 400


def test_no_template_no_parts(client, tmp_path, monkeypatch):
    dump(tmp_path / '中间' / 'T05' / 'manifest.json', {'train_x0': 0, 'train_x1': 10000, 'head_x': 0, 'frame_h': 4096, 'train_panels': []})
    monkeypatch.setattr(io, 'MID', tmp_path / '中间')
    monkeypatch.setattr(io, 'OUT', tmp_path / '成品')
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    ov = client.get('/api/trips/T05/overlay').json()
    assert ov['source'] is None and '模板' in ov['message']
    assert client.get('/api/trips').json()['trips'][0]['verdict'] is None


# ---------- 处理新趟、智能体 ----------

def wait(client, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f'/api/jobs/{job_id}').json()
        if j['status'] != 'running':
            return j
        time.sleep(0.1)
    raise AssertionError('任务没结束')


def test_run_and_agent_missing_command(client, fake, tmp_path, monkeypatch):
    monkeypatch.setitem(cli.STEPS, 'run', '_no_such_pipeline')
    monkeypatch.setitem(cli.STEPS, 'agent', '_no_such_agent')
    r = client.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'})
    assert r.status_code == 501 and 'railside run' in r.json()['error']
    r = client.post('/api/agent', json={'trip': 'T01', 'message': '你好'})
    assert r.status_code == 501 and 'railside agent' in r.json()['error']
    assert client.post('/api/run', json={'src': str(tmp_path / '没有'), 'trip': 'T09'}).status_code == 400
    assert client.post('/api/run', json={'src': str(tmp_path), 'trip': '../x'}).status_code == 400


def test_run_job_failure_is_reported(client, tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'step_ready', lambda step: True)
    monkeypatch.setattr(jobs, 'command', lambda step, argv: [sys.executable, '-c', 'print("坏帧 boom"); raise SystemExit(3)'])
    j = client.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'}).json()
    j = wait(client, j['id'])
    assert j['status'] == 'failed' and j['rc'] == 3 and '坏帧 boom' in j['log']
    monkeypatch.setattr(jobs, 'command', lambda step, argv: ['__railside_no_such_exe__'])
    j = client.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'}).json()
    assert j['status'] == 'failed' and '启动失败' in j['log'][0]
    assert any(x['id'] == j['id'] for x in client.get('/api/jobs').json()['jobs'])


def test_agent_reply(client, fake, monkeypatch):
    monkeypatch.setattr(jobs, 'step_ready', lambda step: True)
    script = 'import json; print("调用 query"); print(json.dumps({"reply": "T01 没有告警"}, ensure_ascii=False))'
    monkeypatch.setattr(jobs, 'command', lambda step, argv: [sys.executable, '-c', script])
    j = client.post('/api/agent', json={'trip': 'T01', 'message': '有告警吗'}).json()
    j = wait(client, j['id'])
    assert j['status'] == 'ok' and j['reply'] == 'T01 没有告警'


def test_export_offline_bundle(fake, tmp_path):
    """离线浏览包：接口结果进 data.js，页面改相对路径，瓦片和局部图齐，非空目录拒绝覆盖。"""
    from railside.app.export import export
    out = tmp_path / '离线包'
    r = export(out, ['T02'], max_level=10, log=lambda m: None)
    html = (out / 'index.html').read_text(encoding='utf-8')
    assert '"/static/' not in html and html.index('data/data.js') < html.index('static/app.js')
    blob = (out / 'data' / 'data.js').read_text(encoding='utf-8')
    assert blob.startswith('window.RS_STATIC = ')
    d = json.loads(blob[len('window.RS_STATIC = '):].rstrip().rstrip(';'))
    assert d['hasOcr'] is False
    assert d['trips'] == ['T02'] and [t['trip'] for t in d['api']['/api/trips']['trips']] == ['T02']
    for k in ('', '/overlay', '/alarms', '/summary', '/parts/C1-B1-BOLT-01'):
        assert f'/api/trips/T02{k}' in d['api'], k
    t = d['tiles']['T02']
    natural = d['api']['/api/trips/T02']['dzi']['levels'] - 1
    assert t['maxLevel'] == 10 and t['tileSize'] == 512 << (natural - 10) and t['overlap'] == 1 << (natural - 10)
    # 前端按 tileSize 算的每层瓦片数，要和导出的文件对得上
    for level in (0, 5, 10):
        cols = -(-t['width'] // (t['tileSize'] << (10 - level)))
        names = {p.name for p in (out / 'data' / 'T02' / 'tiles' / str(level)).iterdir()}
        assert f'{cols - 1}_0.jpg' in names and f'{cols}_0.jpg' not in names, level
    det = d['api']['/api/trips/T02/parts/C1-B1-BOLT-01']
    for side in ('template', 'trip'):
        assert det['crops'][side] and (out / 'data' / 'T02' / 'crops' / f'C1-B1-BOLT-01_{side}.jpg').read_bytes()[:2] == JPEG
        assert (out / 'data' / 'T02' / 'crops' / f'foreign_1_{side}.jpg').read_bytes()[:2] == JPEG
    # 自动附带标准模板参考趟 T01 的瓦片和模板轮廓
    assert 'T01' in d['tiles'] and '/api/trips/T01/overlay' in d['api']
    assert (out / 'data' / 'T01' / 'tiles' / '0' / '0_0.jpg').read_bytes()[:2] == JPEG
    assert (out / 'data' / 'T02' / 'overview.jpg').exists() and r['tiles'] > 0
    with pytest.raises(SystemExit):
        export(out, ['T02'], max_level=10, log=lambda m: None)
