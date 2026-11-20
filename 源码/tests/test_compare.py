"""S8 对比的测试：合成图上的转角、面积、亮度、位移；异物；告警判定；端到端。

单元测试用 test_match 里的 FakeTrip / photo，不读真实数据；端到端复用 test_match 的 proj 夹具。
"""
import numpy as np
import cv2
import pytest
from types import SimpleNamespace

from railside import cli, io, compare as C
from railside.coords import AlignMap
from test_match import FakeTrip, photo, proj                     # noqa: F401  （proj 是夹具）


# ---------------------------------------------------------------- 合成零件

def part_scene(tmp_path, monkeypatch, theta=0.0, scale=1.0, gain=1.0, off=0.0, shift=(0.0, 0.0)):
    """造一个零件：模板局部图（T01）+ 本趟图（可旋转、缩放、压暗、平移）。

    返回 (q, it, t_new, amap)。零件 260×200 放在 (1000, 800)，多边形内缩 6 px。
    """
    root = tmp_path / 'proj'
    for name, val in (('ROOT', root), ('DATA', root / '数据'), ('MID', root / '数据' / '中间'),
                      ('OUT', root / '数据' / '成品')):
        monkeypatch.setattr(io, name, val)
    tex = photo(200, 260, seed=11)
    poly = [[6, 6], [254, 6], [254, 194], [6, 194]]
    mask = np.zeros(tex.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(poly, float)).astype(np.int32)], 255)
    d = root / '数据' / '成品' / '模板' / 'v1'
    io.imwrite(d / 'crops' / 'P1.png', tex)
    io.imwrite(d / 'masks' / 'P1.png', mask)
    ox, oy = 1000, 800
    ys, xs = np.nonzero(mask)
    q = {'code': 'P1', 'type': 'BOLT', 'parent': None, 'car': 1, 'zone': 'B1',
         'bbox_uv': [ox + 6, oy + 6, ox + 255, oy + 195], 'poly_uv': [[x + ox, y + oy] for x, y in poly],
         'centroid_uv': [ox + float(xs.mean()), oy + float(ys.mean())], 'area_px': int(len(xs)),
         'angle_deg': 0.0, 'size_mm': None, 'crop': io.rel(d / 'crops' / 'P1.png'),
         'mask': io.rel(d / 'masks' / 'P1.png'), 'crop_origin_uv': [ox, oy], 'feat': {},
         'source': 'human', 'verified': True, 'label_id': 1}
    # 本趟图：把局部图按 (theta, scale) 变换后贴到 canvas 上
    T = cv2.getRotationMatrix2D((tex.shape[1] / 2, tex.shape[0] / 2), theta, scale)
    W2, H2 = tex.shape[1] + 120, tex.shape[0] + 120
    T[0, 2] += (W2 - tex.shape[1]) / 2
    T[1, 2] += (H2 - tex.shape[0]) / 2
    warped = cv2.warpAffine(tex, T, (W2, H2), borderMode=cv2.BORDER_REPLICATE)
    warped = np.clip(warped.astype(np.float32) * gain + off, 0, 255).astype(np.uint8)
    px = int(round(ox - (W2 - tex.shape[1]) / 2 + shift[0]))
    py = int(round(oy - (H2 - tex.shape[0]) / 2 + shift[1]))
    canvas = np.full((1400, 1800), 100, np.uint8)
    canvas[py:py + H2, px:px + W2] = warped
    # 实测：模板多边形经过同一个变换，再平移到 (px, py)
    Tp = cv2.getRotationMatrix2D((tex.shape[1] / 2, tex.shape[0] / 2), theta, scale)
    Tp[0, 2] += (W2 - tex.shape[1]) / 2 + px
    Tp[1, 2] += (H2 - tex.shape[0]) / 2 + py
    p = np.hstack([np.asarray(poly, float), np.ones((len(poly), 1))]) @ Tp.T
    m2 = np.zeros(canvas.shape, np.uint8)
    cv2.fillPoly(m2, [np.round(p).astype(np.int32)], 255)
    ys, xs = np.nonzero(m2)
    it = {'code': 'P1', 'bbox_xy': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
          'centroid_xy': [float(xs.mean()), float(ys.mean())], 'area_px': int(len(xs)), 'found_by': 'pixel'}
    amap = AlignMap([0, 1800], [0, 1800])
    return q, it, FakeTrip(canvas), amap


# ---------------------------------------------------------------- 指标

@pytest.mark.parametrize('theta', [5.0, 10.0, 20.0])
def test_dtheta(tmp_path, monkeypatch, theta):
    q, it, t, amap = part_scene(tmp_path, monkeypatch, theta=theta)
    r = C.measure(t, q, it, amap)
    assert abs(r['dtheta'] - theta) < 1.2, r
    assert r['ncc'] > 0.85


def test_no_rotation_gives_zero(tmp_path, monkeypatch):
    q, it, t, amap = part_scene(tmp_path, monkeypatch)
    r = C.measure(t, q, it, amap)
    assert abs(r['dtheta']) < 1.0 and r['ncc'] > 0.95


def test_area_change(tmp_path, monkeypatch):
    q, it, t, amap = part_scene(tmp_path, monkeypatch, scale=1.10)
    r = C.measure(t, q, it, amap)
    assert abs(r['s'] / r['s_amap'] - 1.10) < 0.02
    assert abs(r['s'] ** 2 / r['s_amap'] ** 2 - 1 - (1.10 ** 2 - 1)) < 0.05


def test_brightness_insensitive(tmp_path, monkeypatch):
    q, it, t, amap = part_scene(tmp_path, monkeypatch, gain=0.6, off=25.0)
    r = C.measure(t, q, it, amap)
    assert r['ncc'] > 0.9 and abs(r['dtheta']) < 1.0


def test_offset_and_registration(tmp_path, monkeypatch):
    """零件平移了 (9, −6)：没有 offset_uv 时按质心反算出偏移，局部配准以它为中心，相似度仍高。"""
    q, it, t, amap = part_scene(tmp_path, monkeypatch, shift=(9.0, -6.0))
    du, dv = C.offset_of(it, q, amap)
    assert abs(du - 9) < 1.5 and abs(dv + 6) < 1.5
    assert C.offset_of(dict(it, offset_uv=[3.0, 4.0]), q, amap) == (3.0, 4.0)   # 有就用 S6b 量的
    r = C.measure(t, q, it, amap)
    assert r['ncc'] > 0.9


# ---------------------------------------------------------------- 异物

def _parent_scene(tmp_path, monkeypatch, blob=None):
    root = tmp_path / 'proj'
    for name, val in (('ROOT', root), ('DATA', root / '数据'), ('MID', root / '数据' / '中间'),
                      ('OUT', root / '数据' / '成品')):
        monkeypatch.setattr(io, name, val)
    rng = np.random.default_rng(3)
    base = cv2.resize(rng.integers(0, 255, (120, 160), dtype=np.uint8),
                      (1800, 1400), interpolation=cv2.INTER_CUBIC)
    base = cv2.GaussianBlur(base, (0, 0), 2)
    cv2.line(base, (100, 300), (1600, 1100), 30, 7)
    t_tpl = FakeTrip(base)
    new = base.copy()
    if blob is not None:
        cv2.rectangle(new, blob[:2], (blob[2] - 1, blob[3] - 1), 235, -1)
    t_new = FakeTrip(new)
    tparts = [{'code': 'BOGIE-1', 'type': 'BOGIE', 'parent': None, 'bbox_uv': [200, 200, 1200, 1000],
               'poly_uv': [[200, 200], [1200, 200], [1200, 1000], [200, 1000]], 'centroid_uv': [700, 600]},
              {'code': 'BOLT-1', 'type': 'BOLT', 'parent': 'BOGIE-1', 'bbox_uv': [500, 500, 620, 620],
               'poly_uv': [[500, 500], [620, 500], [620, 620], [500, 620]], 'centroid_uv': [560, 560]}]
    inst = {'BOGIE-1': {'code': 'BOGIE-1', 'bbox_xy': [200, 200, 1200, 1000], 'centroid_xy': [700, 600],
                        'found_by': 'pixel'},
            'BOLT-1': {'code': 'BOLT-1', 'bbox_xy': [500, 500, 620, 620], 'centroid_xy': [560, 560],
                       'found_by': 'pixel', 'poly_xy': [[500, 500], [620, 500], [620, 620], [500, 620]]}}
    amap = AlignMap([0, 1800], [0, 1800])
    return t_tpl, t_new, tparts, inst, amap


def test_foreign_blob_detected(tmp_path, monkeypatch):
    t_tpl, t_new, tparts, inst, amap = _parent_scene(tmp_path, monkeypatch, blob=(900, 350, 1000, 450))
    f = C.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts, inst, [], amap,
                       {'diff': 1.5, 'min_area': 400, 'min_score': 2.0})
    assert len(f) == 1, f
    b = f[0]['bbox_xy']
    assert b[0] <= 900 and b[2] >= 1000 and b[1] <= 350 and b[3] >= 450
    assert f[0]['src'] == 'diff' and f[0]['area'] >= 400


def test_no_foreign_on_clean_image(tmp_path, monkeypatch):
    t_tpl, t_new, tparts, inst, amap = _parent_scene(tmp_path, monkeypatch)
    assert C.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts, inst, [], amap,
                          {'diff': 1.5, 'min_area': 400, 'min_score': 2.0}) == []


def test_foreign_ignores_change_inside_part_mask(tmp_path, monkeypatch):
    """零件自己变了不算异物（有掩膜挡着）。"""
    t_tpl, t_new, tparts, inst, amap = _parent_scene(tmp_path, monkeypatch, blob=(520, 520, 600, 600))
    assert C.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts, inst, [], amap,
                          {'diff': 1.5, 'min_area': 200, 'min_score': 2.0}) == []


def test_foreign_ignores_whole_wheel_disc(tmp_path, monkeypatch):
    """车轮会转：标注只盖下半段，上半个轮面的变化也不算异物；轮盘外的照报。"""
    t_tpl, t_new, tparts, inst, amap = _parent_scene(tmp_path, monkeypatch, blob=(640, 480, 720, 560))
    wheel = {'code': 'WHEEL-1', 'type': 'WHEEL', 'parent': 'BOGIE-1', 'bbox_uv': [500, 700, 900, 850],
             'poly_uv': [[500, 700], [900, 700], [900, 850], [500, 850]], 'centroid_uv': [700, 775]}
    inst['WHEEL-1'] = {'code': 'WHEEL-1', 'bbox_xy': [500, 700, 900, 850], 'centroid_xy': [700, 775],
                       'found_by': 'pixel'}
    cfg = {'diff': 1.5, 'min_area': 400, 'min_score': 2.0}
    assert len(C.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts, inst, [], amap, cfg)) == 1
    assert C.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts + [wheel], inst, [], amap, cfg) == []
    t_tpl, t_new, tparts, inst, amap = _parent_scene(tmp_path, monkeypatch, blob=(1000, 350, 1100, 450))
    inst['WHEEL-1'] = {'code': 'WHEEL-1', 'bbox_xy': [500, 700, 900, 850], 'centroid_xy': [700, 775],
                       'found_by': 'pixel'}
    assert len(C.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts + [wheel], inst, [], amap, cfg)) == 1


# ---------------------------------------------------------------- 判定与输出

@pytest.fixture
def ready(proj):
    """跑完 match 的项目。"""
    cli.main(['match', 'T02'])
    return proj


def _patch_measure(monkeypatch, **kw):
    base = {'ncc': 0.95, 'ncc0': 0.95, 'dtheta': 0.1, 's': 1.0, 's_amap': 1.0, 'w': 1.0}
    base.update(kw)
    monkeypatch.setattr(C, 'measure', lambda *a, **k: dict(base))


def test_status_ok_and_missing(ready, capsys):
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    assert [i['status'] for i in cj['items']] == ['ok', 'ok']
    pj = io.read_json(io.MID / 'T02' / 'parts.json')
    for i in pj['instances']:
        if i['code'] == 'C1-B1-BOLT-01':
            i['found_by'] = 'missing'
    io.write_json(io.MID / 'T02' / 'parts.json', pj)
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    bolt = [i for i in cj['items'] if i['code'] == 'C1-B1-BOLT-01'][0]
    assert bolt['status'] == 'missing' and bolt['alarms'][0]['type'] == 'missing'
    assert bolt['ncc'] is None


def _set_offsets(**offs):
    pj = io.read_json(io.MID / 'T02' / 'parts.json')
    for i in pj['instances']:
        if i['code'] in offs:
            i['offset_uv'] = list(offs[i['code']])
    io.write_json(io.MID / 'T02' / 'parts.json', pj)


def test_alarm_types(ready, monkeypatch, capsys):
    # 父件相对整车 (40, 10)，子件 (60, 20)：子件相对父件 (20, 10) 模板 px → 横向 ×0.8 → 3.8 mm、3.0 mm
    _set_offsets(**{'C1-B1-BOGIE-01': (40, 10), 'C1-B1-BOLT-01': (60, 20)})
    _patch_measure(monkeypatch, ncc=0.1, dtheta=12.0, s=1.1, s_amap=0.8)
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    types = {a['type'] for i in cj['items'] for a in i['alarms']}
    assert {'ncc', 'shift', 'dtheta', 'area'} <= types
    assert all(i['status'] == 'alarm' for i in cj['items'])
    it = cj['items'][0]
    assert it['shift_mm'] is not None and it['shift_vec_mm'] and len(it['shift_vec_mm']) == 2


def test_child_shift_relative_to_parent(ready, monkeypatch, capsys):
    """转向架整台平移了，螺栓跟着它走：只报父件位移，子件相对父件为 0。"""
    _set_offsets(**{'C1-B1-BOGIE-01': (40, 10), 'C1-B1-BOLT-01': (40, 10)})
    _patch_measure(monkeypatch)
    cli.main(['compare', 'T02'])
    by = {i['code']: i for i in io.read_json(io.MID / 'T02' / 'compare.json')['items']}
    assert by['C1-B1-BOLT-01']['shift_mm'] < 0.01 and by['C1-B1-BOLT-01']['status'] == 'ok'
    assert [a['type'] for a in by['C1-B1-BOGIE-01']['alarms']] == ['shift']


def test_thresholds_from_csv(ready, capsys):
    """阈值表里的值要生效：把 BOGIE 的 ncc_min 抬到实测 NCC 之上，就该报 ncc。"""
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    ncc = [i for i in cj['items'] if i['code'] == 'C1-B1-BOGIE-01'][0]['ncc']
    p = io.ROOT / '主数据' / '零件类型.csv'
    txt = p.read_text(encoding='utf-8').replace(f'BOGIE,转向架,,走行部,0.5', f'BOGIE,转向架,,走行部,{ncc + 0.02:.3f}')
    p.write_text(txt, encoding='utf-8')
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    assert [a['type'] for i in cj['items'] if i['code'] == 'C1-B1-BOGIE-01' for a in i['alarms']] == ['ncc']


def test_qc_skip(ready, capsys):
    """qc2 标了质量差的位置：落在那里的零件 skip + qc 告警，不判别的。"""
    al = io.read_json(io.MID / 'T02' / 'align.json')
    al['qc2'] = [{'u': 1425, 'x': 1462, 'ncc': 0.4, 'blur_ratio': 0.5, 'used': False, 'flags': ['mismatch']}]
    io.write_json(io.MID / 'T02' / 'align.json', al)
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    bogie = [i for i in cj['items'] if i['code'] == 'C1-B1-BOGIE-01'][0]
    assert bogie['status'] == 'skip' and [a['type'] for a in bogie['alarms']] == ['qc']
    assert bogie['ncc'] is not None          # 指标照算，只是不判


def test_suggest_thresholds(ready, capsys):
    # 这里只验证统计函数：夹具为合成数据，真实 T02 不允许生成阈值建议。
    cli.main(['compare', 'T02'])
    C.suggest(io.read_json(io.trip_dir('T02') / 'compare.json')['items'],
              SimpleNamespace(trip='T01', suggest_out='实验/E3', quantile=99.0, relax=1.5))
    d = io.ROOT / '实验' / 'E3'
    rows = (d / '阈值建议.csv').read_text(encoding='utf-8-sig').splitlines()
    assert rows[0].split(',')[:4] == ['type', 'n', 'ncc_min', 'shift_mm_max']
    assert {r.split(',')[0] for r in rows[1:]} == {'BOGIE', 'BOLT'}
    js = io.read_json(d / '阈值建议.json')
    assert js['quantile'] == 99.0 and len(js['suggest']) == 2
    assert (d / '指标统计.csv').is_file()


@pytest.mark.parametrize('trip', ['T02', 'T02_synth'])
def test_evaluation_trip_cannot_generate_thresholds(tmp_path, monkeypatch, trip):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    monkeypatch.setattr(io, 'MID', tmp_path / '中间')
    with pytest.raises(SystemExit, match='仅允许 T01'):
        cli.main(['compare', trip, '--suggest', '--suggest-out', 'thresholds'])
    assert not (tmp_path / 'thresholds').exists()

