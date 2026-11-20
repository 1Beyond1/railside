"""评测考题与评分：小夹具，不读整趟原图。"""
import cv2
import numpy as np
import pytest

from railside import io
from railside.coords import H, W
from railside.eval import badframes, score, synth
from railside.eval.frameset import FrameCopy


def _frames(d, n, seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    for f in range(1, n + 1):
        im = cv2.GaussianBlur(rng.integers(0, 255, (H // 8, W // 8), dtype=np.uint8), (0, 0), 1)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imencode('.jpg', im)[1].tofile(str(d / f'{f}.jpg'))
    return d


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    monkeypatch.setattr(io, 'MID', tmp_path / '数据/中间')
    monkeypatch.setattr(io, 'OUT', tmp_path / '数据/成品')
    return tmp_path


def test_framecopy_cross_frame_and_untouched_bytes(root):
    src = _frames(root / 'src', 3)
    fc = FrameCopy(src, root / 'out')
    im, o = fc.get(W - 50, W + 50, 100, 200)        # 跨第 1、2 帧
    assert im.shape == (100, 100) and o == (W - 50, 100)
    fc.put(np.full_like(im, 7), o)
    fc.write(drop={3})
    assert sorted(p.name for p in (root / 'out').iterdir()) == ['1.jpg', '2.jpg']
    a = io.imread(root / 'out' / '1.jpg')
    b = io.imread(root / 'out' / '2.jpg')
    assert abs(int(a[150, W - 10]) - 7) <= 3 and abs(int(b[150, 10]) - 7) <= 3
    assert fc.dirty == {1, 2}


def _parts(n_car=2):
    """每节车一台转向架（父件）和几个子件，坐标落在 4 帧内。"""
    inst = []
    for car in range(1, n_car + 1):
        x0 = (car - 1) * 3000 + 200
        inst.append({'code': f'C{car}-B1-BOGIE-01', 'type': 'BOGIE', 'parent': None, 'car': car,
                     'bbox_xy': [x0, 500, x0 + 2600, 3500], 'found_by': 'pixel', 'match_score': 0.95,
                     'poly_xy': [[x0, 500], [x0 + 2600, 500], [x0 + 2600, 3500], [x0, 3500]]})
        for k, typ in enumerate(['BOLT', 'AXLE_END', 'LINK', 'DAMPER', 'BOLT', 'SPRING1']):
            cx, cy = x0 + 200 + k * 400, 1500 + 300 * (k % 3)
            inst.append({'code': f'C{car}-B1-{typ}-{k:02d}', 'type': typ, 'parent': f'C{car}-B1-BOGIE-01', 'car': car,
                         'bbox_xy': [cx - 30, cy - 30, cx + 30, cy + 30], 'found_by': 'pixel', 'match_score': 0.9,
                         'poly_xy': [[cx - 30, cy - 30], [cx + 30, cy - 30], [cx + 30, cy + 30], [cx - 30, cy + 30]]})
    return {'trip': 'T02', 'instances': inst}


PLAN = [('foreign', ('BOGIE',), [{'shape': 'blob'}, {'shape': 'wire'}]),
        ('missing', ('BOLT',), [{}]), ('rotate', ('AXLE_END', 'BOLT'), [{'deg': 10}]),
        ('shift', ('LINK', 'DAMPER'), [{'k': 2.0}]), ('occlude', ('SPRING1', 'DAMPER'), [{}])]


def test_synth_reproducible_and_only_touches_defect_frames(root, monkeypatch):
    monkeypatch.setattr(synth, 'SPACING', 100)
    monkeypatch.setattr(synth, 'load_types', lambda: {'LINK': {'shift_mm_max': '8'}, 'DAMPER': {'shift_mm_max': '8'}})
    src = _frames(root / 'src', 4)
    pj = root / 'parts.json'
    io.write_json(pj, _parts())
    outs = []
    for i in range(2):
        out = root / f'out{i}'
        r = synth.build(src, pj, root / 'nope.json', out, root / f'truth{i}.json', seed=7, plan=PLAN)
        assert r['defects'] == 6
        outs.append(out)
    t0, t1 = io.read_json(root / 'truth0.json'), io.read_json(root / 'truth1.json')
    assert t0['defects'] == t1['defects']                      # 同种子同真值
    for p in outs[0].iterdir():                                # 同种子同像素
        assert p.read_bytes() == (outs[1] / p.name).read_bytes()
    touched = set(t0['frames_modified'])
    for p in outs[0].iterdir():
        if int(p.stem) not in touched:
            assert p.read_bytes() == (src / p.name).read_bytes()
    d = {x['kind']: x for x in t0['defects']}
    assert d['shift']['params']['shift_mm'] == 16.0
    assert abs(d['rotate']['params']['deg']) == 10
    # 真的改了像素：缺失处和原图差别明显
    m = d['missing']
    x0, y0, x1, y1 = m['part_bbox_xy']
    a = FrameCopy(src, root / 'x').get(x0, x1, y0, y1)[0].astype(int)
    b = FrameCopy(outs[0], root / 'y').get(x0, x1, y0, y1)[0].astype(int)
    assert np.abs(a - b).mean() > 10


def test_synth_skips_alarmed_and_weak_parts():
    parts = _parts(1)
    parts['instances'][1]['match_score'] = 0.5
    cmp = {'items': [{'code': parts['instances'][2]['code'], 'status': 'alarm'}]}
    c = {p['code'] for p in synth.candidates(parts, cmp, {'BOLT', 'AXLE_END', 'LINK', 'BOGIE'})}
    assert parts['instances'][1]['code'] not in c and parts['instances'][2]['code'] not in c
    assert parts['instances'][3]['code'] in c and parts['instances'][0]['code'] in c


def test_badframes_truth(root):
    src = _frames(root / 'src', 30)
    r = badframes.build(src, root / 'out', root / 'truth.json', seed=1, per_kind=1, train=(5, 25))
    t = io.read_json(root / 'truth.json')
    assert r['bad'] == 5 and {b['kind'] for b in t['bad']} == {'missing', 'dark', 'over', 'blur', 'occlusion'}
    miss = [b['f'] for b in t['bad'] if b['kind'] == 'missing']
    assert not (root / 'out' / f'{miss[0]}.jpg').exists()
    dark = next(b['f'] for b in t['bad'] if b['kind'] == 'dark')
    assert io.imread(root / 'out' / f'{dark}.jpg').mean() < io.imread(src / f'{dark}.jpg').mean() * 0.3


def test_score_qc_and_compare():
    truth = {'trip': 'X', 'bad': [{'f': 3, 'kind': 'missing', 'params': {}}, {'f': 7, 'kind': 'dark', 'params': {}},
                                  {'f': 9, 'kind': 'blur', 'params': {}}]}
    qc = {'missing': [3], 'per_frame': [{'f': 7, 'flags': ['dark']}, {'f': 9, 'flags': []}, {'f': 12, 'flags': ['over']}]}
    s = score.score_qc(truth, qc)
    assert (s['hit'], s['n'], s['false_flags']) == (2, 3, [12])
    st = {'trip': 'S', 'defects': [
        {'id': 'D1', 'kind': 'rotate', 'code': 'A', 'bbox_xy': [0, 0, 10, 10], 'part_bbox_xy': [0, 0, 10, 10], 'params': {}},
        {'id': 'D2', 'kind': 'shift', 'code': 'B', 'bbox_xy': [5000, 0, 5010, 10], 'part_bbox_xy': [5000, 0, 5010, 10], 'params': {}},
        {'id': 'D3', 'kind': 'foreign', 'code': 'G', 'bbox_xy': [9000, 0, 9100, 100], 'part_bbox_xy': [8000, 0, 12000, 4000], 'params': {}}]}
    cmp = {'items': [{'code': 'A', 'status': 'alarm', 'alarms': [{'type': 'ncc'}], 'bbox_xy': [0, 0, 10, 10]},
                     {'code': 'B', 'status': 'ok', 'alarms': [], 'bbox_xy': [5000, 0, 5010, 10]},
                     {'code': 'Z', 'status': 'alarm', 'alarms': [{'type': 'ncc'}], 'bbox_xy': [20000, 0, 20010, 10]}],
           'foreign': [{'bbox_xy': [9050, 50, 9200, 200]}]}
    c = score.score_compare(st, cmp, {'items': [{'status': 'alarm'}, {'status': 'ok'}], 'foreign': []})
    assert (c['hit'], c['hit_type'], c['n']) == (2, 1, 3)       # 转角报成外观：检出但类型不对
    assert c['synth_far_alarms'] == 1 and c['base_trip_alarms'] == 1


def test_score_missing_inputs_do_not_crash(root):
    s = score.score_all()
    assert all(v['status'] == '缺' for v in s.values())
    assert '缺' in score.table(s)


def test_score_align_skips_unlabeled_rows(root):
    from railside.eval import pairs
    (root / '数据/中间/T01').mkdir(parents=True)
    io.write_json(root / '数据/中间/T01/manifest.json', {'head_x': 1000})
    align = {'template_trip': 'T01', 'g': [{'x': 0, 'u': 0}, {'x': 100000, 'u': 100000}]}
    p = root / '对应点.csv'
    pairs.write_rows([{'id': 'P01', 'x_t01': 5000, 'y_t01': 300, 'x_t02': 4003, 'y_t02': 304},
                      {'id': 'P02', 'x_t01': 9000, 'y_t01': 300, 'x_t02': '', 'y_t02': '', 'note': 'skip'}], p)
    r = score.score_align(p, align)
    assert r['n'] == 1 and r['median_px'] == 5.0          # (3, 4) → 5 px


def test_pairs_picks_compact_feature_not_edge():
    from railside.eval import pairs
    im = np.full((200, 200), 60, np.float32)
    cv2.circle(im, (60, 100), 6, 250, -1)                 # 螺栓头大小的亮点：两个方向都定得住
    im[:, 150:] = 200                                     # 直边：沿边方向定不住
    b = cv2.GaussianBlur(im, (0, 0), 3)
    assert pairs._distinct(b, 60, 100) < 0.7
    assert pairs._distinct(b, 150, 100) > 0.9
