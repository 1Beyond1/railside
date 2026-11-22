from pathlib import Path

import numpy as np
import pytest

from railside import cli, io
from railside import segment as seg

FIX = Path(__file__).parent / 'fixtures' / 'labels_smoke.json'
HAS_T01 = (io.trip_dir('T01') / 'trip.json').exists() and (io.trip_dir('T01') / 'manifest.json').exists()


@pytest.mark.parametrize('s', [1, 4])
def test_win_trip_roundtrip(s):
    x, y = seg.win_to_trip([0, 319, 17], [0, 319, 5], 60000, 2880, s)
    assert list(x) == [60000, 60000 + 319 * s, 60000 + 17 * s] and list(y) == [2880, 2880 + 319 * s, 2880 + 5 * s]
    px, py = seg.trip_to_win(x, y, 60000, 2880, s)
    assert list(px) == [0, 319, 17] and list(py) == [0, 319, 5]
    assert seg.parse_win_name(seg.win_name('T01', s, 60000, 2880)) == ('T01', s, 60000, 2880)


def test_window_grid_per_scale():
    xs, ys = seg.window_grid(56400, 60000, 1024)
    assert ys[0] == 0 and ys[-1] + 1024 == 4096 and all(b - a <= 768 for a, b in zip(ys, ys[1:]))
    assert xs[0] == 56400 and xs[-1] + 1024 == 60000
    # scale 4、imgsz 1024：整个高度正好一行，横向每窗覆盖 4096、步长 3072
    xs, ys = seg.window_grid(56400, 70000, 1024, 4)
    assert ys == [0] and xs[0] == 56400 and xs[1] - xs[0] == 3072 and xs[-1] + 4096 == 70000
    # scale 4、imgsz 320：多行，左上角都在 4 的整数倍上
    xs, ys = seg.window_grid(56400, 70000, 320, 4)
    assert ys[0] == 0 and ys[-1] + 1280 == 4096 and all(y % 4 == 0 for y in ys)


def test_crop_pads_short_strip():
    strip = np.full((1024, 320), 7, np.uint8)
    c = seg.crop(strip, 4 * 960, 320, 4)
    assert c.shape == (320, 320) and c[:64].min() == 7 and c[64:].max() == 0


def test_poly_labels_normalized_and_clipped():
    # 整个落在窗口内：反归一化后回到原多边形（±1 px × scale）
    for s, wy0 in ((1, 1920), (4, 1280)):
        rect = [[60100, 2000], [60180, 2000], [60180, 2080], [60100, 2080]]
        (c,) = seg.poly_to_labels(rect, 60000, wy0, 320, s)
        x, y = seg.win_to_trip(c[:, 0] * 320, c[:, 1] * 320, 60000, wy0, s)
        assert abs(x.min() - 60100) <= s and abs(x.max() - 60180) <= s
        assert abs(y.min() - 2000) <= s and abs(y.max() - 2080) <= s
    # 跨出窗口：裁到 [0, 1]
    big = [[59900, 1900], [60500, 1900], [60500, 2500], [59900, 2500]]
    (c,) = seg.poly_to_labels(big, 60000, 1920, 320)
    assert c.min() >= 0 and c.max() <= 1 and c.max() > 0.99
    # 不相交：没有轮廓
    assert seg.poly_to_labels(big, 70000, 0, 320) == []


def test_class_table(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    parts = [{'type': 'BOX'}, {'type': 'BOLT'}, {'type': 'BOX'}]
    assert seg.class_names(parts) == ['BOLT', 'BOX']
    assert seg.scales_for(['BOLT', 'BOX', 'WHEEL']) == [1, 4, 4]
    (tmp_path / '主数据').mkdir()
    csvp = tmp_path / '主数据' / '零件类型.csv'
    csvp.write_text('type,name,scale\nWHEEL,车轮,4\nBOX,箱,2\nBOLT,螺栓,\n', encoding='utf-8')
    assert seg.class_names(parts) == ['WHEEL', 'BOX', 'BOLT']
    assert seg.scales_for(['WHEEL', 'BOX', 'BOLT']) == [4, 2, 1]
    csvp.write_text('type,name\nBOX,箱\nBOLT,螺栓\n', encoding='utf-8')  # 没有 scale 列
    assert seg.scales_for(['BOX', 'BOLT']) == [4, 1]
    with pytest.raises(SystemExit):
        seg.class_names(parts + [{'type': 'NUT'}])


def test_merge_dets():
    d = lambda c, s, b: {'cls': c, 'score': s, 'bbox_xy': b, 'poly_xy': []}
    dets = [
        d('WHEEL', 0.6, [0, 0, 1000, 800]),       # 完整的车轮
        d('WHEEL', 0.9, [600, 0, 1000, 800]),     # 窗口边截断的一块：IoU 0.4，但全落在大框里 → 合并，留高分
        d('WHEEL', 0.5, [2000, 0, 3000, 800]),    # 另一个车轮
        d('BOX', 0.7, [600, 0, 1000, 800]),       # 不同类不合并
        d('BOLT', 0.8, [0, 0, 10, 10]),
        d('BOLT', 0.7, [3, 0, 13, 10]),           # 交集占 0.7 → 不超过阈值，保留
        d('BOLT', 0.6, [2, 0, 12, 10]),           # 交集占 0.8 → 合并
    ]
    kept = seg.merge_dets(dets)
    assert sorted((k['cls'], k['score']) for k in kept) == [
        ('BOLT', 0.7), ('BOLT', 0.8), ('BOX', 0.7), ('WHEEL', 0.5), ('WHEEL', 0.9)]
    assert seg.merge_dets([]) == []


def test_mask_to_poly():
    m = np.zeros((50, 50), bool)
    m[10:20, 5:30] = True
    p = seg.mask_to_poly(m)
    assert p.min(0).tolist() == [5, 10] and p.max(0).tolist() == [29, 19]
    assert seg.mask_to_poly(np.zeros((5, 5), bool)) is None


def test_train_rejects_t02():
    with pytest.raises(SystemExit) as e:
        cli.main(['train', '--trip', 'T02', '--labels', str(FIX), '--export-only'])
    assert 'T02' in str(e.value)


@pytest.mark.skipif(not HAS_T01, reason='没有 T01 的 trip.json / manifest.json')
def test_build_dataset_t01(tmp_path, monkeypatch):
    YAML = pytest.importorskip('ultralytics.utils').YAML
    monkeypatch.setattr(seg, 'type_table', lambda: None)  # 用默认表：BOLT=1，BOX=4
    out = tmp_path / 'ds'
    st = seg.build_dataset(FIX, 'T01', 320, out)
    assert st['names'] == ['BOLT', 'BOX'] and st['scales'] == {'BOLT': 1, 'BOX': 4}
    want_cls = {1: 0, 4: 1}  # 每种尺度的窗口里只有该尺度的类别
    m = io.read_json(io.trip_dir('T01') / 'manifest.json')
    vx0, vx1 = m['train_panels'][-1]['x0'], m['train_panels'][-1]['x1']
    for sp in ('train', 'val'):
        imgs = sorted((out / 'images' / sp).glob('*.jpg'))
        assert len(imgs) == st[sp]['images'] > 0
        assert st[sp]['background'] >= 2 and all(v > 0 for v in st[sp]['instances'].values())
        seen, n_empty = set(), 0
        for p in imgs:
            trip, s, wx0, wy0 = seg.parse_win_name(p.name)
            span = 320 * s
            seen.add(s)
            assert trip == 'T01' and 0 <= wy0 <= 4096 - span and wy0 % s == 0
            if sp == 'val':
                assert vx0 <= wx0 and wx0 + span <= vx1
            else:
                assert wx0 + span <= vx0 or wx0 >= vx1
            assert io.imread(p).shape == (320, 320)
            txt = (out / 'labels' / sp / p.name).with_suffix('.txt').read_text(encoding='utf-8').strip().splitlines()
            n_empty += not txt
            for ln in txt:
                v = ln.split()
                assert int(v[0]) == want_cls[s]
                xy = np.array(v[1:], float)
                assert len(xy) % 2 == 0 and len(xy) >= 6
                assert xy.min() >= 0 and xy.max() <= 1
        assert seen == {1, 4} and n_empty == st[sp]['background']
    y = YAML.load(out / 'data.yaml')
    assert y['names'] == {0: 'BOLT', 1: 'BOX'} and 'path' not in y


@pytest.mark.skipif(not HAS_T01, reason='没有 T01 的 trip.json / manifest.json')
def test_scaled_window_matches_trip_pixels():
    # scale 4 窗口里的像素 = 趟图对应 4×4 块的平均
    from railside.frames import Trip
    t = Trip.from_id('T01')
    strip = seg.read_strip(t, 60000, 320, 4)
    win = seg.crop(strip, 1280, 320, 4)
    px, py = 100, 50
    x, y = seg.win_to_trip(px, py, 60000, 1280, 4)
    block = t.region(int(x), int(x) + 4, int(y), int(y) + 4)
    assert abs(int(win[py, px]) - block.mean()) <= 1
