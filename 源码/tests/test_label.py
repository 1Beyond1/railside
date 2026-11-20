"""label.py：几何换算、SAM 窗口分组、搬运的映射与 NCC 精修（合成图，不跑 SAM）。"""
import cv2
import numpy as np

from railside import label


class FakeTrip:
    """按趟坐标取区域的假 Trip：底图是一张数组，越界补 0。"""

    def __init__(self, im):
        self.im = im

    def region(self, x0, x1, y0=0, y1=4096, step=1):
        h, w = self.im.shape
        out = np.zeros((y1 - y0, x1 - x0), np.uint8)
        a0, a1, b0, b1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
        if a1 > a0 and b1 > b0:
            out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = self.im[b0:b1, a0:a1]
        if step > 1:
            out = cv2.resize(out, (max(1, out.shape[1] // step), max(1, out.shape[0] // step)), interpolation=cv2.INTER_AREA)
        return out


def textured(h, w, seed=0):
    rng = np.random.default_rng(seed)
    im = rng.integers(0, 255, (h // 8, w // 8)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 2)


def test_mask_to_poly_rectangle_in_trip_coords():
    m = np.zeros((200, 300), np.uint8)
    m[50:150, 80:220] = 1
    poly = label.mask_to_poly(m, (1000, 2000), 16)
    b = label.box_of(poly)
    assert abs(b[0] - 1080) <= 2 and abs(b[1] - 2050) <= 2
    assert abs(b[2] - 1220) <= 3 and abs(b[3] - 2150) <= 3
    assert len(poly) <= 16
    assert label.mask_to_poly(np.zeros((10, 10), np.uint8), (0, 0), 16) is None


def test_poly_mask_roundtrip_area():
    poly = label.ellipse_poly([100, 200, 300, 300], n=32)
    m = label.poly_mask(poly, (100, 200), (100, 200))
    area = m.sum()
    assert abs(area - np.pi * 100 * 50) / (np.pi * 100 * 50) < 0.05


def test_plan_windows_groups_small_and_isolates_big():
    qs = [{'id': 1, 'box': [1000, 1000, 1060, 1060]}, {'id': 2, 'box': [1300, 1500, 1360, 1560]},
          {'id': 3, 'box': [9000, 1000, 9060, 1060]}, {'id': 4, 'box': [2000, 1000, 5600, 3900]}]
    wins = label.plan_windows(qs)
    ids = sorted(sorted(q['id'] for q in w[4]) for w in wins)
    assert ids == [[1, 2], [3], [4]]
    for x0, y0, x1, y1, group in wins:
        assert 0 <= y0 < y1 <= 4096
        for q in group:
            b = q['box']
            assert x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1


def test_axle_map_maps_both_axles():
    src = ((1000.0, 2800.0), (11000.0, 2790.0))
    dst = ((50000.0, 2810.0), (59800.0, 2800.0))
    sx, ox, oy = label.axle_map(src, dst)
    assert abs(sx - 0.98) < 1e-9
    assert abs(ox + sx * 1000 - 50000) < 1e-6 and abs(ox + sx * 11000 - 59800) < 1e-6
    assert abs(oy - 10) < 1e-9
    b = label.map_box([1000, 100, 11000, 200], (sx, ox, oy))
    assert abs(b[0] - 50000) < 1e-6 and abs(b[2] - 59800) < 1e-6 and b[1] == 110


def test_ncc_refine_recovers_shift():
    im = textured(4096, 6000)
    t = FakeTrip(im)
    src = [1000, 1500, 1300, 1800]
    # 目标处是同一块纹理平移 (+2000+37, −23)：预测框故意偏 37、−23
    patch = im[1400:1900, 900:1400]
    im[1400 - 23:1900 - 23, 2900 + 37:3400 + 37] = patch
    pred = [3000, 1500, 3300, 1800]
    dx, dy, v = label.ncc_refine(t, src, pred, 1.0, reach=80, step=1)
    assert abs(dx - 37) <= 1 and abs(dy + 23) <= 1
    assert v > 0.95


def test_make_part_modes_and_fallback():
    q = {'id': 5001, 'type': 'BOLT', 'box': [10, 20, 50, 60], 'bogie': 'C3B1', 'source': 'ai'}
    p = label.make_part(q, None)
    assert p['flags'] == ['sam_empty'] and p['bbox'] == [10, 20, 51, 61] and p['verified'] is False
    p = label.make_part(q | {'mode': 'ellipse'}, None)
    assert len(p['poly']) == 16 and 'flags' not in p
    p = label.make_part(q, ([[10, 20], [50, 20], [50, 60]], 0.9, 0.05))
    assert p['flags'] == ['sam_small'] and p['sam_score'] == 0.9
