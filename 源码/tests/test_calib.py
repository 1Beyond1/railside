"""S3 标定：合成图上的圆检测、比例计算、插值外推、缺数据退化。"""
import numpy as np
import pytest

from railside import calib


def _ring_pts(cx, cy, a, b, n=180, ang=0.0, noise=0.0, outliers=0, seed=0):
    """理想椭圆上的点（可加噪声/离群点），用来单独测拟合。"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    c, s = np.cos(ang), np.sin(ang)
    P = np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)
    if noise:
        P = P + np.random.default_rng(seed).normal(0, noise, P.shape)
    if outliers:
        rng = np.random.default_rng(seed + 1)
        k = rng.integers(0, n, outliers)
        P[k] += rng.normal(0, 25, (outliers, 2))
    return P


def test_fit_ellipse_recovers_shape():
    P = _ring_pts(500, 420, 402, 415, noise=2.0, outliers=20)
    f = calib.fit_ellipse(P)
    assert abs(f['cx'] - 500) < 2 and abs(f['cy'] - 420) < 2
    assert abs(f['a'] - 402) < 6 and abs(f['b'] - 415) < 6
    assert f['n'] > 120


def test_axis_ab_swaps_for_rotated_ellipse():
    assert tuple(round(v) for v in calib._axis_ab(200, 100, 0)) == (200, 100)
    assert tuple(round(v) for v in calib._axis_ab(200, 100, 90)) == (100, 200)


def test_ring_points_finds_dark_circle():
    yy, xx = np.mgrid[:400, :400]
    r = np.hypot(xx - 200.0, yy - 180.0)
    im = (140 - 110 * np.exp(-((r - 100.0) / 4.0) ** 2)).clip(0, 255).astype(np.uint8)
    pts = calib._ring_points(im, 200.0, 180.0, 75.0, 125.0)
    rr = np.hypot(pts[:, 0] - 200, pts[:, 1] - 180)
    assert np.all(np.abs(rr - 100) < 2)


def test_pick_quad_ignores_clutter():
    true = [(1000, 2800, 420), (3466, 2800, 420), (14467, 2800, 420), (16933, 2800, 420)]
    junk = [(6000, 2600, 400), (9000, 3000, 430), (2500, 2200, 420)]
    cands = junk[:1] + [true[0], true[2]] + junk[1:] + [true[1], true[3]]
    sc, quad = calib.pick_quad(cands, wheelbase=(2200, 2800), dy_max=60, center=(13000, 14000))
    assert sc is not None
    assert quad == sorted(true)


def test_pick_quad_returns_none_when_incomplete():
    _, quad = calib.pick_quad([(0, 2800, 400), (2466, 2800, 400)], wheelbase=(2200, 2800))
    assert quad == []


def test_sx_at_interpolates_and_extrapolates():
    curve = [{'x': 100.0, 'mm_per_px': 0.230}, {'x': 300.0, 'mm_per_px': 0.234},
             {'x': 600.0, 'mm_per_px': 0.240}]
    v = calib.sx_at({'sx_curve': curve}, [0, 200, 450, 800])
    assert v[0] == pytest.approx(0.228)          # 左端外推，斜率 (0.234-0.230)/200
    assert v[1] == pytest.approx(0.232)
    assert v[2] == pytest.approx(0.237)
    assert v[3] == pytest.approx(0.244)          # 右端外推，斜率 (0.240-0.234)/300


def test_sx_at_single_knot_is_constant():
    assert list(calib.sx_at({'sx_curve': [{'x': 0, 'mm_per_px': 0.235}]}, [10, 20])) == [0.235, 0.235]


def test_make_curve_sorts_dedups_and_keeps_one_knot():
    s = [{'x': 300, 'mm_per_px': 0.24}, {'x': 100, 'mm_per_px': 0.23},
         {'x': 300.04, 'mm_per_px': 0.26}]
    c = calib.make_curve(s)
    assert [p['x'] for p in c] == [100.0, 300.0]
    assert c[1]['mm_per_px'] == pytest.approx(0.25) and c[1]['n'] == 2
    assert calib.make_curve([]) == []
    assert len(calib.make_curve([{'x': 5, 'mm_per_px': 0.2}])) == 1


def test_spread_basic():
    s = calib._spread([0.23, 0.23, 0.23])
    assert s['sd'] == 0 and s['range_pct'] == 0 and s['n'] == 3
    assert calib._spread([]) is None


# ---------------------------------------------------------------- 合成整车

class FakeTrip:
    """只有 region()，按趟坐标取区域、按 step 缩小。"""

    def __init__(self, im):
        self.im = im

    def region(self, x0, x1, y0=0, y1=None, step=1):
        y1 = self.im.shape[0] if y1 is None else y1
        x0, x1 = max(0, int(x0)), min(self.im.shape[1], int(x1))
        a = self.im[max(0, int(y0)):y1, x0:x1]
        if step > 1:
            a = a[::step, ::step]
        return a.copy()


def _dark_ring(im, cx, cy, a, b, depth, sigma):
    """加一圈椭圆暗环：半轴 a、b，径向宽度 sigma（最暗在椭圆上）。"""
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    q = np.hypot((xx - cx) / a, (yy - cy) / b)
    return (im.astype(np.float32) - depth * np.exp(-(((q - 1.0) * a) / sigma) ** 2)).clip(0, 255)


def _fake_car(w=19000, h=900, y=400, a=104.0, b=107.0, seed=3):
    """一节车 4 个轴端：左平盖（亮面 + 外圈暗环），右球面盖（暗面 + 小凸台 + 外圈暗环）。"""
    rng = np.random.default_rng(seed)
    im = (110 + rng.normal(0, 3, (h, w))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    xs = [1000, 3466, 14467, 16933]
    for k, cx in enumerate(xs):
        face = ((xx - cx) / a) ** 2 + ((yy - y) / b) ** 2 <= 1.0
        im[face] = 140 if k % 2 == 0 else 118
        if k % 2:
            hub = ((xx - cx) / (a * 0.30)) ** 2 + ((yy - y) / (b * 0.30)) ** 2 <= 1.0
            im[hub] = 175
            im = _dark_ring(im, cx, y, a * 0.30 + 8, (a * 0.30 + 8) * b / a, 80, 3.0)
        im = _dark_ring(im, cx, y, a + 8, (a + 8) * b / a, 95, 5.0)
    return im.clip(0, 255).astype(np.uint8), xs


def test_detect_axles_synthetic_car():
    im, xs = _fake_car()
    trip = FakeTrip(im)
    manifest = {'train_panels': [{'car': 1, 'x0': 0, 'x1': im.shape[1]}]}
    bogies, cars = calib.detect_axles(
        trip, manifest, y_band=(300, 500), step=2,
        hough=dict(dp=1.2, minDist=80, param1=80, param2=16, minRadius=44, maxRadius=64),
        quad=dict(wheelbase=(2200, 2800), dy_max=60, center=(13000, 14000)),
        band_cap=(88, 130), band_hub=(18, 48), crop_cap=160, crop_hub=80,
        wb_range=(2200, 2800))
    assert len(bogies) == 2
    for k, bog in enumerate(bogies):
        assert bog['ok'], bog['left']
        assert bog['wheelbase_px'] == pytest.approx(2466, abs=8)
        assert bog['left']['x'] == pytest.approx(xs[2 * k], abs=8)
        assert bog['right']['x'] == pytest.approx(xs[2 * k + 1], abs=8)
        assert bog['left']['ratio'] == pytest.approx(107 / 104, abs=0.04)


def test_detect_axles_without_disks_returns_nothing():
    im = np.full((600, 4000), 120, np.uint8)
    trip = FakeTrip(im)
    manifest = {'train_panels': [{'car': 1, 'x0': 0, 'x1': 4000}]}
    bogies, cars = calib.detect_axles(
        trip, manifest, y_band=(200, 400), step=2,
        hough=dict(dp=1.2, minDist=80, param1=80, param2=25, minRadius=44, maxRadius=64))
    assert bogies == [] and cars[0]['found'] is False
