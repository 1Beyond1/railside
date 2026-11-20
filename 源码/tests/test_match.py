"""S6b 匹配的测试：合成图上的平移、比例、缺失、贴边、亮度；分割配对；一条端到端。

单元测试用 FakeTrip（只有 region()，接口和 frames.Trip 一样），不读真实数据；
端到端在 tmp_path 里造 3 帧原图 + 模板 + 对齐，几秒跑完。
"""
import numpy as np
import cv2
import pytest

from railside import cli, io, match as M


# ---------------------------------------------------------------- 夹具

class FakeTrip:
    """只实现 region() 的假趟；坐标语义和 frames.Trip 一致（未翻转、缺位置补 0、step 区域平均）。"""

    def __init__(self, im):
        self.im = im
        self.height, self.width = im.shape

    def region(self, x0, x1, y0=0, y1=None, step=1):
        y1 = self.height if y1 is None else y1
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        out = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
        xa = max(x0, 0)
        while xa < min(x1, self.width):
            c1 = min(self.width, xa + (x1 - xa))
            out[:, xa - x0: xa - x0 + (c1 - xa)] = self.im[y0:y1, xa:c1]
            xa = c1
        if step > 1:
            h, w = out.shape
            out = cv2.resize(out, (max(1, w // step), max(1, h // step)), interpolation=cv2.INTER_AREA)
        return out


def photo(h=240, w=300, seed=0):
    """有结构、有纹理的“零件局部图”：随机底 + 粗线 + 圆 + 小方块。

    峰唯一（能定位），又有足够的细节（转角 5° 就能测出来）。
    """
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    im = np.clip(base * 55 + 128, 0, 255).astype(np.uint8)
    for _ in range(4):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(im, p0, p1, int(rng.integers(20, 240)), int(rng.integers(3, 9)))
    cv2.circle(im, (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))),
               int(min(h, w) // 6), int(rng.integers(30, 220)), -1)
    for _ in range(6):        # 小方块：给转角提供高频细节
        x0, y0 = int(rng.integers(0, w - 14)), int(rng.integers(0, h - 14))
        cv2.rectangle(im, (x0, y0), (x0 + 13, y0 + 13), int(rng.integers(20, 240)), -1)
    return im


def place(canvas, tex, x0, y0):
    canvas[y0:y0 + tex.shape[0], x0:x0 + tex.shape[1]] = tex


def full_mask(shape):
    return np.full(shape, 255, np.uint8)


# ---------------------------------------------------------------- 局部搜索（在对齐后的图上）

from railside.coords import AlignMap

IDENT = AlignMap([0, 10000], [0, 10000])


def test_search_finds_part_and_reports_offset():
    tex = photo(240, 300, seed=1)
    canvas = np.full((700, 1000), 100, np.uint8)
    place(canvas, tex, 300, 200)
    t = FakeTrip(canvas)
    r = M._search(t, IDENT, tex, full_mask(tex.shape), (300, 200), (0.0, 0.0), 40)
    assert r['score'] > 0.9 and abs(r['du']) < 0.6 and abs(r['dv']) < 0.6
    # 模板位置偏了 (−12, +9)：要能找回来，偏移量带正确的符号
    r = M._search(t, IDENT, tex, full_mask(tex.shape), (288, 209), (0.0, 0.0), 40)
    assert abs(r['du'] - 12) < 0.6 and abs(r['dv'] + 9) < 0.6
    # 父件平移当起点：结果是总平移，不是相对起点
    r = M._search(t, IDENT, tex, full_mask(tex.shape), (288, 209), (10.0, -8.0), 20)
    assert abs(r['du'] - 12) < 0.6 and abs(r['dv'] + 9) < 0.6


def test_search_scale_absorbed_by_alignment():
    """新趟横向拉长 4%、纵向不变：对齐映射吸收横向比例后，高大件的纵向位置不能被带偏。

    旧做法各向同性地搜比例，3300 px 高的转向架纵向偏了 50 px（真实 T02）。"""
    tex = photo(900, 400, seed=2)                        # 高大于宽，纵向误差最显眼
    wide = cv2.resize(tex, (416, 900), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((1200, 1400), 100, np.uint8)
    place(canvas, wide, 520, 150)                        # x = 1.04·u → u0 = 500
    amap = AlignMap([0, 1040 * 10], [0, 1000 * 10])
    r = M._search(FakeTrip(canvas), amap, tex, full_mask(tex.shape), (500, 150), (0.0, 0.0), 40)
    assert r['score'] > 0.9
    assert abs(r['du']) < 1.0 and abs(r['dv']) < 1.0


def test_search_big_patch_coarse_then_refine():
    """长边超过 COARSE_MAX 的局部图走降采样粗搜 + 中心块精修，位置要准。"""
    tex = photo(700, 1100, seed=3)
    canvas = np.full((1100, 2600), 90, np.uint8)
    place(canvas, tex, 500, 150)
    t = FakeTrip(canvas)
    r = M._search(t, IDENT, tex, full_mask(tex.shape), (492, 156), (0.0, 0.0), 60)
    assert r['score'] > 0.8
    assert abs(r['du'] - 8) < 1.5 and abs(r['dv'] + 6) < 1.5


def test_search_missing_returns_low_score():
    tex = photo(240, 300, seed=4)
    canvas = np.full((700, 1000), 100, np.uint8)
    place(canvas, photo(240, 300, seed=99), 300, 200)   # 那个位置是别的东西
    t = FakeTrip(canvas)
    r = M._search(t, IDENT, tex, full_mask(tex.shape), (300, 200), (0.0, 0.0), 40)
    assert r is None or r['score'] < M.ACCEPT_PIXEL   # 低于第二层门槛：不能被当成找到


def test_search_at_image_edge():
    """贴边的件：局部图超出图像上边的部分裁掉再搜，不能报错。"""
    tex = photo(240, 300, seed=5)
    canvas = np.full((300, 800), 90, np.uint8)
    place(canvas, tex[30:], 200, 0)                      # 图上只看得到下面 210 行
    t = FakeTrip(canvas)
    r = M._search(t, IDENT, tex, full_mask(tex.shape), (200, -30), (0.0, 0.0), 20)
    assert r is not None and r['score'] > 0.8
    assert abs(r['du']) < 1 and abs(r['dv']) < 1


def test_search_brightness_insensitive():
    tex = photo(240, 300, seed=6)
    canvas = np.full((700, 1000), 100, np.uint8)
    place(canvas, np.clip(tex.astype(np.float32) * 0.6 + 18, 0, 255).astype(np.uint8), 300, 200)
    r = M._search(FakeTrip(canvas), IDENT, tex, full_mask(tex.shape), (300, 200), (0.0, 0.0), 40)
    assert r['score'] > 0.9 and abs(r['du']) < 0.6


def test_aligned_region_inverts_mapping():
    """aligned_region 把新趟按 x = 1.03·u + 20、y = v + 5 采回模板坐标，应和模板一致。"""
    tex = photo(300, 400, seed=8)
    canvas = np.full((400, 700), 100, np.uint8)
    place(canvas, cv2.resize(tex, (412, 300), interpolation=cv2.INTER_LINEAR), 20 + 103, 5 + 50)
    amap = AlignMap([0, 1030 * 5], [-20 / 1.03, (1030 * 5 - 20) / 1.03], [5, 5])   # dy = 新趟 y − 模板 v
    R = M.aligned_region(FakeTrip(canvas), amap, 100, 500, 50, 350)
    d = np.abs(R[20:-20, 20:-20].astype(float) - tex[20:-20, 20:-20].astype(float))
    assert d.mean() < 8


# ---------------------------------------------------------------- 端到端

W, H = 2000, 4096


def _frames(root, trip, canvas, head_x=0):
    """把 canvas 切成 2000 宽的帧存成原图，写 trip.json / manifest.json。"""
    src = root / '数据' / '原始' / trip / 'camImg'
    src.mkdir(parents=True, exist_ok=True)
    n = canvas.shape[1] // W
    for i in range(n):
        io.imwrite(src / f'{i + 1}.jpg', canvas[:, i * W:(i + 1) * W], quality=95)
    io.write_json(root / '数据' / '中间' / trip / 'trip.json',
                  {'trip': trip, 'src_dir': f'数据/原始/{trip}/camImg', 'camera': 'camImg', 'n_frames': n})
    io.write_json(root / '数据' / '中间' / trip / 'manifest.json',
                  {'trip': trip, 'src_dir': f'数据/原始/{trip}/camImg', 'n_frames': n, 'frame_w': W, 'frame_h': H,
                   'width_px': n * W, 'height_px': H, 'display_flip': True, 'train_x0': 0, 'train_x1': n * W,
                   'head_x': head_x, 'train_frames': [1, n], 'blank': [], 'head_protect': {'x0': 0, 'x1': 8000},
                   'train_panels': [{'panel': 1, 'car': 1, 'x0': 0, 'x1': n * W, 'width_px': n * W, 'height_px': H}],
                   'seams': [], 'bad_seams': 0})


def _tpl_part(root, code, typ, tex, poly_local, origin, parent=None, box_local=None):
    """把局部图、掩膜写进模板目录，返回 template.json 里的一项。"""
    h, w = tex.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(poly_local, float)).astype(np.int32)], 255)
    d = root / '数据' / '成品' / '模板' / 'v1'
    io.imwrite(d / 'crops' / f'{code}.png', tex)
    io.imwrite(d / 'masks' / f'{code}.png', mask)
    ys, xs = np.nonzero(mask)
    return {'code': code, 'type': typ, 'parent': parent, 'car': 1, 'zone': 'B1',
            'bbox_uv': [int(xs.min() + origin[0]), int(ys.min() + origin[1]),
                        int(xs.max() + origin[0] + 1), int(ys.max() + origin[1] + 1)],
            'poly_uv': [[round(x + origin[0], 1), round(y + origin[1], 1)] for x, y in poly_local],
            'centroid_uv': [round(float(xs.mean()) + origin[0], 1), round(float(ys.mean()) + origin[1], 1)],
            'area_px': int(len(xs)), 'angle_deg': 0.0, 'size_mm': None,
            'crop': io.rel(d / 'crops' / f'{code}.png'), 'mask': io.rel(d / 'masks' / f'{code}.png'),
            'crop_origin_uv': [int(origin[0]), int(origin[1])], 'feat': {}, 'source': 'human',
            'verified': True, 'label_id': len(code)}


def _template(root, parts):
    io.write_json(root / '数据' / '成品' / '模板' / 'v1' / 'template.json',
                  {'version': 'v1', 'trip': 'T01', 'head_x': 0, 'anchors': [], 'parts': parts, 'calib': None})


def _align(root, trip, shift_x, shift_y=0, qc2=()):
    """本趟 x = u + shift_x，y = v + shift_y（内容整体平移）；u 是模板坐标。"""
    xs = [0, 6000]
    io.write_json(root / '数据' / '中间' / trip / 'align.json',
                  {'trip': trip, 'template_ver': 'v1', 'g': [{'x': x, 'u': x - shift_x, 'dy': shift_y} for x in xs],
                   'qc2': list(qc2), 'residual_px': {'median': 1.0, 'p95': 2.0}})


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """临时项目：两趟原图（T02 = T01 平移 + 一处改动）、一个模板、对齐、标定。"""
    root = tmp_path / 'proj'
    for name, val in (('ROOT', root), ('DATA', root / '数据'), ('MID', root / '数据' / '中间'),
                      ('OUT', root / '数据' / '成品')):
        monkeypatch.setattr(io, name, val)
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (512, 750), dtype=np.uint8)
    canvas = cv2.resize(base, (6000, H), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 2)
    cv2.line(canvas, (300, 900), (5200, 3300), 20, 9)
    cv2.circle(canvas, (3300, 2100), 420, 240, -1)
    big = canvas[1800:2300, 1200:1650].copy()        # 大件：500×450
    small = canvas[2600:2700, 3800:3900].copy()      # 螺栓：100×100
    _frames(root, 'T01', canvas)
    # T02：内容整体平移 (+37, +8)
    moved = cv2.warpAffine(canvas, np.array([[1, 0, 37], [0, 1, 8]], np.float32), (6000, H),
                           borderMode=cv2.BORDER_REPLICATE)
    _frames(root, 'T02', moved)
    parts = [_tpl_part(root, 'C1-B1-BOGIE-01', 'BOGIE', big,
                       [[8, 8], [442, 8], [442, 492], [8, 492]], (1200, 1800)),
             _tpl_part(root, 'C1-B1-BOLT-01', 'BOLT', small,
                       [[6, 6], [94, 6], [94, 94], [6, 94]], (3800, 2600), parent='C1-B1-BOGIE-01')]
    _template(root, parts)
    _align(root, 'T02', shift_x=37, shift_y=8)
    io.write_json(root / '数据' / '中间' / 'T02' / 'calib.json',
                  {'sx_curve': [{'x': 0, 'mm_per_px': 0.24}, {'x': 6000, 'mm_per_px': 0.24}],
                   'sy_mm_per_px': 0.30, 'refs': [], 'source': '测试'})
    (root / '主数据').mkdir(parents=True, exist_ok=True)
    (root / '主数据' / '零件类型.csv').write_text(
        'type,name,parent_type,desc,ncc_min,shift_mm_max,dtheta_max_deg,area_change_max,scale\n'
        'BOGIE,转向架,,走行部,0.5,6,5,0.4,4\nBOLT,螺栓,BOGIE,紧固件,0.5,3,8,0.4,1\n', encoding='utf-8')
    return root


def test_end_to_end_match_and_compare(proj, capsys):
    assert cli.main(['match', 'T02']) is None
    pj = io.read_json(io.MID / 'T02' / 'parts.json')
    assert len(pj['instances']) == 2 and pj['extra'] == []
    by = {i['code']: i for i in pj['instances']}
    assert all(i['found_by'] == 'pixel' for i in pj['instances'])
    # 本趟内容整体 +(37, 8)，对齐映射已经解释了；剩下的残差应该很小
    box = by['C1-B1-BOGIE-01']['bbox_xy']   # 多边形比局部图内缩 8 px
    assert abs(box[0] - (1200 + 37 + 8)) < 6 and abs(box[1] - (1800 + 8 + 8)) < 6
    assert by['C1-B1-BOGIE-01']['src_frames'] == [1, 1]   # 整个落在第 1 帧（x < 2000）
    for i in pj['instances']:
        assert (io.ROOT / i['crop']).is_file() and (io.ROOT / i['mask']).is_file()

    assert cli.main(['compare', 'T02']) is None
    cj = io.read_json(io.MID / 'T02' / 'compare.json')
    assert len(cj['items']) == 2 and all(i['status'] == 'ok' for i in cj['items']), cj['items']
    assert cj['foreign'] == []
    it = {i['code']: i for i in cj['items']}['C1-B1-BOLT-01']
    assert it['ncc'] > 0.6 and abs(it['dtheta_deg']) < 1.5
    assert it['shift_mm'] is not None and it['shift_mm'] < 1.0     # 平移已对齐，残余接近 0
    out = capsys.readouterr().out
    assert '"step": "match"' in out and '"step": "compare"' in out


def test_match_seg_pairing_and_extra(proj, monkeypatch):
    io.write_json(io.MID / 'T02' / 'seg.json', {
        'model': {'name': 'test'}, 'dets': [
            {'cls': 'BOLT', 'score': 0.91, 'bbox_xy': [3830, 2630, 3930, 2730],
             'poly_xy': [[3832, 2632], [3928, 2632], [3928, 2728], [3832, 2728]]},
            {'cls': 'BOLT', 'score': 0.7, 'bbox_xy': [1000, 3000, 1100, 3100],
             'poly_xy': [[1000, 3000], [1100, 3000], [1100, 3100], [1000, 3100]]}]})
    cli.main(['match', 'T02'])
    pj = io.read_json(io.MID / 'T02' / 'parts.json')
    bolt = [i for i in pj['instances'] if i['type'] == 'BOLT'][0]
    assert bolt['found_by'] == 'seg' and bolt['seg_score'] == 0.91
    assert abs(bolt['centroid_xy'][0] - 3880) < 5
    assert len(pj['extra']) == 1 and pj['extra'][0]['bbox_xy'] == [1000, 3000, 1100, 3100]


def test_match_parent_fallback(proj, monkeypatch):
    """第二层门槛抬高到不可能达到：第三层按父件相对位置再找一次，记 parent。"""
    monkeypatch.setattr(M, 'ACCEPT_PIXEL', 1.01)
    cli.main(['match', 'T02'])
    by = {i['code']: i['found_by'] for i in io.read_json(io.MID / 'T02' / 'parts.json')['instances']}
    assert by['C1-B1-BOLT-01'] == 'parent' and by['C1-B1-BOGIE-01'] == 'pixel'


def test_match_missing(proj):
    """本趟把螺栓那块抹掉：记 missing，不写局部图。"""
    t2 = io.read_json(io.MID / 'T02' / 'manifest.json')
    src = io.ROOT / t2['src_dir']
    canvas = np.hstack([io.imread(src / f'{i + 1}.jpg') for i in range(t2['n_frames'])])
    canvas[2620:2760, 3820:3960] = 105
    _frames(io.ROOT, 'T02', canvas)
    cli.main(['match', 'T02'])
    by = {i['code']: i for i in io.read_json(io.MID / 'T02' / 'parts.json')['instances']}
    assert by['C1-B1-BOLT-01']['found_by'] == 'missing'
    assert by['C1-B1-BOLT-01']['crop'] is None and by['C1-B1-BOLT-01']['area_px'] == 0
    assert by['C1-B1-BOGIE-01']['found_by'] == 'pixel'

