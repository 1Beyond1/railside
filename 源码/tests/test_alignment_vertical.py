"""对齐模块的实测偏移与公共坐标转换必须采用同一个正方向。"""
import cv2
import numpy as np
import pytest

from railside import align
from railside.coords import AlignMap
from test_match import FakeTrip
from test_compare import part_scene, _parent_scene
from railside import compare


@pytest.mark.parametrize('dy', [-20, 20])
def test_measured_vertical_shift_maps_to_observed_pixel(dy):
    rng = np.random.default_rng(8)
    template = rng.integers(40, 220, (4096, 4000), dtype=np.uint8)
    new = cv2.warpAffine(template, np.float32([[1, 0, 0], [0, 1, dy]]),
                         (4000, 4096))
    block = align.dense(FakeTrip(new), FakeTrip(template), [2000], lambda x: x)[0]
    assert block[2] == pytest.approx(dy, abs=1)
    mapping = AlignMap([0, 4000], [0, 4000], [block[2], block[2]])
    x, y = mapping.to_trip(2000, 1000)
    assert y == pytest.approx(1000 + dy, abs=1)
    u, v = mapping.to_template(x, y)
    assert (u, v) == pytest.approx((2000, 1000))


@pytest.mark.parametrize('dy', [-20, 20])
def test_camera_vertical_shift_is_not_a_part_displacement(tmp_path, monkeypatch, dy):
    q, item, new, _ = part_scene(tmp_path, monkeypatch, shift=(0, dy))
    mapping = AlignMap([0, 1800], [0, 1800], [dy, dy])
    assert compare.offset_of(item, q, mapping) == pytest.approx((0, 0), abs=1)   # 位移从这里来
    measured = compare.measure(new, q, item, mapping)
    assert measured['ncc'] > 0.95


@pytest.mark.parametrize('blob', [None, (900, 350, 1000, 450)])
def test_foreign_detection_uses_same_vertical_mapping(tmp_path, monkeypatch, blob):
    template, new, parts, instances, _ = _parent_scene(tmp_path, monkeypatch, blob=blob)
    dy = 20
    shifted = cv2.warpAffine(new.im, np.float32([[1, 0, 0], [0, 1, dy]]), (1800, 1400))
    for item in instances.values():
        item['centroid_xy'][1] += dy
        item['bbox_xy'][1] += dy
        item['bbox_xy'][3] += dy
        if 'poly_xy' in item:
            item['poly_xy'] = [[x, y + dy] for x, y in item['poly_xy']]
    mapping = AlignMap([0, 1800], [0, 1800], [dy, dy])
    found = compare.find_foreign(template, FakeTrip(shifted), {'head_x': 0}, parts, instances, [],
                                 mapping, {'diff': 1.5, 'min_area': 400, 'min_score': 2.0})
    assert len(found) == (0 if blob is None else 1)
    if blob:
        assert found[0]['bbox_xy'] == pytest.approx([900, 370, 1000, 470], abs=4)
