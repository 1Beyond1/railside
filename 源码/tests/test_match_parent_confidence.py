"""找不到父件时，不能让其随机匹配峰把完好的子件带偏。"""
from railside import io, cli
from test_match import proj, photo  # noqa: F401


def test_unmatched_parent_does_not_displace_visible_child(proj):
    # 只替换父件的图像；螺栓在另一帧中，像素保持原样。
    frame = io.ROOT / '数据/原始/T02/camImg/1.jpg'
    image = io.imread(frame)
    image[1808:2308, 1237:1687] = photo(500, 450, seed=2)
    io.imwrite(frame, image)
    cli.main(['match', 'T02'])
    parts = {p['type']: p for p in io.read_json(io.trip_dir('T02') / 'parts.json')['instances']}
    assert parts['BOGIE']['found_by'] == 'missing'
    assert parts['BOLT']['found_by'] != 'missing'
    assert parts['BOLT']['match_score'] > 0.8
