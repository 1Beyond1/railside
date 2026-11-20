import re

import pytest

from fixtures.two_trips import make_project
from railside import io, report


@pytest.fixture
def proj(tmp_path, monkeypatch):
    return make_project(tmp_path, monkeypatch)


def img_refs(html):
    return re.findall(r'<img src="([^"]+)"', html)


def test_report_alarms(proj):
    out = proj / '报告T02'
    s = report.write_report('T02', out)
    html = (out / 'index.html').read_text(encoding='utf-8')
    assert s['alarms'] == 3 and s['missing'] == []
    for code in ('C1-B1-BOLT-02', 'C2-BODY-BOX-01', '异物'):
        assert code in html
    assert 'C1-B1-BOLT-01' not in html.split('7 告警')[1]  # 无告警的零件不进告警列表
    assert '未运行' not in html
    refs = img_refs(html)
    assert 'img/preview.jpg' in refs
    assert 'img/alarm001_tpl.jpg' in refs and 'img/alarm001_trip.jpg' in refs
    for r in refs:
        assert not re.match(r'^([a-zA-Z]:|/)', r)  # 相对路径
        assert (out / r).is_file(), r
    # 本趟局部图翻转显示
    crop = io.imread(io.trip_dir('T02') / 'crops' / 'C1-B1-BOLT-02.png')
    shown = io.imread(out / 'img' / 'alarm001_trip.jpg').astype(int)
    assert abs(shown - crop[:, ::-1]).mean() < abs(shown - crop).mean() / 2


def test_report_missing_steps(proj):
    out = proj / '报告T03'
    s = report.write_report('T03', out)
    html = (out / 'index.html').read_text(encoding='utf-8')
    assert html.count('未运行') >= 5
    assert set(s['missing']) == {'qc.json', 'calib.json', 'align.json', 'seg.json', 'parts.json', 'compare.json'}
    # 没有预览图时整车一节也是未运行
    assert 'img/preview.jpg' not in html


def test_report_rerun_clears_old_images(proj):
    out = proj / '报告T02'
    report.write_report('T02', out)
    (out / 'img' / 'stale.jpg').write_bytes(b'x')
    report.write_report('T02', out)
    assert not (out / 'img' / 'stale.jpg').exists()


def test_preview_marks_position():
    man = {'train_x0': 1000, 'train_x1': 17000, 'train_panels': [{'car': 1, 'x0': 1000, 'x1': 17000}]}
    import numpy as np
    pv = np.zeros((512, 2000), np.uint8)
    im = report.preview_with_marks(man, pv, [(1, (16000, 2000, 16100, 2100))])
    # 车头在右：x=16000 靠近车尾（左侧）。翻转列 = (17000//8-1) - 16000//8 = 124 → /4 = 31
    red = np.where((im[:, :, 2] > 200) & (im[:, :, 0] < 50))
    assert red[1].min() >= 20 and red[1].max() <= 40
