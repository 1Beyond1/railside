"""端到端脱敏演示命令 `railside demo` 的回归测试。"""
import json

from railside import cli, io
from railside.app import data


def test_demo_end_to_end(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'demo_data'
    cli.main(['demo', '--out', str(out)])
    res = json.loads(capsys.readouterr().out)
    assert res['status'] == 'created' and res['synthetic'] is True
    assert res['trips'] == ['T01', 'T02'] and res['cars'] == 3
    assert {'missing', 'shift', 'dtheta', 'foreign'} <= set(res['alarm_types'])
    assert (out / '成品' / '库.sqlite').is_file()
    assert (out / '成品' / '报告' / 'T02' / 'index.html').is_file()

    # 应用层接口（含双屏标准模板对齐映射与异物对比图）
    monkeypatch.setattr(io, 'DATA', out)
    monkeypatch.setattr(io, 'MID', out / '中间')
    monkeypatch.setattr(io, 'OUT', out / '成品')
    ov = data.overlay('T02')
    al = data.alarms('T02')
    assert ov['compare_ref']['template_trip'] == 'T01'
    assert len(ov['compare_ref']['map']) >= 2
    fo = next(r for r in al['rows'] if r['kind'] == 'foreign')
    assert data.foreign_crop('T02', fo['idx'], 'template')[:2] == b'\xff\xd8'
    assert data.foreign_crop('T02', fo['idx'], 'trip')[:2] == b'\xff\xd8'
