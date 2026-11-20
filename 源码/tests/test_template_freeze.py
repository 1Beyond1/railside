"""已发布模板及其编码表不能被后续标注静默覆盖。"""
from types import SimpleNamespace

import pytest

from railside import io, template, cli
from test_match import proj  # noqa: F401


@pytest.mark.parametrize('frozen', ['template', 'partial'])
def test_existing_version_is_rejected_before_any_write(tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    monkeypatch.setattr(io, 'MID', tmp_path / '数据/中间')
    monkeypatch.setattr(io, 'OUT', tmp_path / '数据/成品')
    p = template.tpl_dir('v1') / ('template.json' if frozen == 'template' else 'anchors/old.png')
    p.parent.mkdir(parents=True)
    p.write_bytes(b'frozen original')
    before = {x.relative_to(tmp_path): x.read_bytes() for x in tmp_path.rglob('*') if x.is_file()}
    # 故意不提供原图和标注：必须先保护已有版本，再进入任何读图/写图步骤。
    with pytest.raises(SystemExit, match='版本.*已存在'):
        template.cmd_build(SimpleNamespace(trip='T01', ver='v1'))
    assert {x.relative_to(tmp_path): x.read_bytes() for x in tmp_path.rglob('*') if x.is_file()} == before


def test_new_version_build_keeps_old_version_and_codes(proj):
    old = template.tpl_dir('v1') / 'template.json'
    frozen = old.read_bytes()
    source = io.read_json(old)['parts']
    ids = {p['code']: i for i, p in enumerate(source, 1)}
    labels = [dict(id=ids[p['code']], type=p['type'], bbox=p['bbox_uv'], poly=p['poly_uv'],
                   parent=ids.get(p.get('parent'))) for p in source]
    io.write_json(template.label_dir('T01') / '零件.json', {'parts': labels})
    io.write_json(template.label_dir('T01') / '定位靶.json', {'anchors': []})
    codes = io.ROOT / '主数据/零件实例_v1.csv'
    codes.write_text('frozen code table')
    cli.main(['template', 'build', '--trip', 'T01', '--ver', 'v2'])
    assert len(template.load_template('v2')['parts']) == 2
    assert (io.ROOT / '主数据/零件实例_v2.csv').is_file()
    assert old.read_bytes() == frozen
    assert codes.read_text() == 'frozen code table'


def _labels_from_v1(proj):
    source = io.read_json(template.tpl_dir('v1') / 'template.json')['parts']
    ids = {p['code']: i for i, p in enumerate(source, 1)}
    labels = [dict(id=ids[p['code']], type=p['type'], bbox=p['bbox_uv'], poly=p['poly_uv'],
                   parent=ids.get(p.get('parent'))) for p in source]
    io.write_json(template.label_dir('T01') / '零件.json', {'parts': labels})
    io.write_json(template.label_dir('T01') / '定位靶.json', {'anchors': []})


def test_frozen_codes_rebuild_must_match(proj):
    """编码表进了 git、模板目录没有（另一台机器复现）：重建出同样的编码就放行，编码表一字不改。"""
    _labels_from_v1(proj)
    cli.main(['template', 'build', '--trip', 'T01', '--ver', 'v2'])
    codes = io.ROOT / '主数据/零件实例_v2.csv'
    frozen = codes.read_bytes()
    template.tpl_dir('v2').rename(template.tpl_dir('v2_first'))
    out = template.cmd_build(SimpleNamespace(trip='T01', ver='v2'))
    assert out['codes'] == 'verified'
    assert codes.read_bytes() == frozen
    assert len(template.load_template('v2')['parts']) == 2


def test_frozen_codes_mismatch_rejected_and_cleaned(proj):
    _labels_from_v1(proj)
    codes = io.ROOT / '主数据/零件实例_v2.csv'
    codes.parent.mkdir(parents=True, exist_ok=True)
    codes.write_text('code,type,parent,car,zone,u,v\nC9-BODY-X-01,X,,9,BODY,0,0\n', encoding='utf-8')
    frozen = codes.read_bytes()
    with pytest.raises(SystemExit, match='不一致'):
        template.cmd_build(SimpleNamespace(trip='T01', ver='v2'))
    assert not template.tpl_dir('v2').exists()
    assert codes.read_bytes() == frozen


def test_default_template_is_latest(proj):
    """没给 --template 时取最新的 v<n>，不再默认作废的 v1。"""
    _labels_from_v1(proj)
    cli.main(['template', 'build', '--trip', 'T01', '--ver', 'v10'])
    assert template.resolve_ver(None) == 'v10'
    assert template.resolve_ver('v1') == 'v1'
    assert template.load_template()['version'] == 'v10'
