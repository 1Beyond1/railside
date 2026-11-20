import json
import sqlite3

import pytest

from fixtures.two_trips import make_project
from railside import cli, db, io, query


def dump(path):
    con = sqlite3.connect(path)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'meta' ORDER BY name")]
        return {t: con.execute(f'SELECT * FROM {t} ORDER BY rowid').fetchall() for t in tables}
    finally:
        con.close()


@pytest.fixture
def proj(tmp_path, monkeypatch):
    return make_project(tmp_path, monkeypatch)


@pytest.fixture
def built(proj):
    path = proj / '库.sqlite'
    db.build(path, ['T01', 'T02'], 'v1')
    return path


def test_build_idempotent_and_all_tables(proj):
    path = proj / '库.sqlite'
    s1 = db.build(path, ['T01', 'T02'], 'v1')
    d1 = dump(path)
    s2 = db.build(path, ['T01', 'T02'], 'v1')
    assert dump(path) == d1
    assert s1['rows'] == s2['rows']
    for t in ('trip', 'panel', 'anchor', 'part_type', 'part_instance', 'part_feature', 'comparison',
              'template_version', 'agent_log', 'alarm', 'source_file'):
        assert s1['rows'][t] > 0, t
    # 结果表的来龙去脉列
    con = sqlite3.connect(path)
    row = con.execute("SELECT src, code_ver, created_at, template_ver FROM part_instance WHERE kind='trip' AND trip_id='T02' "
                      "AND code='C1-B1-BOLT-02'").fetchone()
    con.close()
    assert row[0] == 'frames:2-2' and row[1] and row[2] and row[3] == 'v1'


def test_default_trips_and_template(proj):
    s = db.build(proj / '库.sqlite')
    assert s['trips'] == ['T01', 'T02', 'T03'] and s['template_ver'] == 'v1'


def test_part_history_across_trips(built):
    h = query.part_history('C1-B1-BOLT-02', db_path=built)
    assert [t['trip_id'] for t in h['trips']] == ['T01', 'T02']
    assert h['template'][0]['template_ver'] == 'v1' and h['template'][0]['features']['size_mm[0]'] == 12.0
    assert h['part_type'][0]['name'] == '螺栓'
    t02 = h['trips'][1]
    assert t02['instances'][0]['features']['angle_deg'] == 12.0
    assert [a['type'] for a in t02['alarms']] == ['rotate']
    assert h['trips'][0]['alarms'] == [] and len(h['trips'][0]['comparisons']) == 1
    only = query.part_history('C1-B1-BOLT-02', trip='T01', db_path=built)
    assert [t['trip_id'] for t in only['trips']] == ['T01']


def test_alarm_filters(built):
    assert len(query.alarms(db_path=built)) == 4
    t02 = query.alarms(trip='T02', db_path=built)
    assert sorted(a['type'] for a in t02) == ['foreign', 'missing', 'rotate']
    miss = query.alarms(trip='T02', type='missing', db_path=built)
    assert len(miss) == 1 and miss[0]['code'] == 'C2-BODY-BOX-01' and miss[0]['value'] == 1
    assert [a['code'] for a in query.alarms(trip='T01', db_path=built)] == ['C1-B1-BOLT-01']
    assert query.alarms(type='nope', db_path=built) == []


def test_trip_info(built):
    t = query.trip_info('T02', db_path=built)
    assert t['info']['qc_verdict'] == 'pass' and t['info']['align_residual'] == {'median': 1.5, 'p95': 4.0}
    assert len(t['panels']) == 2 and t['panels'][1]['src'] == 'frames:5-9'
    assert t['parts_by_found_by'] == {'missing': 1, 'pixel': 2}
    assert t['n_agent_log'] == 1
    assert query.trip_info('T09', db_path=built) == {'trip': 'T09', 'found': False}


@pytest.mark.parametrize('q', [
    'DELETE FROM trip',
    'DROP TABLE trip',
    'WITH x AS (SELECT 1) DELETE FROM trip',
    'SELECT 1; DELETE FROM trip',
    'PRAGMA query_only = OFF',
])
def test_sql_rejects_writes(built, q):
    with pytest.raises(ValueError):
        query.sql(q, db_path=built)
    assert query.sql('SELECT count(*) n FROM trip', db_path=built) == {'columns': ['n'], 'rows': [[2]]}


def test_sql_select(built):
    r = query.sql("WITH a AS (SELECT * FROM alarm WHERE trip_id='T02') SELECT type FROM a ORDER BY type", db_path=built)
    assert r['rows'] == [['foreign'], ['missing'], ['rotate']]


def test_missing_files_do_not_crash(proj):
    # T03 只有 trip/manifest；再删掉主数据和模板
    (proj / '主数据' / '零件类型.csv').unlink()
    s = db.build(proj / '库.sqlite', ['T03'], 'v9')
    assert s['rows']['panel'] == 2 and s['rows']['part_type'] == 0 and s['rows']['template_version'] == 0
    for f in ('主数据/零件类型.csv', '数据/成品/模板/v9/template.json', '数据/中间/T03/qc.json',
              '数据/中间/T03/compare.json', '数据/中间/T03/agent_log.jsonl'):
        assert f in s['skipped'], f
    t = query.trip_info('T03', db_path=proj / '库.sqlite')
    assert t['found'] and t['info']['qc_verdict'] is None and t['alarms_by_type'] == {}
    # 完全没有的趟也不崩
    s = db.build(proj / '库2.sqlite', ['T77'])
    assert s['rows']['trip'] == 1 and '数据/中间/T77/manifest.json' in s['skipped']


def test_query_without_db(proj):
    with pytest.raises(FileNotFoundError):
        query.alarms(db_path=proj / '没有.sqlite')


def test_append_agent_log_and_rebuild(built):
    rec = db.append_agent_log('T01', 'align', 'railside align', {'trip': 'T01'}, {'residual': 2}, 'ok', db_path=built)
    lines = (io.trip_dir('T01') / 'agent_log.jsonl').read_text(encoding='utf-8').splitlines()
    assert json.loads(lines[-1])['ts'] == rec['ts']
    assert query.sql("SELECT tool, status FROM agent_log WHERE trip_id='T01'", db_path=built)['rows'] == [['railside align', 'ok']]
    db.build(built, ['T01', 'T02'], 'v1')  # 库由 jsonl 重建后仍在
    assert query.trip_info('T01', db_path=built)['n_agent_log'] == 1
    # 库不存在时只写 jsonl
    db.append_agent_log('T02', 'report', 'railside report', {}, None, 'ok', db_path=built.with_name('无.sqlite'))
    assert not built.with_name('无.sqlite').exists()


def test_cli(proj, capsys):
    cli.main(['db', 'build', '--trips', 'T01', 'T02', '--db', '库.sqlite'])
    s = json.loads(capsys.readouterr().out)
    assert s['step'] == 'db' and s['rows']['alarm'] == 4
    cli.main(['query', '--db', '库.sqlite', 'alarms', '--trip', 'T02', '--type', 'rotate'])
    out = json.loads(capsys.readouterr().out)
    assert out['n'] == 1 and out['alarms'][0]['code'] == 'C1-B1-BOLT-02'
    cli.main(['query', '--db', '库.sqlite', 'part', 'C2-BODY-BOX-01'])
    assert len(json.loads(capsys.readouterr().out)['trips']) == 2
    with pytest.raises(SystemExit):
        cli.main(['query', '--db', '库.sqlite', 'sql', 'DELETE FROM trip'])
