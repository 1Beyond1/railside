"""S9 一键流程：步骤顺序、缺模块跳过、质检失败停下、--template 不认时重试、每步留痕。"""
import json

import pytest

from fixtures.two_trips import make_project
from railside import cli, io, pipeline

def ALL(step):
    """所有步骤模块都当存在（子进程另用 monkeypatch 假造）。"""
    return True, ''


@pytest.fixture
def proj(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    (root / 'src').mkdir()
    return root


def summaries(calls, extra=None):
    """按步骤名返回一行 JSON 摘要的假子进程。"""
    extra = extra or {}

    def _spawn(argv):
        calls.append(list(argv))
        # 模拟本次运行生成的新产物，不能依靠运行前留下的旧文件。
        if argv[0] == 'stitch':
            io.write_json(io.trip_dir('T09') / 'manifest.json',
                          io.read_json(io.trip_dir('T01') / 'manifest.json'))
        elif argv[0] == 'align':
            ver = argv[argv.index('--template') + 1] if '--template' in argv else 'v1'
            io.write_json(io.trip_dir('T09') / 'align.json', {'template_ver': ver})
        elif argv[0] == 'match':
            io.write_json(io.trip_dir('T09') / 'parts.json', {'instances': []})
        return 0, json.dumps(extra.get(argv[0], {'step': argv[0]}), ensure_ascii=False) + '\n', ''
    return _spawn


def test_all_steps_in_order(proj, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, 'prereq', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, '_spawn', summaries(calls, {'ingest': {'step': 'ingest', 'verdict': 'pass'},
                                                              'report': {'step': 'report', 'out': '数据/成品/报告/T09/index.html'}}))
    out = pipeline.run_trip(proj / 'src', 'T09')
    # segment 没给权重，跳过；其余都跑
    assert [c[0] for c in calls] == ['ingest', 'stitch', 'calib', 'align', 'match', 'compare', 'db', 'report']
    assert [s['status'] for s in out['steps']] == ['ok', 'ok', 'ok', 'ok', 'skipped', 'ok', 'ok', 'ok', 'ok']
    assert out['status'] == 'ok' and out['qc_verdict'] == 'pass'
    assert out['report'] == '数据/成品/报告/T09/index.html'
    assert calls[0] == ['ingest', str(proj / 'src'), '--trip', 'T09']
    # 入库和报告跟本次流程的模板走（没指定时就是最新的 v1），不再由子步骤自己另取
    assert calls[-2] == ['db', 'build', '--template', 'v1'] and calls[-1] == ['report', 'T09', '--template', 'v1']
    assert out['elapsed_s'] >= 0 and all(s['seconds'] >= 0 for s in out['steps'])


def test_every_step_logged(proj, monkeypatch):
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, 'prereq', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, '_spawn', summaries([]))
    pipeline.run_trip(proj / 'src', 'T09')
    lines = [json.loads(x) for x in (io.trip_dir('T09') / 'agent_log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert [r['step'] for r in lines] == ['ingest', 'stitch', 'calib', 'align', 'segment', 'match', 'compare', 'db',
                                          'report', 'run']
    assert lines[4]['status'] == 'skipped' and 'weights' in lines[4]['result']['reason']
    assert lines[-1]['tool'] == 'pipeline.run' and lines[-1]['status'] == 'ok'
    assert lines[-1]['result']['steps'][0][:2] == ['ingest', 'ok']


def test_missing_module_is_skipped(proj, monkeypatch):
    calls = []

    def available(step):
        ok = step in ('ingest', 'stitch', 'db', 'report')
        return ok, '' if ok else f'模块 railside.{step} 还没实现'

    monkeypatch.setattr(pipeline, 'available', available)
    io.write_json(io.trip_dir('T09') / 'manifest.json', io.read_json(io.trip_dir('T01') / 'manifest.json'))
    monkeypatch.setattr(pipeline, '_spawn', summaries(calls, {'ingest': {'step': 'ingest', 'verdict': 'pass'}}))
    out = pipeline.run_trip(proj / 'src', 'T09')
    assert out['status'] == 'partial'  # 缺核心模块时仍出报告，但不算完整检测成功
    skipped = {s['step']: s for s in out['steps'] if s['status'] == 'skipped'}
    assert set(skipped) == {'calib', 'align', 'segment', 'match', 'compare'}
    assert '还没实现' in skipped['align']['reason']
    assert 'report' not in skipped and calls[-1] == ['report', 'T09', '--template', 'v1']


def test_qc_fail_stops_and_reports(proj, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'available', ALL)

    def _spawn(argv):
        calls.append(list(argv))
        return 0, json.dumps({'step': 'ingest', 'verdict': 'fail', 'reasons': ['缺帧 [5]']}, ensure_ascii=False) + '\n', ''

    monkeypatch.setattr(pipeline, '_spawn', _spawn)
    out = pipeline.run_trip(proj / 'src', 'T09')
    assert [c[0] for c in calls] == ['ingest']
    assert out['status'] == 'failed' and '缺帧 [5]' in out['reason']
    assert out['steps'][-1]['step'] == 'ingest' and out['steps'][-1]['status'] == 'failed'
    lines = (io.trip_dir('T09') / 'agent_log.jsonl').read_text(encoding='utf-8').splitlines()
    assert json.loads(lines[0])['status'] == 'failed'


def test_failed_step_stops(proj, monkeypatch):
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, 'prereq', lambda *a, **k: None)

    def _spawn(argv):
        if argv[0] == 'align':
            return 1, '', 'Traceback...\nRuntimeError: 定位靶只找到 2 个，无法对齐\n'
        return 0, json.dumps({'step': argv[0], 'verdict': 'pass'}) + '\n', ''

    monkeypatch.setattr(pipeline, '_spawn', _spawn)
    out = pipeline.run_trip(proj / 'src', 'T09')
    assert out['status'] == 'failed' and '无法对齐' in out['reason']
    assert out['steps'][-1]['step'] == 'align' and 'db' not in [s['step'] for s in out['steps']]


def test_template_arg_retried_without(proj, monkeypatch):
    assert pipeline.step_argv('calib', 'T09') == ['T09']  # calib 不带 --template
    seen = []
    monkeypatch.setattr(pipeline, 'available', ALL)

    def _spawn(argv):
        seen.append(list(argv))
        if '--template' in argv:
            return 2, '', 'railside align: error: unrecognized arguments: --template v1\n'
        return 0, json.dumps({'step': argv[0]}) + '\n', ''

    monkeypatch.setattr(pipeline, '_spawn', _spawn)
    r = pipeline.run_step('align', pipeline.step_argv('align', 'T09'))
    assert r['status'] == 'ok' and r['summary']['step'] == 'align'
    assert seen == [['align', 'T09', '--template', 'v1'], ['align', 'T09']]
    assert r['argv'] == ['T09']


def test_step_argv_omits_template_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'OUT', tmp_path / 'empty_out')
    for step in ('align', 'match', 'compare', 'report'):
        argv = pipeline.step_argv(step, 'T09')
        assert '--template' not in argv and 'None' not in argv
    assert pipeline.step_argv('db', 'T09') == ['build']


def test_unrecognized_without_template_is_failed(proj, monkeypatch):
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, '_spawn', lambda argv: (2, '', 'railside align: error: multiple values for trip\n'))
    r = pipeline.run_step('align', ['T09'])
    assert r['status'] == 'failed' and '退出码 2' in r['reason']


def test_no_summary_is_noted(proj, monkeypatch):
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, '_spawn', lambda argv: (0, '只有普通输出\n', ''))
    r = pipeline.run_step('stitch', ['T09'])
    assert r['status'] == 'ok' and r['summary'] is None and 'JSON' in r['reason']


def test_cli_run_emits_one_summary(proj, monkeypatch, capsys):
    src = proj / 'src'
    src.mkdir(exist_ok=True)
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, 'prereq', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, '_spawn', summaries([], {'ingest': {'step': 'ingest', 'verdict': 'pass'},
                                                           'report': {'step': 'report', 'out': 'x/index.html'}}))
    cli.main(['run', str(src), '--trip', 'T09'])
    text = capsys.readouterr().out
    out = [x for x in text.splitlines() if x.startswith('{')]
    assert len(out) == 1
    s = json.loads(out[-1])
    assert s['step'] == 'run' and s['trip'] == 'T09' and s['status'] == 'ok'
    assert '[1/9] ingest' in text and '[9/9] report' in text


def test_cli_run_exits_nonzero_on_qc_fail(proj, monkeypatch, capsys):
    src = proj / 'src'
    src.mkdir(exist_ok=True)
    monkeypatch.setattr(pipeline, 'available', ALL)
    monkeypatch.setattr(pipeline, '_spawn', summaries([], {'ingest': {'step': 'ingest', 'verdict': 'fail', 'reasons': ['没找到列车']}}))
    with pytest.raises(SystemExit) as e:
        cli.main(['run', str(src), '--trip', 'T09'])
    assert e.value.code == 1
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith('{')]
    assert json.loads(lines[-1])['status'] == 'failed'


def test_prereq_missing_inputs(proj):
    """缺输入先看得到、写清楚缺什么，不让子进程报一串错。"""
    assert '缺 align.json' in pipeline.prereq('match', 'T09')
    assert '缺 parts.json' in pipeline.prereq('compare', 'T09')
    assert 'manifest' in pipeline.prereq('report', 'T09')
    assert pipeline.prereq('stitch', 'T09') is None
    assert '模板 v9 不存在' in pipeline.prereq('align', 'T09', 'v9')


def test_match_skipped_when_template_has_no_parts(proj):
    io.write_json(io.trip_dir('T09') / 'align.json', {'template_ver': 'v1'})
    tp = io.OUT / '模板' / 'v1' / 'template.json'
    t = io.read_json(tp)
    t['parts'] = []
    io.write_json(tp, t)
    why = pipeline.prereq('match', 'T09')
    assert '没有零件' in why and '跳过' in why


def test_prereq_skip_still_reaches_report(proj, monkeypatch):
    """模板没零件：match、compare 记跳过，后面入库、报告照跑到。"""
    monkeypatch.setattr(pipeline, 'available', ALL)
    calls = []
    d = io.trip_dir('T09')
    io.write_json(d / 'manifest.json', io.read_json(io.trip_dir('T01') / 'manifest.json'))
    io.write_json(d / 'align.json', {'template_ver': 'v1'})
    tp = io.OUT / '模板' / 'v1' / 'template.json'
    t = io.read_json(tp)
    t['parts'] = []
    io.write_json(tp, t)
    monkeypatch.setattr(pipeline, '_spawn', summaries(calls, {'ingest': {'step': 'ingest', 'verdict': 'pass'},
                                                              'report': {'step': 'report', 'out': 'x/index.html'}}))
    out = pipeline.run_trip(proj / 'src', 'T09')
    st = {s['step']: s for s in out['steps']}
    assert out['status'] == 'partial'
    assert st['match']['status'] == 'skipped' and '没有零件' in st['match']['reason']
    assert st['compare']['status'] == 'skipped' and 'parts.json' in st['compare']['reason']
    assert st['db']['status'] == 'ok' and st['report']['status'] == 'ok'
    assert out['report'] == 'x/index.html'
    assert 'match' not in [c[0] for c in calls] and 'compare' not in [c[0] for c in calls]


def test_cli_run_missing_src(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(['run', str(tmp_path / '没有这个目录'), '--trip', 'T09'])
    assert '原图目录不存在' in str(e.value)
