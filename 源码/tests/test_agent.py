"""S9 智能体：离线意图路由、工具留痕、假 LLM 的 tool calling 循环（多轮、工具报错）、报告、MCP。"""
import asyncio
import json
import os
import sys

import pytest

from fixtures.two_trips import make_project
from railside import agent, db, io, pipeline, query


@pytest.fixture
def proj(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    db.build(root / '数据' / '成品' / '库.sqlite', ['T01', 'T02', 'T03'], 'v1')
    return root


def logs(trip='T02'):
    p = io.trip_dir(trip) / 'agent_log.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


# ---------- 离线路由 ----------

def test_offline_alarms(proj):
    r = agent.ask('T02', '这趟有哪些告警')
    assert r['mode'] == '离线' and [c['tool'] for c in r['tools']] == ['alarms']
    assert 'T02' in r['reply'] and '3 条告警' in r['reply'] and 'missing' in r['reply']


def test_offline_alarm_type_filter(proj):
    r = agent.ask('T02', '有没有缺失的零件')
    assert [c['tool'] for c in r['tools']] == ['alarms'] and r['tools'][0]['args'] == {'trip': 'T02', 'type': 'missing'}
    assert 'C2-BODY-BOX-01' in r['reply']


def test_offline_part_two_trips(proj):
    r = agent.ask('T02', 'C1-B1-BOLT-02 两趟比怎么样')
    assert [c['tool'] for c in r['tools']] == ['part_history']
    assert r['tools'][0]['args'] == {'code': 'C1-B1-BOLT-02'}  # 两趟：不带 trip 过滤
    assert 'T01' in r['reply'] and 'T02' in r['reply'] and 'NCC' in r['reply'] and 'rotate' in r['reply']


def test_offline_part_one_trip(proj):
    r = agent.ask('T02', 'C1-B1-BOLT-02 这趟怎么样')
    assert r['tools'][0]['args'] == {'code': 'C1-B1-BOLT-02', 'trip': 'T02'}
    assert 'T01' not in r['reply']


def test_offline_step_summaries(proj):
    q = agent.ask('T02', '质检怎么样')
    assert [c['tool'] for c in q['tools']] == ['qc_summary'] and '通过' in q['reply']
    a = agent.ask('T02', '对齐残差多大')
    assert [c['tool'] for c in a['tools']] == ['align_summary'] and 'p95' in a['reply'] and '4.0' in a['reply']
    c = agent.ask('T02', '横向比例和标定')
    assert [c['tool'] for c in c['tools']] == ['calib_summary'] and 'mm/px' in c['reply']


def test_offline_sql(proj):
    r = agent.ask('T02', 'select count(*) from trip')
    assert [c['tool'] for c in r['tools']] == ['sql'] and '3' in r['reply']


def test_offline_unknown_is_overview(proj):
    r = agent.ask('T02', '你好')
    assert [c['tool'] for c in r['tools']] == ['qc_summary', 'align_summary', 'alarms']
    assert all(x in r['reply'] for x in ('T02', '对齐', '告警'))


def test_unknown_trip_is_error(proj):
    r = agent.ask('T99', '这趟有哪些告警')
    assert r['tools'][0]['status'] == 'error' and 'T99' in r['reply']
    assert '没有趟次' in r['reply']


def test_missing_file_degrades(proj):
    # T03 只有 manifest：各摘要给“未运行”，不崩
    r = agent.ask('T03', '质检怎么样')
    assert '还没有质检结果' in r['reply']
    r = agent.ask('T03', '对齐呢')
    assert '还没有对齐结果' in r['reply']


# ---------- 留痕 ----------

def test_every_call_is_logged(proj):
    before = len(logs())
    agent.ask('T02', '这趟有哪些告警')
    rec = logs()[-1]
    assert rec['tool'] == 'alarms' and rec['status'] == 'ok' and rec['step'] == 'agent'
    assert rec['result']['n'] == 3 and rec['trip'] == 'T02'
    assert len(logs()) == before + 1
    assert query.trip_info('T02')['n_agent_log'] == before + 1  # 库也同步写了


def test_run_step_tool_logged(proj, monkeypatch):
    seen = []

    def _spawn(argv):
        seen.append(list(argv))
        return 0, json.dumps({'step': argv[0]}, ensure_ascii=False) + '\n', ''

    monkeypatch.setattr(pipeline, '_spawn', _spawn)
    r = agent.ask('T02', '跑一下对齐')
    assert [c['tool'] for c in r['tools']] == ['run_step'] and r['tools'][0]['args']['step'] == 'align'
    assert seen == [['align', 'T02', '--template', 'v1']]  # 子进程参数由 pipeline.step_argv 拼
    assert logs()[-1]['tool'] == 'run_step' and logs()[-1]['status'] == 'ok'


# ---------- 假 LLM ----------

def fake_llm(monkeypatch, replies):
    sent = []

    def _post(url, payload, key, timeout=60):
        sent.append(payload)
        return {'choices': [{'message': replies[len(sent) - 1]}]}

    monkeypatch.setattr(agent, '_post', _post)
    monkeypatch.setenv('RAILSIDE_LLM_BASE_URL', 'http://假模型/v1')
    monkeypatch.setenv('RAILSIDE_LLM_MODEL', 'fake-chat')
    monkeypatch.setenv('RAILSIDE_LLM_API_KEY', 'test-key')
    return sent


def test_llm_tool_calling_round(proj, monkeypatch):
    sent = fake_llm(monkeypatch, [
        {'content': None, 'tool_calls': [{'id': 'c1', 'type': 'function',
                                          'function': {'name': 'alarms', 'arguments': '{"trip": "T02"}'}}]},
        {'content': 'T02 有 3 条告警，其中 1 个零件缺失。', 'tool_calls': []},
    ])
    r = agent.ask('T02', '这趟有哪些告警')
    assert r['mode'] == '在线' and r['reply'] == 'T02 有 3 条告警，其中 1 个零件缺失。'
    assert [c['tool'] for c in r['tools']] == ['alarms']
    assert sent[0]['model'] == 'fake-chat' and sent[0]['tools'][0]['type'] == 'function'
    second = sent[1]['messages']
    assert [m['role'] for m in second] == ['system', 'user', 'assistant', 'tool']
    assert second[-1]['tool_call_id'] == 'c1' and 'C2-BODY-BOX-01' in second[-1]['content']


def test_llm_multi_round(proj, monkeypatch):
    sent = fake_llm(monkeypatch, [
        {'tool_calls': [{'id': 'a', 'function': {'name': 'qc_summary', 'arguments': '{}'}}]},
        {'tool_calls': [{'id': 'b', 'function': {'name': 'part_history', 'arguments': '{"code": "C1-B1-BOLT-02"}'}}]},
        {'content': '质检通过，螺栓在两趟都找到了。', 'tool_calls': []},
    ])
    r = agent.ask('T02', '帮我看看')
    assert [c['tool'] for c in r['tools']] == ['qc_summary', 'part_history']
    assert r['reply'].startswith('质检通过') and len(sent) == 3


def test_llm_tool_error_goes_back_to_model(proj, monkeypatch):
    sent = fake_llm(monkeypatch, [
        {'tool_calls': [{'id': 'x', 'function': {'name': 'run_step', 'arguments': '{"step": "nope"}'}}]},
        {'content': '没有 nope 这一步。', 'tool_calls': []},
    ])
    r = agent.ask('T02', '跑一下 nope')
    assert r['tools'][0]['status'] == 'error'
    payload = json.loads(sent[1]['messages'][-1]['content'])
    assert 'ValueError' in payload['error'] and '不认识的步骤' in payload['error']
    assert logs()[-1]['status'] == 'error' and logs()[-1]['tool'] == 'run_step'
    assert r['reply'] == '没有 nope 这一步。'


def test_llm_bad_json_arguments(proj, monkeypatch):
    fake_llm(monkeypatch, [
        {'tool_calls': [{'id': 'x', 'function': {'name': 'alarms', 'arguments': '{不是 JSON'}}]},
        {'content': '好的。', 'tool_calls': []},
    ])
    r = agent.ask('T02', '看告警')
    assert r['tools'][0]['tool'] == 'alarms' and r['tools'][0]['status'] == 'ok'  # 参数空着也答得出来


def test_llm_failure_falls_back_offline(proj, monkeypatch):
    def _post(url, payload, key, timeout=60):
        raise OSError('连不上')

    monkeypatch.setattr(agent, '_post', _post)
    monkeypatch.setenv('RAILSIDE_LLM_BASE_URL', 'http://假模型/v1')
    monkeypatch.setenv('RAILSIDE_LLM_MODEL', 'fake-chat')
    r = agent.ask('T02', '这趟有哪些告警')
    assert '大模型调用失败' in r['reply'] and '3 条告警' in r['reply']


def test_llm_round_limit(proj, monkeypatch):
    loop = {'tool_calls': [{'id': 'x', 'function': {'name': 'qc_summary', 'arguments': '{}'}}]}
    fake_llm(monkeypatch, [loop] * 8)
    r = agent.ask('T02', '一直查')
    assert '轮次用完' in r['reply']


# ---------- 报告 ----------

def test_agent_report(proj):
    rep = agent.write_agent_report('T02')
    assert rep['mode'] == '离线'
    assert rep['path'] == io.OUT / '报告' / 'T02' / '智能体报告.md'
    text = rep['path'].read_text(encoding='utf-8')
    for h in ('# T02', '## 1 质检', '## 2 对齐质量', '## 3 标定', '## 4 告警', '## 5 需人工复核', '## 6 局限'):
        assert h in text
    assert 'C2-BODY-BOX-01' in text and '| # | 编码 |' in text
    assert [c['tool'] for c in rep['tools']] == ['qc_summary', 'align_summary', 'calib_summary', 'alarms']
    assert logs()[-1]['tool'] == 'alarms'


def test_agent_report_cli(proj, capsys):
    from railside import cli
    cli.main(['agent', '--trip', 'T02', '--report'])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith('{')][-1]
    s = json.loads(line)
    assert '智能体报告' in s['reply'] and s['report'].endswith('智能体报告.md')


def test_agent_cli_message(proj, capsys):
    from railside import cli
    cli.main(['agent', '--trip', 'T02', '--message', '这趟有哪些告警'])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith('{')][-1]
    s = json.loads(line)
    assert '3 条告警' in s['reply'] and s['tools'][0]['tool'] == 'alarms' and s['mode'] == '离线'


def test_agent_needs_message_or_report(proj):
    from railside import cli
    with pytest.raises(SystemExit) as e:
        cli.main(['agent', '--trip', 'T02'])
    assert '--message' in str(e.value)


# ---------- MCP ----------

def test_mcp_stdio_lists_tools_and_queries(proj):
    pytest.importorskip('mcp')
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def main():
        params = StdioServerParameters(
            command=sys.executable, args=['-m', 'railside.cli', 'mcp'],
            env={**os.environ, 'RAILSIDE_ROOT': str(io.ROOT), 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = await s.list_tools()
                names = {t.name for t in tools.tools}
                assert {'qc_summary', 'align_summary', 'calib_summary', 'alarms', 'part_history', 'sql',
                        'run_step', 'run_trip', 'report'} <= names
                res = await s.call_tool('alarms', {'trip': 'T02'})
                data = json.loads(res.content[0].text)
                assert data['n'] == 3 and data['source'] == '库（railside.query）'
                bad = await s.call_tool('part_history', {'code': 'C9-B1-NOPE-99'})
                assert json.loads(bad.content[0].text)['trips'] == []

    asyncio.run(asyncio.wait_for(main(), 90))