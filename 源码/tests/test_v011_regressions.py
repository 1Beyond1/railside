"""v0.1.1 补丁版回归测试：MCP stdio/路径防线、算法边界、评测误报、并发锁与前端静态脱敏检查。"""
import asyncio
import json
import os
import sqlite3
import sys
import threading
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fixtures.two_trips import make_project
from railside import agent, align, calib, cli, compare, db, io, match, pipeline, query, report
from railside.app import data, dzi, jobs
from railside.app.server import create_app
from railside.coords import AlignMap, H, W
from railside.eval import score
from railside.frames import Trip
from test_compare import _parent_scene
from test_match import FakeTrip, photo, proj  # noqa: F401


# ---------- P0-1: MCP stdio 不污染 stdout + 新趟次启动 + 报告不传 tools ----------

def test_agent_tools_run_trip_silent_stdout_and_allows_new_trip(tmp_path, monkeypatch, capsys):
    root = make_project(tmp_path, monkeypatch)
    src = root / 'new_cam'
    src.mkdir()
    seen = []

    def fake_spawn(argv):
        seen.append(list(argv))
        if argv[0] == 'stitch':
            io.write_json(io.trip_dir('T88') / 'manifest.json', io.read_json(io.trip_dir('T01') / 'manifest.json'))
        return 0, json.dumps({'step': argv[0], 'verdict': 'pass'}, ensure_ascii=False) + '\n', ''

    monkeypatch.setattr(pipeline, '_spawn', fake_spawn)
    monkeypatch.setattr(pipeline, 'available', lambda s: (True, ''))
    monkeypatch.setattr(pipeline, 'prereq', lambda *a, **k: None)

    tools = agent.Tools()
    # 新趟次目录尚不存在时，run_step(ingest) 与 run_trip 均允许启动
    r_ing = tools.call('run_step', {'step': 'ingest', 'trip': 'T87', 'src': str(src)})
    assert r_ing['status'] == 'ok'
    r_run = tools.call('run_trip', {'trip': 'T88', 'src': str(src)})
    assert r_run['status'] == 'ok' and r_run['trip'] == 'T88'
    captured = capsys.readouterr()
    assert captured.out == ''  # 绝不向 stdout 打印进度行


def test_mcp_stdio_run_trip_does_not_corrupt_protocol(tmp_path, monkeypatch):
    pytest.importorskip('mcp')
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    root = make_project(tmp_path, monkeypatch)
    db.build(root / '数据' / '成品' / '库.sqlite', ['T01', 'T02'], 'v1')
    empty_src = root / 'empty_cam'
    empty_src.mkdir()

    async def main():
        params = StdioServerParameters(
            command=sys.executable, args=['-m', 'railside.cli', 'mcp'],
            env={**os.environ, 'RAILSIDE_ROOT': str(root), 'RAILSIDE_DATA': str(root / '数据'),
                 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                # 在真实 MCP stdio 子进程中调用 run_trip（新趟次，空目录在 ingest 失败，但 stdout 必须保持协议纯净）
                res = await s.call_tool('run_trip', {'trip': 'T89', 'src': str(empty_src)})
                payload = json.loads(res.content[0].text)
                assert payload['trip'] == 'T89' and payload['status'] == 'failed'
                # 后续调用必须仍能正常通信（若 stdout 被 print 污染，协议已断）
                res2 = await s.call_tool('alarms', {'trip': 'T02'})
                assert json.loads(res2.content[0].text)['n'] == 3

    asyncio.run(asyncio.wait_for(main(), 90))


def test_write_agent_report_omits_tools_param(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch)
    sent = []

    def fake_post(url, payload, key, timeout=60):
        sent.append(payload)
        return {'choices': [{'message': {'content': '# T02 在线报告\n内容正常'}}]}

    monkeypatch.setattr(agent, '_post', fake_post)
    monkeypatch.setenv('RAILSIDE_LLM_BASE_URL', 'http://fake/v1')
    monkeypatch.setenv('RAILSIDE_LLM_MODEL', 'fake-model')
    rep = agent.write_agent_report('T02')
    assert rep['mode'] == '在线'
    assert 'tools' not in sent[0]
    assert '在线报告' in rep['path'].read_text(encoding='utf-8')


# ---------- P0-2: io.trip_dir / MCP / report 统一路径遍历防线 ----------

@pytest.mark.parametrize('bad', ['', '../evil', '..\\evil', 'T01/../T02', 'T 01', 'a' * 33, '.', '..'])
def test_trip_dir_and_report_reject_traversal(tmp_path, monkeypatch, bad):
    make_project(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        io.trip_dir(bad)
    with pytest.raises(ValueError):
        report.write_report(bad)
    with pytest.raises(ValueError):
        agent.write_agent_report(bad)


def test_agent_tools_logs_invalid_trip_to_safe_bucket(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch)
    tools = agent.Tools()
    res = tools.call('alarms', {'trip': '../../pwned'})
    assert 'error' in res
    assert not (tmp_path / 'pwned').exists()
    assert not (io.DATA / 'pwned').exists()
    safe_log = io.trip_dir('_invalid') / 'agent_log.jsonl'
    assert safe_log.is_file()
    rec = json.loads(safe_log.read_text(encoding='utf-8').splitlines()[-1])
    assert rec['trip'] == '_invalid' and rec['status'] == 'error'


# ---------- P1-5 & P1-6 & P1-10: compare 线性面积、bbox_xy、缺失子件遮除与 score 误报 ----------

def test_compare_seg_diff_area_linear_s_amap_and_bbox_xy(proj, monkeypatch):
    cli.main(['match', 'T02'])
    pj = io.read_json(io.trip_dir('T02') / 'parts.json')
    tpl = io.read_json(io.OUT / '模板' / 'v1' / 'template.json')
    tmap = {q['code']: q for q in tpl['parts']}
    # 将螺栓改为 seg 找到，且设局部横向比例 s_amap = 2.0，面积正好 2 倍（即真实无面积变化）
    for it in pj['instances']:
        if it['code'] == 'C1-B1-BOLT-01':
            it['found_by'] = 'seg'
            it['area_px'] = int(round(tmap['C1-B1-BOLT-01']['area_px'] * 2.0))
    io.write_json(io.trip_dir('T02') / 'parts.json', pj)
    monkeypatch.setattr(compare, 'measure', lambda *a, **k: {
        'ncc': 0.96, 'ncc0': 0.96, 'dtheta': 0.0, 's': 2.0, 's_amap': 2.0, 'w': 1.0,
    })
    cli.main(['compare', 'T02'])
    cj = io.read_json(io.trip_dir('T02') / 'compare.json')
    bolt = next(i for i in cj['items'] if i['code'] == 'C1-B1-BOLT-01')
    assert abs(bolt['diff_area']) < 0.01  # 线性换算应为 0%，旧 s_amap**2 会误算成 -50%
    assert bolt['bbox_xy'] and len(bolt['bbox_xy']) == 4
    assert all(i.get('bbox_xy') for i in cj['items'])


def test_find_foreign_masks_missing_child(tmp_path, monkeypatch):
    """缺失子件已被单独报 missing，不应在同一位置再误报一条异物（一缺双报）。"""
    t_tpl, t_new, tparts, inst, amap = _parent_scene(tmp_path, monkeypatch, blob=(520, 520, 600, 600))
    inst['BOLT-1']['found_by'] = 'missing'
    found = compare.find_foreign(t_tpl, t_new, {'head_x': 0}, tparts, inst, [], amap,
                                 {'diff': 1.5, 'min_area': 200, 'min_score': 2.0})
    assert found == []


def test_score_compare_counts_part_fp_via_parts_json_fallback():
    truth = {'trip': 'S', 'defects': [
        {'id': 'D1', 'kind': 'rotate', 'code': 'A', 'bbox_xy': [0, 0, 50, 50], 'part_bbox_xy': [0, 0, 50, 50], 'params': {}},
        {'id': 'D2', 'kind': 'foreign', 'code': 'G', 'bbox_xy': [3000, 100, 3080, 180], 'part_bbox_xy': [2800, 0, 3500, 400], 'params': {}},
    ]}
    # 旧格式 compare.json 的 items 不带 bbox_xy
    cmp_legacy = {'items': [
        {'code': 'A', 'status': 'alarm', 'alarms': [{'type': 'dtheta'}]},
        {'code': 'C_ON_FOREIGN', 'status': 'alarm', 'alarms': [{'type': 'ncc'}]},
        {'code': 'FP_FAR', 'status': 'alarm', 'alarms': [{'type': 'ncc'}]},
    ], 'foreign': []}
    parts = {'instances': [
        {'code': 'A', 'bbox_xy': [0, 0, 50, 50]},
        {'code': 'C_ON_FOREIGN', 'bbox_xy': [3010, 110, 3070, 170]},
        {'code': 'FP_FAR', 'bbox_xy': [20000, 100, 20080, 180]},
    ]}
    res = score.score_compare(truth, cmp_legacy, None, parts=parts)
    assert res['hit'] == 2  # D2 落在子件 C_ON_FOREIGN 上通过 parts.json bbox_xy 计为检出
    assert res['synth_far_alarms'] == 1  # FP_FAR 正确计为 1 条远离缺陷误报


# ---------- P1-7 & P1-8: Trip.region 越界/流式降采样 & calib/align/match 边界保护 ----------

def test_trip_region_vertical_bounds_and_streaming_downsample(tmp_path):
    src = tmp_path / 'frames'
    src.mkdir()
    rng = np.random.default_rng(123)
    full = np.hstack([rng.integers(20, 230, (H, W), dtype=np.uint8) for _ in range(3)])
    for i in range(3):
        io.imwrite(src / f'{i + 1}.jpg', full[:, i * W:(i + 1) * W], quality=95)
    t = Trip(src)
    # 1. 纵向上下越界与浮点坐标不崩溃、越界区补 0
    top = t.region(100.0, 200.0, -30, 50)
    assert top.shape == (80, 100) and top[:30].max() == 0 and top[30:].mean() > 20
    bot = t.region(100, 200, H - 40, H + 30)
    assert bot.shape == (70, 100) and bot[40:].max() == 0 and bot[:40].mean() > 20
    outside = t.region(100, 200, H + 10, H + 50)
    assert outside.shape == (40, 100) and outside.max() == 0
    # 2. 大跨度 step>1 触发流式分块降采样，结果与整块 resize 一致
    streamed = t.region(0, 3 * W, 0, H, step=16)
    expected = cv2.resize(t._copy_raw(0, 3 * W, 0, H), (3 * W // 16, H // 16), interpolation=cv2.INTER_AREA)
    assert streamed.shape == expected.shape
    assert np.max(np.abs(streamed.astype(int) - expected.astype(int))) <= 1


def test_calib_refine_cap_no_zero_division(monkeypatch):
    im = np.full((600, 600), 120, np.uint8)
    monkeypatch.setattr(calib, 'fit_ellipse', lambda pts: {
        'cx': 0.0, 'cy': 0.0, 'a': 0.0, 'b': 0.0, 'ang': 0.0, 'n': 0, 'sd': 0.0, 'off': 0.0, 'ok': False,
    })
    res = calib.refine_cap(FakeTrip(im), 300, 300, cap=True, crop=100, band=(20, 60))
    assert res['ok'] is False


def test_align_rejects_fewer_than_two_valid_blocks_and_filters_nan(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    # 造全常数图（会产生 NaN 相关），保证不产生未处理异常且显式报有效块不足
    blank = np.full((H, 2 * W), 128, dtype=np.uint8)
    for trip in ('T01', 'T02'):
        src = root / '数据' / '原始' / trip / 'camImg'
        src.mkdir(parents=True, exist_ok=True)
        for f in (1, 2):
            io.imwrite(src / f'{f}.jpg', blank[:, (f - 1) * W:f * W])
        io.write_json(io.trip_dir(trip) / 'trip.json', {'trip': trip, 'src_dir': f'数据/原始/{trip}/camImg', 'n_frames': 2})
        io.write_json(io.trip_dir(trip) / 'manifest.json', {
            'trip': trip, 'train_x0': 0, 'train_x1': 2 * W, 'head_x': 0, 'train_panels': [{'car': 1, 'x0': 0, 'x1': 2 * W}]})
    monkeypatch.setattr(align, 'match_anchors', lambda *a, **k: [
        {'id': f'A{i}', 'car': 1, 'u': i * 500, 'v': 1000, 'x_tpl': i * 500, 'x': i * 500, 'y': 1000,
         'ncc': 0.9, 'peak_margin': 0.2, 'ok': True} for i in range(4)
    ])
    with pytest.raises(SystemExit, match='有效加密块不足'):
        align.run(SimpleNamespace(trip='T02', template='v1'))

    # Pass-2 必须使用 amap1.dy_at_x 平滑后的 dy，而不是 Pass-1 离群块的原始 dy
    seen_dys = []

    def fake_dense(_tn, _tt, x1s, _pred):
        return [(x1, float(x1), 10.0 if i != 1 else 999.0, 0.9 if i != 1 else 0.1) for i, x1 in enumerate(x1s)]

    def fake_refine(_tn, _tt, x1s, amap1, head_t, dys=None, reach=40):
        seen_dys.extend(float(d) for d in dys)
        return [(x1, float(x1), float(d), 0.95) for x1, d in zip(x1s, dys)]

    monkeypatch.setattr(align, 'dense', fake_dense)
    monkeypatch.setattr(align, 'refine', fake_refine)
    monkeypatch.setattr(align, '_overlay', lambda *a, **k: None)
    align.run(SimpleNamespace(trip='T02', template='v1'))
    assert seen_dys and all(abs(d - 10.0) < 1e-6 for d in seen_dys)


def test_match_aligned_region_extreme_dy_and_tier3_parent_pfound(proj, monkeypatch):
    # 极端 dy 导致 y_hi 贴边时不除零
    canvas = np.full((H, 2000), 100, dtype=np.uint8)
    amap_neg = AlignMap([0, 2000], [0, 2000], [-5000, -5000])
    amap_pos = AlignMap([0, 2000], [0, 2000], [5000, 5000])
    assert match.aligned_region(FakeTrip(canvas), amap_neg, 100, 300, 100, 300).shape == (200, 200)
    assert match.aligned_region(FakeTrip(canvas), amap_pos, 100, 300, 100, 300).shape == (200, 200)

    # 父件第一层超出 PARENT_REACH，靠第三层大窗找回后回写 pfound 供子件使用
    monkeypatch.setattr(match, 'PARENT_REACH', 4)
    orig_search = match._search
    seen_child_offs = []

    def spy_search(t_new, amap, crop, mask, org_uv, off, reach):
        if crop.shape == (100, 100):  # BOLT 子件
            seen_child_offs.append(off)
        return orig_search(t_new, amap, crop, mask, org_uv, off, reach)

    # 人为给对齐引入 20px 偏差（> PARENT_REACH=4），父件第一层失败，第三层找回约 +20px
    al = io.read_json(io.trip_dir('T02') / 'align.json')
    for p in al['g']:
        p['u'] += 20
    io.write_json(io.trip_dir('T02') / 'align.json', al)
    monkeypatch.setattr(match, '_search', spy_search)
    cli.main(['match', 'T02'])
    assert seen_child_offs and abs(seen_child_offs[0][0] - 20.0) < 5.0


# ---------- P1-9: Web 后台任务锁、同趟互斥、partial 识别与取消接口 ----------

def test_job_lines_thread_safe_and_partial_with_trailing_json():
    job = jobs.Job('run', 'T09', [])
    stop = threading.Event()
    errors = []

    def writer():
        for i in range(1500):
            with job._lock:
                job.lines.append(f'line-{i}')
        stop.set()

    def reader():
        while not stop.is_set():
            try:
                job.view(tail=50)
            except Exception as e:
                errors.append(e)

    tw, tr = threading.Thread(target=writer), threading.Thread(target=reader)
    tw.start()
    tr.start()
    tw.join()
    tr.join()
    assert not errors

    # 尾部有非 run 的 JSON 行时，仍能向前找到 partial 摘要（验证 break 缩进修复）
    j2 = jobs.Job('run', 'T09', [])
    j2.proc = SimpleNamespace(
        stdout=StringIO(json.dumps({'step': 'run', 'status': 'partial'}) + '\n'
                        + json.dumps({'trailing': True}) + '\n'),
        wait=lambda: 2,
    )
    jobs._pump(j2)
    assert j2.view()['status'] == 'partial'


def test_web_same_trip_conflict_and_cancel_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'step_ready', lambda step: True)
    monkeypatch.setattr(jobs, 'command', lambda step, argv: [sys.executable, '-c', 'import time; time.sleep(30)'])
    with TestClient(create_app()) as c:
        r1 = c.post('/api/run', json={'src': str(tmp_path), 'trip': 'T71'})
        assert r1.status_code == 200
        jid = r1.json()['id']
        # 同一趟并发触发应返回 409
        r2 = c.post('/api/run', json={'src': str(tmp_path), 'trip': 'T71'})
        assert r2.status_code == 409
        # 取消运行中的任务
        rc = c.delete(f'/api/jobs/{jid}')
        assert rc.status_code == 200 and rc.json()['status'] == 'cancelled'
        assert c.delete('/api/jobs/nonexistent').status_code == 404


# ---------- P1-10 & P2: query NULL 绑定、SQL 限制、app/data 越界框与 bbox_xy 优先、前端脱敏 ----------

def test_query_null_type_and_sql_limits(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    dbp = root / '库.sqlite'
    db.build(dbp, ['T01', 'T02'], 'v1')
    # 写入一条 type=NULL 的实例，验证 part_history 不报绑定数不匹配
    con = sqlite3.connect(dbp)
    try:
        con.execute("INSERT INTO part_instance (kind, trip_id, code, type) VALUES ('trip', 'T02', 'C1-B1-BOLT-02', NULL)")
        con.commit()
    finally:
        con.close()
    h = query.part_history('C1-B1-BOLT-02', db_path=dbp)
    assert h['part_type'][0]['type'] == 'BOLT'

    # max_rows 截断
    r_lim = query.sql('WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n < 50) SELECT n FROM c',
                      db_path=dbp, max_rows=10)
    assert len(r_lim['rows']) == 10 and r_lim['truncated'] is True

    # 超时中断
    with pytest.raises(ValueError, match='超时'):
        query.sql('WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n < 5000000) SELECT sum(n) FROM c',
                  db_path=dbp, timeout_s=0.001)


def test_app_crop_out_of_bounds_and_foreign_bbox_xy_priority(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    src = root / '数据' / '原始' / 'T02' / 'camImg'
    src.mkdir(parents=True, exist_ok=True)
    im = np.full((H, W), 120, dtype=np.uint8)
    io.imwrite(src / '1.jpg', im)
    io.write_json(io.trip_dir('T02') / 'trip.json', {'trip': 'T02', 'src_dir': str(src), 'n_frames': 1})
    io.write_json(io.trip_dir('T02') / 'manifest.json', {
        'trip': 'T02', 'train_x0': 0, 'train_x1': W, 'head_x': 0, 'frame_w': W, 'frame_h': H,
        'train_panels': [{'panel': 1, 'car': 1, 'x0': 0, 'x1': W, 'width_px': W, 'height_px': H}]})
    cmp_ = io.read_json(io.trip_dir('T02') / 'compare.json')
    # 故意让 bbox_xy 与 uv_to_xy 不同，验证 alarms / report 优先使用 bbox_xy
    cmp_['foreign'] = [{'bbox_uv': [100, 100, 200, 200], 'bbox_xy': [500, 600, 620, 720], 'area': 14400}]
    io.write_json(io.trip_dir('T02') / 'compare.json', cmp_)
    data._crop.cache_clear()
    dzi._reader.cache_clear()

    al = data.alarms('T02')
    fo = next(r for r in al['rows'] if r['kind'] == 'foreign')
    assert fo['xy'] == [500.0, 600.0, 620.0, 720.0]

    # 越界框返回 NotFound（404）而非崩溃
    with pytest.raises(data.NotFound):
        data._crop(str(io.trip_dir('T02')), 1, 'T02', (999999, 100, 1000100, 200), (0, 0, 255))

    # report.py 也优先使用 foreign['bbox_xy']
    rep_out = root / 'rep_t02'
    report.write_report('T02', rep_out)
    html = (rep_out / 'index.html').read_text(encoding='utf-8')
    assert 'x 500–620' in html


def test_frontend_sanitized_and_eff_cmp_mode():
    static_dir = Path(data.__file__).parent / 'static'
    html = (static_dir / 'index.html').read_text(encoding='utf-8')
    js = (static_dir / 'app.js').read_text(encoding='utf-8')
    assert 'ZXB_LC05D' not in html
    assert 'function effCmpMode()' in js
    assert "fetch('ocr.html'" not in js
    assert 'fmtDiffArea' in js
