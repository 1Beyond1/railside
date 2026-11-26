"""同一趟号重新处理时，不允许混入上次运行的识别或告警结果。"""
import json

import pytest

from fixtures.two_trips import make_project
from railside import io, pipeline
from railside.app import dzi


def test_rerun_without_weights_cannot_consume_previous_segmentation(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    (root / 'new-src').mkdir()
    trip = 'T02'
    d = io.trip_dir(trip)
    io.write_json(d / 'seg.json', {'model': {'name': 'old-model'}, 'dets': [{'stale': True}]})
    old = (d / 'seg.json').read_bytes()
    seen = []

    def spawn(argv):
        step = argv[0]
        if step == 'stitch':
            io.write_json(d / 'manifest.json', io.read_json(io.trip_dir('T01') / 'manifest.json'))
        if step == 'align':
            io.write_json(d / 'align.json', {'template_ver': 'v1'})
        if step == 'match':
            seen.append((d / 'seg.json').exists())
            io.write_json(d / 'parts.json', {'instances': []})
        return 0, json.dumps({'step': step, 'verdict': 'pass'}), ''

    monkeypatch.setattr(pipeline, '_spawn', spawn)
    pipeline.run_trip(root / 'new-src', trip)
    assert seen == [False]
    assert not (d / 'seg.json').exists()
    assert any(p.read_bytes() == old for p in d.glob('previous_runs/*/seg.json'))


def test_missing_src_does_not_archive_previous_run(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    d = io.trip_dir('T02')
    old_parts = (d / 'parts.json').read_bytes()
    with pytest.raises(FileNotFoundError, match='原图目录不存在'):
        pipeline.run_trip(root / 'absent-src', 'T02')
    assert (d / 'parts.json').read_bytes() == old_parts
    assert not list(d.glob('previous_runs/*'))


def test_requested_template_is_not_overridden_by_previous_alignment(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch)
    assert '模板 v2 不存在' in pipeline.prereq('align', 'T02', 'v2')


def test_rerun_cannot_compare_old_parts_after_alignment_is_skipped(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    (root / 'new-src').mkdir()
    d = io.trip_dir('T02')
    old_parts = (d / 'parts.json').read_bytes()
    steps = []

    def spawn(argv):
        steps.append(argv[0])
        if argv[0] == 'stitch':
            io.write_json(d / 'manifest.json', io.read_json(io.trip_dir('T01') / 'manifest.json'))
        return 0, json.dumps({'step': argv[0], 'verdict': 'pass'}), ''

    monkeypatch.setattr(pipeline, '_spawn', spawn)
    result = pipeline.run_trip(root / 'new-src', 'T02', template='absent')
    assert 'match' not in steps and 'compare' not in steps
    assert not (d / 'parts.json').exists()
    assert not (d / 'compare.json').exists()
    assert any(p.read_bytes() == old_parts for p in d.glob('previous_runs/*/parts.json'))
    assert result['previous_run']


def test_match_rejects_mapping_from_another_template(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch)
    assert '不一致' in pipeline.prereq('match', 'T02', 'v2')


def test_web_reader_refreshes_when_trip_source_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'MID', tmp_path / '中间')
    for name in ('old-source', 'new-source'):
        source = tmp_path / name
        source.mkdir()
        (source / '1.jpg').write_bytes(b'frame enumeration only')
        io.write_json(io.trip_dir('T09') / 'trip.json', {'src_dir': str(source)})
        assert dzi.reader('T09').src == str(source)
