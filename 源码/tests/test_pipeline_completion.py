"""缺核心检测步骤的任务不能作为检测成功交付。"""
import json
from io import StringIO
from types import SimpleNamespace

import pytest

from fixtures.two_trips import make_project
from railside import cli, io, pipeline
from railside.app import jobs


def test_missing_template_is_partial_and_cli_returns_nonzero(tmp_path, monkeypatch, capsys):
    root = make_project(tmp_path, monkeypatch)
    src = root / 'src'
    src.mkdir()

    def spawn(argv):
        if argv[0] == 'stitch':
            io.write_json(io.trip_dir('T09') / 'manifest.json',
                          io.read_json(io.trip_dir('T01') / 'manifest.json'))
        return 0, json.dumps({'step': argv[0], 'verdict': 'pass', 'out': 'report.html'}), ''

    monkeypatch.setattr(pipeline, '_spawn', spawn)
    with pytest.raises(SystemExit) as exc:
        cli.main(['run', str(src), '--trip', 'T09', '--template', 'absent'])
    assert exc.value.code != 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result['status'] == 'partial'
    assert result['report'] == 'report.html'
    assert {'align', 'match', 'compare'} <= set(result['incomplete_steps'])


def test_web_job_keeps_partial_status_from_pipeline():
    job = jobs.Job('run', 'T09', [])
    job.proc = SimpleNamespace(
        stdout=StringIO(json.dumps({'step': 'run', 'status': 'partial'}) + '\n'),
        wait=lambda: 2)
    jobs._pump(job)
    assert job.view()['status'] == 'partial'
    assert job.rc == 2


def test_optional_segmentation_does_not_make_template_detection_partial(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, 'prereq', lambda *a: None)
    monkeypatch.setattr(pipeline, '_spawn', lambda args: (0, json.dumps({'step': args[0]}), ''))
    result = pipeline.run_trip(root, 'T09')
    assert result['status'] == 'ok'
    assert result['incomplete_steps'] == []


def test_requested_segmentation_module_is_required(tmp_path, monkeypatch):
    root = make_project(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, 'prereq', lambda *a: None)
    monkeypatch.setattr(pipeline, '_spawn', lambda args: (0, json.dumps({'step': args[0]}), ''))
    monkeypatch.setattr(pipeline, 'available', lambda step: (step != 'segment', 'missing module'))
    result = pipeline.run_trip(root, 'T09', weights='model.pt')
    assert result['status'] == 'partial'
    assert result['incomplete_steps'] == ['segment']
