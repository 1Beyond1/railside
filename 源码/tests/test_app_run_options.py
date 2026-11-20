"""网页启动的处理任务必须使用用户选择的分割权重和模板。"""
from fastapi.testclient import TestClient
import pytest

from railside.app import jobs
from railside.app.server import create_app


def test_web_run_forwards_model_and_template(tmp_path, monkeypatch):
    weights = tmp_path / '模型.pt'
    weights.write_bytes(b'fixture: only argument forwarding is tested')
    calls = []

    class Job:
        def view(self):
            return {'id': 'run-test', 'status': 'running'}

    def start(kind, step, argv, trip, meta=None):
        calls.append(argv)
        return Job()

    monkeypatch.setattr(jobs, 'start', start)
    result = TestClient(create_app()).post('/api/run', json={
        'src': str(tmp_path), 'trip': 'T09', 'weights': str(weights), 'template': 'v2'})
    assert result.status_code == 200
    assert calls == [[str(tmp_path), '--trip', 'T09', '--template', 'v2', '--weights', str(weights)]]


@pytest.mark.parametrize('options', [{}, {'weights': ''}, {'weights': None}])
def test_web_run_remains_compatible_without_weights(tmp_path, monkeypatch, options):
    calls = []

    class Job:
        def view(self):
            return {'status': 'running'}

    def start(kind, step, argv, trip, meta=None):
        calls.append(argv)
        return Job()

    monkeypatch.setattr(jobs, 'start', start)
    result = TestClient(create_app()).post('/api/run', json={'src': str(tmp_path), 'trip': 'T09', **options})
    assert result.status_code == 200
    # 没选模板：不替用户填版本，交给 railside run 取最新的（不再默认作废的 v1）
    assert calls == [[str(tmp_path), '--trip', 'T09']]


@pytest.mark.parametrize('options', [{'weights': 'nonexistent.pt'}, {'template': '../v2'}])
def test_bad_model_options_do_not_start_processing(tmp_path, monkeypatch, options):
    def unexpected(*a, **k):
        pytest.fail('invalid options must not start a subprocess')

    monkeypatch.setattr(jobs, 'start', unexpected)
    result = TestClient(create_app()).post('/api/run', json={'src': str(tmp_path), 'trip': 'T09', **options})
    assert result.status_code == 400
