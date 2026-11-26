"""网页启动的处理任务必须使用用户选择的分割权重和模板，并满足路径白名单与非回环鉴权安全边界。"""
from fastapi.testclient import TestClient
import pytest

from railside import io
from railside.app import jobs
from railside.app.server import create_app


def test_web_run_forwards_model_and_template(tmp_path, monkeypatch):
    weights = tmp_path / '模型.pt'
    weights.write_bytes(b'fixture: only argument forwarding is tested')
    monkeypatch.setenv('RAILSIDE_MODELS_DIR', str(tmp_path))
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
    assert calls == [[str(tmp_path), '--trip', 'T09', '--template', 'v2', '--weights', str(weights.resolve())]]


def test_web_run_accepts_onnx_in_root_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    monkeypatch.delenv('RAILSIDE_MODELS_DIR', raising=False)
    models = tmp_path / 'models'
    models.mkdir()
    weights = models / 'seg.onnx'
    weights.write_bytes(b'onnx')
    calls = []

    class Job:
        def view(self):
            return {'id': 'run-onnx', 'status': 'running'}

    monkeypatch.setattr(jobs, 'start', lambda kind, step, argv, trip, meta=None: (calls.append(argv), Job())[1])
    res = TestClient(create_app()).post('/api/run', json={
        'src': str(tmp_path), 'trip': 'T09', 'weights': 'models/seg.onnx'})
    assert res.status_code == 200
    assert calls == [[str(tmp_path), '--trip', 'T09', '--weights', str(weights.resolve())]]


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


def test_web_run_rejects_unc_and_untrusted_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    monkeypatch.delenv('RAILSIDE_MODELS_DIR', raising=False)
    outside = tmp_path / 'outside.pt'
    outside.write_bytes(b'pt')
    models = tmp_path / 'models'
    models.mkdir()
    bad_ext = models / 'evil.py'
    bad_ext.write_text('print(1)', encoding='utf-8')

    monkeypatch.setattr(jobs, 'start', lambda *a, **k: pytest.fail('must not start'))
    c = TestClient(create_app())
    for unc in (r'\\192.168.1.9\share\camImg', '//192.168.1.9/share/camImg'):
        r = c.post('/api/run', json={'src': unc, 'trip': 'T09'})
        assert r.status_code == 400 and 'UNC' in r.json()['error']
        r2 = c.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09', 'weights': unc})
        assert r2.status_code == 400 and 'UNC' in r2.json()['error']
    r_out = c.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09', 'weights': str(outside)})
    assert r_out.status_code == 400 and '模型目录' in r_out.json()['error']
    r_ext = c.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09', 'weights': str(bad_ext)})
    assert r_ext.status_code == 400 and '.pt' in r_ext.json()['error']


def test_non_loopback_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv('RAILSIDE_APP_TOKEN', raising=False)

    class Job:
        def view(self):
            return {'id': 'run-auth', 'status': 'running'}

    monkeypatch.setattr(jobs, 'start', lambda *a, **k: Job())

    # 非回环且未配置 Token：直接拒绝写操作（403）
    c_no_tok = TestClient(create_app(host='0.0.0.0'))
    r = c_no_tok.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'})
    assert r.status_code == 403 and 'RAILSIDE_APP_TOKEN' in r.json()['error']
    assert c_no_tok.delete('/api/jobs/run-1').status_code == 403

    # 非回环且配置 Token：无 Token/错误 Token 401，Bearer 与 X-Railside-Token 均放行
    monkeypatch.setenv('RAILSIDE_APP_TOKEN', 'secret-token-123')
    c_tok = TestClient(create_app(host='0.0.0.0'))
    assert c_tok.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'}).status_code == 401
    assert c_tok.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'},
                      headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert c_tok.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'},
                      headers={'Authorization': 'Bearer secret-token-123'}).status_code == 200
    assert c_tok.post('/api/run', json={'src': str(tmp_path), 'trip': 'T09'},
                      headers={'X-Railside-Token': 'secret-token-123'}).status_code == 200
