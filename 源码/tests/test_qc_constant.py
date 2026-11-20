"""全黑、全白等坏帧的质检结果仍必须是合法 JSON。"""
import json

import numpy as np
import pytest

from railside import io, qc, cli


@pytest.mark.parametrize('level', [0, 150, 255])
def test_constant_frame_stats_are_finite(level):
    result = qc.frame_stats(np.full((256, 320), level, np.uint8))
    assert result['blur_r'] == 0.0
    json.dumps(result, allow_nan=False)


def test_black_frame_ingest_writes_strict_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(io, 'MID', tmp_path / '中间')
    src = tmp_path / 'camImg'
    io.imwrite(src / '1.jpg', np.zeros((4096, 2000), np.uint8))
    cli.main(['ingest', str(src), '--trip', 'T09'])
    result = io.read_json(io.trip_dir('T09') / 'qc.json')
    assert result['verdict'] == 'fail'
    assert 'dark' in result['per_frame'][0]['flags']
    json.dumps(result, allow_nan=False)
