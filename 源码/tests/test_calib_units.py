"""物理量回归：同一个圆在横、纵方向换算后半径必须相等。"""
from types import SimpleNamespace

import pytest

from railside import calib, io


@pytest.mark.parametrize('ratio', [0.95, 1.0, 1.1])
def test_calibration_preserves_circle_radius(tmp_path, monkeypatch, ratio):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    monkeypatch.setattr(io, 'MID', tmp_path / '中间')
    io.write_json(io.trip_dir('T99') / 'manifest.json', {})
    monkeypatch.setattr(calib.Trip, 'from_id', lambda trip: object())
    left = dict(x=1000., y=2800., a=400., b=400. * ratio,
                ratio=ratio, sd=0., ok=True)
    right = dict(left, x=11000.)
    bogie = dict(left=left, right=right, x=6000., wheelbase_px=10000.,
                 car=1, zone='B1', ok=True)
    monkeypatch.setattr(calib, 'detect_axles', lambda *a: ([bogie], []))
    calib.run(SimpleNamespace(trip='T99', out=None))
    result = io.read_json(io.trip_dir('T99') / 'calib.json')
    horizontal_radius = left['a'] * result['sx_mean_mm_per_px']
    vertical_radius = left['b'] * result['sy_mm_per_px']
    assert vertical_radius == pytest.approx(horizontal_radius, abs=0.001)
