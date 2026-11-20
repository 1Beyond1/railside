import numpy as np
import pytest

from railside.coords import AlignMap, raw_to_trip, trip_to_raw
from railside.tiles import parse_tile_name, tile_name, tile_origins


def test_raw_trip_roundtrip():
    x, y = raw_to_trip(30, 1999, 7)
    assert (x, y) == (29 * 2000 + 1999, 7)
    f, c, r = trip_to_raw(x, y)
    assert (f, c, r) == (30, 1999, 7)


def test_alignmap_inverse_and_extrapolate():
    m = AlignMap([0, 1000, 3000], [100, 1150, 3200], [0, 2, 4])
    u = m.x_to_u([500, 2000, 4000, -100])
    assert np.allclose(m.u_to_x(u), [500, 2000, 4000, -100])
    uu, vv = m.to_template(2000, 10)
    xx, yy = m.to_trip(uu, vv)
    assert np.isclose(xx, 2000) and np.isclose(yy, 10)


def test_alignmap_rejects_non_monotone():
    with pytest.raises(ValueError):
        AlignMap([0, 1, 2], [0, 5, 3])


def test_tiles_cover_range():
    xs = tile_origins(56400, 568909)
    assert xs[0] == 56400 and xs[-1] + 4096 == 568909
    assert all(b - a == 3584 for a, b in zip(xs[:-2], xs[1:-1]))
    assert parse_tile_name(tile_name('T01', xs[3])) == ('T01', xs[3])
