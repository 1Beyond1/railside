"""无纹理或空掩膜不应给出“完美匹配”。"""
import numpy as np
import pytest

from railside import match
from railside.coords import AlignMap
from test_match import FakeTrip, photo

IDENT = AlignMap([0, 10000], [0, 10000])


@pytest.mark.parametrize('level', [0, 150, 255])
def test_constant_template_cannot_match_an_absent_part(level):
    patch = np.full((100, 100), level, np.uint8)
    result = match._search(FakeTrip(np.zeros((1400, 1800), np.uint8)), IDENT, patch,
                           np.full_like(patch, 255), (950, 950), (0.0, 0.0), 40)
    assert result is None or result['score'] < match.ACCEPT_PARENT


def test_empty_mask_cannot_match_a_visible_patch():
    patch = photo(100, 100)
    canvas = np.zeros((1400, 1800), np.uint8)
    canvas[950:1050, 950:1050] = patch
    result = match._search(FakeTrip(canvas), IDENT, patch, np.zeros_like(patch), (950, 950), (0.0, 0.0), 40)
    assert result is None
