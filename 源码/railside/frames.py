"""按趟读原图：帧列表、按趟坐标取区域。"""
import functools
import os
from pathlib import Path

import cv2
import numpy as np

from . import io
from .coords import W, H


def list_frames(src):
    """返回 {帧号: 路径}，帧号取文件名里的数字。"""
    out = {}
    for name in os.listdir(src):
        stem, ext = os.path.splitext(name)
        if ext.lower() == '.jpg' and stem.isdigit():
            out[int(stem)] = os.path.join(src, name)
    return dict(sorted(out.items()))


class Trip:
    """一趟原图。区域读取用趟坐标（时间顺序、不翻转）。缺帧的位置补 0。"""

    def __init__(self, src, cache=24):
        self.src = str(src)
        self.frames = list_frames(self.src)
        if not self.frames:
            raise FileNotFoundError(f'没有帧: {src}')
        self.n = max(self.frames)
        self.width = self.n * W
        self._read = functools.lru_cache(maxsize=cache)(self._read_frame)

    @classmethod
    def from_id(cls, trip, **kw):
        meta = io.read_json(io.trip_dir(trip) / 'trip.json')
        src = Path(meta['src_dir'])
        return cls(src if src.is_absolute() else io.ROOT / src, **kw)

    def _read_frame(self, f):
        p = self.frames.get(f)
        if p is None:
            return None
        im = io.imread(p)
        if im.shape != (H, W):
            raise ValueError(f'帧尺寸不对: {p} {im.shape}')
        return im

    def frame(self, f):
        return self._read(f)

    def region(self, x0, x1, y0=0, y1=H, step=1):
        """取趟坐标 [x0, x1) × [y0, y1) 的区域，step>1 时按步长缩小（区域平均）。"""
        x0, x1 = int(x0), int(x1)
        out = np.zeros((y1 - y0, x1 - x0), np.uint8)
        xa = max(x0, 0)
        while xa < min(x1, self.width):
            f = xa // W + 1
            c0 = xa % W
            c1 = min(W, c0 + (x1 - xa))
            im = self._read(f)
            if im is not None:
                out[:, xa - x0: xa - x0 + (c1 - c0)] = im[y0:y1, c0:c1]
            xa += c1 - c0
        if step > 1:
            h, w = out.shape
            out = cv2.resize(out, (max(1, w // step), max(1, h // step)), interpolation=cv2.INTER_AREA)
        return out

    def strip(self, f0, f1, step):
        """帧 f0..f1（含）按时间顺序拼成的缩小条带，缺帧补 0。"""
        w, h = W // step, H // step
        out = np.zeros((h, w * (f1 - f0 + 1)), np.uint8)
        for k, f in enumerate(range(f0, f1 + 1)):
            im = self._read(f)
            if im is not None:
                out[:, k * w:(k + 1) * w] = im if step == 1 else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        return out
