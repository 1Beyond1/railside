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

    def _copy_raw(self, x0, x1, y0, y1):
        h, w = y1 - y0, x1 - x0
        out = np.zeros((h, w), np.uint8)
        ya, yb = max(y0, 0), min(y1, H)
        if ya >= yb:
            return out
        dy = ya - y0
        xa = max(x0, 0)
        xb = min(x1, self.width)
        while xa < xb:
            f = xa // W + 1
            c0 = xa % W
            c1 = min(W, c0 + (xb - xa))
            im = self._read(f)
            if im is not None:
                out[dy:dy + (yb - ya), xa - x0: xa - x0 + (c1 - c0)] = im[ya:yb, c0:c1]
            xa += c1 - c0
        return out

    def region(self, x0, x1, y0=0, y1=H, step=1):
        """取趟坐标 [x0, x1) × [y0, y1) 的区域，step>1 时按步长缩小（区域平均）。越界补 0。"""
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        step = max(1, int(step))
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            return np.zeros((max(0, h), max(0, w)), np.uint8)
        if step == 1:
            return self._copy_raw(x0, x1, y0, y1)
        dw, dh = max(1, w // step), max(1, h // step)
        if w * h <= 4_000_000 or dw <= 1:
            out = self._copy_raw(x0, x1, y0, y1)
            return cv2.resize(out, (dw, dh), interpolation=cv2.INTER_AREA)
        # 大跨度降采样：按目标列分块边读边缩写入目标缓冲，降低内存峰值
        dst = np.zeros((dh, dw), np.uint8)
        chunk_dw = max(1, W // step)
        for j0 in range(0, dw, chunk_dw):
            j1 = min(dw, j0 + chunk_dw)
            sx0 = x0 + (j0 * w) // dw
            sx1 = x0 + (j1 * w) // dw
            if sx1 <= sx0:
                sx1 = sx0 + 1
            chunk = self._copy_raw(sx0, sx1, y0, y1)
            dst[:, j0:j1] = cv2.resize(chunk, (j1 - j0, dh), interpolation=cv2.INTER_AREA)
        return dst

    def strip(self, f0, f1, step):
        """帧 f0..f1（含）按时间顺序拼成的缩小条带，缺帧补 0。"""
        w, h = W // step, H // step
        out = np.zeros((h, w * (f1 - f0 + 1)), np.uint8)
        for k, f in enumerate(range(f0, f1 + 1)):
            im = self._read(f)
            if im is not None:
                out[:, k * w:(k + 1) * w] = im if step == 1 else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        return out
