"""按趟坐标改一份原图拷贝：读需要的帧、改像素、只重写改过的帧，其余帧逐字节拷贝。"""
import shutil
from pathlib import Path

import cv2
import numpy as np

from .. import io
from ..coords import H, W
from ..frames import list_frames


class FrameCopy:
    def __init__(self, src, dst, quality=98):
        self.src, self.dst = Path(src), Path(dst)
        self.frames = list_frames(self.src)
        if not self.frames:
            raise FileNotFoundError(f'没有帧: {src}')
        self.quality = quality
        self.cache = {}
        self.dirty = set()

    def _frame(self, f):
        if f not in self.cache:
            self.cache[f] = io.imread(self.frames[f])
        return self.cache[f]

    def get(self, x0, x1, y0, y1):
        """趟坐标区域 [x0,x1)×[y0,y1) 的拷贝（可跨帧）。"""
        y0, y1 = max(0, int(y0)), min(H, int(y1))
        out = []
        for f in range(int(x0) // W + 1, (int(x1) - 1) // W + 2):
            a, b = max(int(x0), (f - 1) * W), min(int(x1), f * W)
            out.append(self._frame(f)[y0:y1, a - (f - 1) * W: b - (f - 1) * W])
        return np.hstack(out).copy(), (int(x0), y0)

    def put(self, im, origin):
        x0, y0 = origin
        h, w = im.shape
        for f in range(x0 // W + 1, (x0 + w - 1) // W + 2):
            a, b = max(x0, (f - 1) * W), min(x0 + w, f * W)
            self._frame(f)[y0:y0 + h, a - (f - 1) * W: b - (f - 1) * W] = im[:, a - x0: b - x0]
            self.dirty.add(f)

    def set_frame(self, f, im):
        self.cache[f] = im
        self.dirty.add(f)

    def write(self, drop=()):
        """写出拷贝：drop 里的帧不写（造缺帧），改过的帧重新编码，其余逐字节拷贝。"""
        if self.dst.exists():
            shutil.rmtree(self.dst)
        self.dst.mkdir(parents=True)
        for f, p in self.frames.items():
            out = self.dst / Path(p).name
            if f in drop:
                continue
            if f in self.dirty:
                ok, buf = cv2.imencode('.jpg', self.cache[f], [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                assert ok
                buf.tofile(str(out))
            else:
                shutil.copyfile(p, out)
        return self.dst
