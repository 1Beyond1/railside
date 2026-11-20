"""整列长图的 DZI 金字塔：按需生成瓦片并缓存。

显示图 = 车内范围 [train_x0, train_x1) 水平翻转（车头在右）。显示列 xd 与趟坐标 x 的关系见
`coords.display_x`：xd = train_x1 - 1 - x。

- 缩小倍数 ≥ 8 的层从 `预览/整车_1-8.jpg` 缩放得到（它已翻转、已裁到车内）；
- 更细的层用 `Trip.region(..., step=s)` 从原图取，再翻转。
瓦片缓存到 `数据/中间/<趟>/dzi/<train_x0>-<train_x1>/<层>/<列>_<行>.jpg`；车内范围变了（重跑 stitch）自动换目录。
"""
import functools
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import io
from ..frames import Trip

TILE = 512
OVERLAP = 1
PREVIEW_STEP = 8
QUALITY = 85


class TileMissing(Exception):
    """请求的层或瓦片超出范围。"""


@dataclass(frozen=True)
class Pyramid:
    trip: str
    x0: int          # train_x0
    x1: int          # train_x1
    width: int       # 显示图宽 = x1 - x0
    height: int
    max_level: int

    def level(self, level):
        """第 level 层的缩小倍数和尺寸。"""
        if not 0 <= level <= self.max_level:
            raise TileMissing(f'没有第 {level} 层（0–{self.max_level}）')
        s = 1 << (self.max_level - level)
        return s, -(-self.width // s), -(-self.height // s)

    def dzi_xml(self):
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
                f'Format="jpg" Overlap="{OVERLAP}" TileSize="{TILE}">'
                f'<Size Width="{self.width}" Height="{self.height}"/></Image>')

    def cache_dir(self):
        return io.trip_dir(self.trip) / 'dzi' / f'{self.x0}-{self.x1}'


def pyramid(trip):
    p = io.trip_dir(trip) / 'manifest.json'
    return _pyramid(trip, str(p), p.stat().st_mtime_ns)


@functools.lru_cache(maxsize=16)
def _pyramid(trip, path, _mtime):
    m = io.read_json(path)
    x0, x1 = int(m['train_x0']), int(m['train_x1'])
    w, h = x1 - x0, int(m.get('frame_h', 4096))
    return Pyramid(trip, x0, x1, w, h, math.ceil(math.log2(max(w, h))))


def reader(trip):
    """按趟缓存的原图读取器（帧解码有 LRU 缓存）。"""
    p = io.trip_dir(trip) / 'trip.json'
    st = p.stat()
    return _reader(str(p), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=4)
def _reader(trip_json, _mtime, _size):
    meta = io.read_json(trip_json)
    src = Path(meta['src_dir'])
    return Trip(src if src.is_absolute() else io.ROOT / src, cache=32)


_preview_lock = threading.Lock()


def preview_path(trip):
    return io.trip_dir(trip) / '预览' / '整车_1-8.jpg'


@functools.lru_cache(maxsize=48)
def _preview_level(path, mtime, s, w, h):
    """整车 1/8 预览缩放到第 s 倍层的整幅图（只缩放一次，之后裁瓦片）。"""
    with _preview_lock:
        base = _preview(path, mtime)
    if s == PREVIEW_STEP and base.shape[1] == w and base.shape[0] == h:
        return base
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)


@functools.lru_cache(maxsize=2)
def _preview(path, _mtime):
    return io.imread(path)


def render(p: Pyramid, level, col, row):
    """生成一张瓦片（灰度 ndarray）。"""
    s, wl, hl = p.level(level)
    tx, ty = col * TILE, row * TILE
    if col < 0 or row < 0 or tx >= wl or ty >= hl:
        raise TileMissing(f'第 {level} 层没有瓦片 {col}_{row}')
    a0, a1 = max(tx - OVERLAP, 0), min(tx + TILE + OVERLAP, wl)
    b0, b1 = max(ty - OVERLAP, 0), min(ty + TILE + OVERLAP, hl)
    if s >= PREVIEW_STEP:
        pv = preview_path(p.trip)
        if not pv.exists():
            raise TileMissing(f'缺整车预览 {io.rel(pv)}，先跑 stitch')
        return _preview_level(str(pv), pv.stat().st_mtime_ns, s, wl, hl)[b0:b1, a0:a1]
    # 显示列 [a0*s, a1*s) ↔ 趟坐标 [x1 - a1*s, x1 - a0*s)；右缘可能多出不足 s 列，取到的是车外真实像素
    im = reader(p.trip).region(p.x1 - a1 * s, p.x1 - a0 * s, b0 * s, b1 * s, step=s)
    return im[:, ::-1]


def tile_bytes(trip, level, col, row):
    """瓦片 JPEG 字节；先查磁盘缓存。"""
    p = pyramid(trip)
    path = p.cache_dir() / str(level) / f'{col}_{row}.jpg'
    if path.exists():
        return path.read_bytes()
    im = render(p, level, col, row)
    ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(im), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if not ok:
        raise IOError(f'瓦片编码失败: {trip} {level}/{col}_{row}')
    data = buf.tobytes()
    _write_atomic(path, data)
    return data


def _write_atomic(path, data):
    """并发请求同一瓦片时，读方不会读到写了一半的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{threading.get_ident()}.tmp')
    tmp.write_bytes(data)
    try:
        os.replace(tmp, path)
    except OSError:  # Windows 上目标正被别的线程读时会拒绝替换；缓存已由对方写好，丢掉自己的
        tmp.unlink(missing_ok=True)


def overview_bytes(trip, w, h):
    """整车缩略条（导航条用），已翻转。"""
    pv = preview_path(trip)
    if not pv.exists():
        raise TileMissing(f'缺整车预览 {io.rel(pv)}，先跑 stitch')
    return _overview(str(pv), pv.stat().st_mtime_ns, w, h)


@functools.lru_cache(maxsize=16)
def _overview(path, mtime, w, h):
    with _preview_lock:
        base = _preview(path, mtime)
    im = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()

