"""读写工具：非 ASCII 路径的图片读写、带版本信息的 JSON、项目路径。"""
import datetime
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get('RAILSIDE_ROOT') or Path(__file__).resolve().parents[2])
DATA = ROOT / '数据'
MID = DATA / '中间'
OUT = DATA / '成品'
JPEG_MAX = 65500


def imread(path, flags=cv2.IMREAD_GRAYSCALE):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if im is None:
        raise IOError(f'读不了: {path}')
    return im


def imwrite(path, im, quality=92):
    """写图片；JPG 超过 65500 px 时自动改存 PNG。返回实际路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in ('.jpg', '.jpeg') and max(im.shape[:2]) > JPEG_MAX:
        path, ext = path.with_suffix('.png'), '.png'
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(im), params)
    if not ok:
        raise IOError(f'编码失败: {path}')
    buf.tofile(str(path))
    return path


@functools.lru_cache(maxsize=1)
def code_ver():
    """当前代码的 git commit；工作区有未提交改动时加 -dirty。"""
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '源码'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return h + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, obj):
    """写 JSON，自动带 code_ver、created_at。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {**obj, 'code_ver': code_ver(), 'created_at': now()}
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_jsonable), encoding='utf-8')
    os.replace(tmp, path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(type(o))


def emit(summary):
    """结束时在 stdout 打一行 JSON 摘要。"""
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print(json.dumps(summary, ensure_ascii=False, default=_jsonable), flush=True)


def rel(path):
    """项目内路径转成相对项目根的 posix 字符串，项目外的原样返回。"""
    p = Path(os.path.abspath(path))  # 不解析链接：worktree 里的 数据/ 是 junction
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def trip_dir(trip):
    return MID / trip
