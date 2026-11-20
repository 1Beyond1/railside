"""标注/训练/推理用的切片。

切片按趟坐标（时间顺序、不翻转）从车头到车尾铺，覆盖整个高度。
文件名 <趟>_x<x0 七位>.jpg；切片内像素 (px, py) 对应趟坐标 (x0 + px, py)。
"""
from . import io
from .coords import H
from .frames import Trip

SIZE = 4096
OVERLAP = 512


def tile_origins(x0, x1, size=SIZE, overlap=OVERLAP):
    """覆盖 [x0, x1) 的切片左端；最后一片右端对齐 x1。"""
    stride = size - overlap
    xs = list(range(x0, max(x0 + 1, x1 - size + 1), stride))
    if not xs or xs[-1] + size < x1:
        xs.append(max(x0, x1 - size))
    return xs


def tile_name(trip, x0):
    return f'{trip}_x{x0:07d}.jpg'


def parse_tile_name(name):
    stem = name.rsplit('.', 1)[0]
    trip, xs = stem.rsplit('_x', 1)
    return trip, int(xs)


def add_parser(sub, step):
    p = sub.add_parser(step, help='导出切片（标注、训练、推理用）')
    p.add_argument('trip')
    p.add_argument('--size', type=int, default=SIZE)
    p.add_argument('--overlap', type=int, default=OVERLAP)
    p.add_argument('--out', default=None, help='默认 数据/中间/<趟>/切片')
    p.set_defaults(func=run)


def run(args):
    m = io.read_json(io.trip_dir(args.trip) / 'manifest.json')
    t = Trip.from_id(args.trip)
    out = io.trip_dir(args.trip) / '切片' if args.out is None else io.ROOT / args.out
    xs = tile_origins(m['train_x0'], m['train_x1'], args.size, args.overlap)
    for x0 in xs:
        io.imwrite(out / tile_name(args.trip, x0), t.region(x0, x0 + args.size, 0, H), quality=95)
    return {'step': 'tiles', 'trip': args.trip, 'n': len(xs), 'size': args.size, 'overlap': args.overlap, 'out': io.rel(out)}
