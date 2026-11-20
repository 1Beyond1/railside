"""S6a 分割：从模板趟标注造 YOLO-seg 数据集并训练（train），整趟滑窗推理出 seg.json（segment）。

多尺度、一个模型：每个类别有缩小倍数 scale（小件 1，大件 4）。scale=s 的窗口是把趟图缩小 s 倍后切的
imgsz×imgsz 块，覆盖趟坐标 (imgsz·s)² 的范围；横纵都滑、重叠 25%。
窗口文件名 <趟>_s<s>_x<x0 七位>_y<y0 四位>.jpg；窗口内像素 (px, py) 对应趟坐标 (wx0 + s·px, wy0 + s·py)。
"""
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import io
from .coords import H
from .frames import Trip
from .tiles import tile_origins

TRAIN_TRIPS = ('T01',)  # T02 只做评测，不进训练（DECISIONS.md）
NEG_RATIO = 0.1
DEFAULT_SCALE = {'BOLT': 1}  # 主数据/零件类型.csv 没有 scale 时用；不在表里的类型按大件
BIG_SCALE = 4
MERGE_IOMIN = 0.7


# ---------- 坐标 ----------

def win_to_trip(px, py, wx0, wy0, s=1):
    return np.asarray(px) * s + wx0, np.asarray(py) * s + wy0


def trip_to_win(x, y, wx0, wy0, s=1):
    return (np.asarray(x) - wx0) / s, (np.asarray(y) - wy0) / s


def window_grid(x0, x1, size, s=1):
    """scale=s 时覆盖 [x0, x1) × [0, H) 的窗口左上角（趟坐标）：(xs, ys)，重叠 25%。ys 是 s 的整数倍。"""
    span = size * s
    xs = tile_origins(x0, x1, span, span // 4)
    ys = [s * y for y in tile_origins(0, H // s, size, size // 4)]
    return xs, ys


def win_name(trip, s, wx0, wy0):
    return f'{trip}_s{s}_x{wx0:07d}_y{wy0:04d}.jpg'


def parse_win_name(name):
    stem = name.rsplit('.', 1)[0]
    trip, rest = stem.rsplit('_s', 1)
    s, rest = rest.split('_x')
    xs, ys = rest.split('_y')
    return trip, int(s), int(xs), int(ys)


def read_strip(t, wx0, size, s):
    """趟坐标 [wx0, wx0 + size·s) 的整高条带，缩小 s 倍。"""
    return t.region(wx0, wx0 + size * s, 0, H, step=s)


def crop(strip, wy0, size, s):
    """从缩小后的条带取一个窗口；不够高时底部补 0。"""
    c = strip[wy0 // s: wy0 // s + size]
    if c.shape[0] < size:
        c = np.pad(c, ((0, size - c.shape[0]), (0, 0)))
    return np.ascontiguousarray(c)


def rel(path):
    """相对项目根的路径；不解析 junction（数据/ 可能是指向别处的链接）。"""
    try:
        return Path(path).relative_to(io.ROOT).as_posix()
    except ValueError:
        return io.rel(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------- 类别表 ----------

def type_table():
    """主数据/零件类型.csv → {type: scale}（按文件行序）；没有该文件返回 None。"""
    p = io.ROOT / '主数据' / '零件类型.csv'
    if not p.exists():
        return None
    with open(p, encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('type')]
    return {r['type']: int(r['scale']) if (r.get('scale') or '').strip() else default_scale(r['type']) for r in rows}


def default_scale(t):
    return DEFAULT_SCALE.get(t, BIG_SCALE)


def scales_for(names):
    tab = type_table() or {}
    return [tab.get(n, default_scale(n)) for n in names]


def class_names(parts):
    """类别表：主数据/零件类型.csv 的 type 列顺序；没有该文件时用标注里出现的 type 排序。"""
    used = sorted({q['type'] for q in parts})
    tab = type_table()
    if tab is None:
        print(f'提示：没有 主数据/零件类型.csv，类别表用标注里出现的 type 排序：{used}；'
              f'scale 用默认表（{DEFAULT_SCALE}，其余 {BIG_SCALE}）', file=sys.stderr)
        return used
    unknown = [t for t in used if t not in tab]
    if unknown:
        raise SystemExit(f'标注里的类型不在 主数据/零件类型.csv：{unknown}')
    return list(tab)


# ---------- 数据集 ----------

def poly_to_labels(poly, wx0, wy0, size, s=1):
    """趟坐标多边形裁到 scale=s 的窗口内，返回归一化轮廓列表（每个是 (n, 2) 的 [0,1] 数组）。"""
    pts = np.asarray(trip_to_win(*np.asarray(poly, float).T, wx0, wy0, s)).T
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        c = c[:, 0, :].astype(float)
        if len(c) >= 3 and cv2.contourArea(c.astype(np.float32)) >= 4:
            out.append(np.clip(c / size, 0, 1))
    return out


def build_dataset(labels_path, trip, size, out, seed=0):
    """造 YOLO-seg 数据集到 out/。每个标注只进它所属尺度的窗口。返回统计 dict。"""
    lab = io.read_json(labels_path)
    if lab.get('trip', trip) != trip:
        raise SystemExit(f'标注是 {lab.get("trip")} 的，不是 {trip}')
    parts = [q for q in lab.get('parts', []) if len(q.get('poly') or []) >= 3]
    if not parts:
        raise SystemExit(f'{labels_path} 里没有可用的多边形')
    names = class_names(parts)
    scales = scales_for(names)
    cid = {t: i for i, t in enumerate(names)}
    m = io.read_json(io.trip_dir(trip) / 'manifest.json')
    vx0, vx1 = m['train_panels'][-1]['x0'], m['train_panels'][-1]['x1']
    rng = np.random.default_rng(seed)
    windows = {'train': [], 'val': []}  # (s, wx0, wy0, [(cls, 轮廓)])
    n_neg = {'train': 0, 'val': 0}

    for s in sorted({scales[cid[q['type']]] for q in parts}):
        span = size * s
        ps = [q for q in parts if scales[cid[q['type']]] == s]
        xs, ys = (np.asarray(a) for a in window_grid(m['train_x0'], m['train_x1'], size, s))
        hits = {}  # 每个多边形按外框落到哪些窗口
        for k, q in enumerate(ps):
            p = np.asarray(q['poly'], float)
            (bx0, by0), (bx1, by1) = p.min(0), p.max(0)
            for i in np.nonzero((xs < bx1) & (xs + span > bx0))[0]:
                for j in np.nonzero((ys < by1) & (ys + span > by0))[0]:
                    hits.setdefault((i, j), []).append(k)

        def split_of(wx0):
            if wx0 >= vx0 and wx0 + span <= vx1:
                return 'val'
            if wx0 + span <= vx0 or wx0 >= vx1:
                return 'train'
            return None  # 跨过验证车边界的窗口不用，避免训练、验证共用像素

        pos = {'train': [], 'val': []}
        for (i, j), ks in sorted(hits.items()):
            sp = split_of(int(xs[i]))
            if sp is None:
                continue
            wx0, wy0 = int(xs[i]), int(ys[j])
            objs = [(cid[ps[k]['type']], c) for k in ks for c in poly_to_labels(ps[k]['poly'], wx0, wy0, size, s)]
            if objs:
                pos[sp].append((s, wx0, wy0, objs))
        for sp, ws in pos.items():
            cand = [(int(xs[i]), int(ys[j])) for i in range(len(xs)) for j in range(len(ys))
                    if (i, j) not in hits and split_of(int(xs[i])) == sp]
            n = min(len(cand), max(1, round(NEG_RATIO * len(ws)))) if ws else 0
            pick = sorted(rng.choice(len(cand), n, replace=False)) if n else []
            windows[sp] += ws + [(s, *cand[p], []) for p in pick]
            n_neg[sp] += n

    if out.exists():
        shutil.rmtree(out)
    t = Trip.from_id(trip)
    stats = {'trip': trip, 'imgsz': size, 'names': names, 'scales': dict(zip(names, scales)),
             'val_x': [vx0, vx1], 'labels': rel(labels_path), 'labels_sha256': sha256(labels_path)}
    for sp, ws in windows.items():
        (out / 'images' / sp).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / sp).mkdir(parents=True, exist_ok=True)
        inst = [0] * len(names)
        per_scale = {}
        ws.sort(key=lambda w: w[:3])
        key, strip = None, None
        for s, wx0, wy0, objs in ws:
            if (s, wx0) != key:
                key, strip = (s, wx0), read_strip(t, wx0, size, s)
            name = win_name(trip, s, wx0, wy0)
            io.imwrite(out / 'images' / sp / name, crop(strip, wy0, size, s), quality=95)
            lines = []
            for c, pts in objs:
                inst[c] += 1
                lines.append(f'{c} ' + ' '.join(f'{v:.6f}' for v in pts.ravel()))
            (out / 'labels' / sp / name).with_suffix('.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            per_scale[s] = per_scale.get(s, 0) + 1
        stats[sp] = {'images': len(ws), 'background': n_neg[sp], 'images_per_scale': per_scale,
                     'instances': dict(zip(names, inst))}
    if not windows['val']:
        print('提示：验证车（最后一节）里没有标注，验证集为空', file=sys.stderr)

    from ultralytics.utils import YAML
    # 不写 path：ultralytics 按 data.yaml 所在目录找，整个目录拷走也能用
    YAML.save(out / 'data.yaml', {'train': 'images/train', 'val': 'images/val', 'names': dict(enumerate(names))})
    io.write_json(out / 'stats.json', stats)
    return stats


# ---------- 训练 ----------

def pick_device(device):
    if device != 'auto':
        return device
    import torch
    return '0' if torch.cuda.is_available() else 'cpu'


def resolve_weights(model, models_dir):
    """已有路径直接用；否则下载到 models/ 下（不落在工作根）。"""
    p = Path(model)
    if p.exists():
        return p
    if p.parent != Path('.'):
        raise SystemExit(f'权重不存在：{model}')
    dst = models_dir / p.name
    if not dst.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        models_dir.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(dst)
    if not dst.exists():
        raise SystemExit(f'下载预训练权重失败：{p.name}')
    return dst


def log_provenance(models_dir, args, stats, metrics, best, device):
    import ultralytics
    md = models_dir / '出处.md'
    if not md.exists():
        md.write_text('# 权重出处\n\n每份权重一节：训练命令、数据、指标。由 `railside train` 追加。\n', encoding='utf-8')
    cmd = 'uv run railside ' + ' '.join(f'"{a}"' if ' ' in a else a for a in sys.argv[1:])
    sp = {k: stats[k] for k in ('train', 'val')}
    sec = [
        f'\n## {args.name}（{io.now()}）\n',
        f'- 命令：`{cmd}`',
        f'- 代码：{io.code_ver()}；ultralytics {ultralytics.__version__}；设备 {device}',
        f'- 标注：`{stats["labels"]}`，sha256 `{stats["labels_sha256"]}`',
        f'- 类别与缩小倍数：{", ".join(f"{n}×{s}" for n, s in stats["scales"].items())}',
        f'- 数据集：`{rel(io.MID / "数据集" / args.name)}`，imgsz {stats["imgsz"]}，重叠 25%；'
        f'验证集 = 最后一节车 x∈[{stats["val_x"][0]}, {stats["val_x"][1]})',
        *(f'  - {"训练" if k == "train" else "验证"}：{v["images"]} 张（其中无标注 {v["background"]}；'
          f'按 scale {v["images_per_scale"]}），实例 {v["instances"]}' for k, v in sp.items()),
        f'- 验证集（mask）：mAP50 {metrics["map50"]:.4f}，mAP50-95 {metrics["map"]:.4f}',
        f'- 权重：`{io.rel(best)}`，sha256 `{sha256(best)}`',
    ]
    with open(md, 'a', encoding='utf-8') as f:
        f.write('\n'.join(sec) + '\n')
    return md


def run_train(args):
    if args.trip not in TRAIN_TRIPS:
        raise SystemExit(f'{args.trip} 只做评测，不能用来训练（只允许 {", ".join(TRAIN_TRIPS)}）')
    labels = Path(args.labels)
    labels = labels if labels.is_absolute() else io.ROOT / labels
    if not labels.exists():
        raise SystemExit(f'没有标注文件：{rel(labels)}（先在 标注/模板T01/零件.json 标好，格式见 设计.md §4.3）')
    if not (io.trip_dir(args.trip) / 'manifest.json').exists():
        raise SystemExit(f'没有 {args.trip} 的 manifest.json，先跑 railside stitch {args.trip}')
    out = io.MID / '数据集' / args.name
    t0 = time.time()
    stats = build_dataset(labels, args.trip, args.imgsz, out)
    summary = {'step': 'train', 'trip': args.trip, 'dataset': rel(out), 'scales': stats['scales'],
               'train_images': stats['train']['images'], 'val_images': stats['val']['images'],
               'instances': {sp: sum(stats[sp]['instances'].values()) for sp in ('train', 'val')}}
    if args.export_only:
        return {**summary, 'export_only': True, 'seconds': round(time.time() - t0, 1)}

    from ultralytics import YOLO
    models_dir = io.ROOT / 'models'
    device = pick_device(args.device)
    model = YOLO(str(resolve_weights(args.model, models_dir)))
    res = model.train(data=str(out / 'data.yaml'), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                      device=device, fraction=args.fraction, project=str(models_dir), name=args.name,
                      exist_ok=True, seed=0)
    best = Path(model.trainer.best)
    # 类别→缩小倍数跟权重放一起：推理按它切窗，不受之后改 零件类型.csv 影响
    (best.parent / 'scales.json').write_text(json.dumps(stats['scales'], ensure_ascii=False), encoding='utf-8')
    # 训练结束时 ultralytics 用 best.pt 在验证集上重算一次，返回的就是这次的指标
    rd = res if isinstance(res, dict) else res.results_dict
    metrics = {'map50': float(rd['metrics/mAP50(M)']), 'map': float(rd['metrics/mAP50-95(M)'])}
    md = log_provenance(models_dir, args, stats, metrics, best, device)
    return {**summary, 'device': device, 'weights': io.rel(best), 'mask_map50': round(metrics['map50'], 4),
            'mask_map50_95': round(metrics['map'], 4), 'provenance': io.rel(md), 'seconds': round(time.time() - t0, 1)}


# ---------- 推理 ----------

def mask_to_poly(mask):
    """二值掩膜取最大外轮廓，返回 (n, 2) int 数组；没有轮廓返回 None。"""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :]
    return c if len(c) >= 3 else None


def merge_dets(dets, thr=MERGE_IOMIN):
    """跨窗口、跨尺度去重：同类且交集占小框面积 > thr 的合并，留高分那份。返回保留的子列表（按 x、y 排）。"""
    keep = []
    for c in sorted({d['cls'] for d in dets}):
        ds = sorted((d for d in dets if d['cls'] == c), key=lambda d: -d['score'])
        b = np.array([d['bbox_xy'] for d in ds], float)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        alive = np.ones(len(ds), bool)
        for i in range(len(ds)):
            if not alive[i]:
                continue
            keep.append(ds[i])
            iw = np.clip(np.minimum(b[i, 2], b[:, 2]) - np.maximum(b[i, 0], b[:, 0]), 0, None)
            ih = np.clip(np.minimum(b[i, 3], b[:, 3]) - np.maximum(b[i, 1], b[:, 1]), 0, None)
            iomin = iw * ih / np.maximum(np.minimum(area[i], area), 1e-9)
            alive &= ~(iomin > thr)
    return sorted(keep, key=lambda d: (d['bbox_xy'][0], d['bbox_xy'][1]))


def run_segment(args):
    weights = Path(args.weights)
    weights = weights if weights.is_absolute() else io.ROOT / weights
    if not weights.exists():
        raise SystemExit(f'权重不存在：{args.weights}（先跑 railside train，或从 GitHub Release 下载到 models/）')
    mp = io.trip_dir(args.trip) / 'manifest.json'
    if not mp.exists():
        raise SystemExit(f'没有 {args.trip} 的 manifest.json，先跑 railside stitch {args.trip}')
    m = io.read_json(mp)
    x0 = m['train_x0'] if args.x0 is None else args.x0
    x1 = m['train_x1'] if args.x1 is None else args.x1
    full = (x0, x1) == (m['train_x0'], m['train_x1'])
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else io.ROOT / args.out
    else:  # 只跑了一段的结果不冒充整趟的 seg.json
        out = io.trip_dir(args.trip) / 'seg.json' if full else io.MID / '_smoke' / args.trip / 'seg.json'
    size = args.imgsz
    from ultralytics import YOLO
    model = YOLO(str(weights))
    names = [model.names[k] for k in sorted(model.names)]
    sj = weights.parent / 'scales.json'
    scales = [json.loads(sj.read_text(encoding='utf-8'))[n] for n in names] if sj.exists() else scales_for(names)
    device = None if args.device == 'auto' else args.device
    t = Trip.from_id(args.trip)
    t0 = time.time()
    raw, passes = [], {}
    for s in sorted(set(scales)):
        cls_ids = [k for k, v in enumerate(scales) if v == s]
        xs, ys = window_grid(x0, x1, size, s)
        ts, ti = time.time(), 0.0
        for wx0 in xs:
            strip = read_strip(t, wx0, size, s)
            crops = [cv2.cvtColor(crop(strip, wy0, size, s), cv2.COLOR_GRAY2BGR) for wy0 in ys]
            tp = time.time()
            res = model.predict(crops, imgsz=size, conf=args.conf, classes=cls_ids, retina_masks=True,
                                device=device, verbose=False)
            ti += time.time() - tp
            for wy0, r in zip(ys, res):
                if r.masks is None:
                    continue
                ms = r.masks.data.cpu().numpy() > 0.5
                for mk, c, sc in zip(ms, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    poly = mask_to_poly(mk)
                    if poly is None:
                        continue
                    px, py = win_to_trip(poly[:, 0], poly[:, 1], wx0, wy0, s)
                    raw.append({'cls': names[int(c)], 'scale': s, 'score': round(float(sc), 4),
                                'poly_xy': np.stack([px, py], 1).astype(int).tolist(),
                                'bbox_xy': [int(px.min()), int(py.min()), int(px.max()) + s, int(py.max()) + s]})
        n = len(xs) * len(ys)
        passes[s] = {'windows': n, 'seconds': round(time.time() - ts, 1),
                     'ms_per_window': round(1000 * (time.time() - ts) / n, 1), 'infer_ms_per_window': round(1000 * ti / n, 1)}
    dets = merge_dets(raw)
    io.write_json(out, {'trip': args.trip, 'model': {'name': weights.name, 'path': io.rel(weights), 'sha256': sha256(weights),
                                                     'names': names, 'scales': dict(zip(names, scales))},
                        'imgsz': size, 'conf': args.conf, 'x0': x0, 'x1': x1, 'full_range': full,
                        'passes': passes, 'dets': dets})
    return {'step': 'segment', 'trip': args.trip, 'x0': x0, 'x1': x1, 'passes': passes, 'dets_raw': len(raw),
            'dets': len(dets), 'seconds': round(time.time() - t0, 1), 'out': rel(out)}


# ---------- 命令行 ----------

def add_parser(sub, step):
    if step == 'train':
        p = sub.add_parser(step, help='S6a 造 YOLO-seg 数据集并训练（只用 T01）')
        p.add_argument('--labels', default='标注/模板T01/零件.json')
        p.add_argument('--trip', default='T01')
        p.add_argument('--model', default='yolo11s-seg.pt', help='预训练权重：路径，或文件名（下载到 models/）')
        p.add_argument('--imgsz', type=int, default=1024)
        p.add_argument('--epochs', type=int, default=100)
        p.add_argument('--batch', type=int, default=-1, help='-1 = 按显存自动（CPU 上 ultralytics 退回 16）')
        p.add_argument('--device', default='auto', help='auto：有 CUDA 用 0，否则 cpu')
        p.add_argument('--fraction', type=float, default=1.0)
        p.add_argument('--name', default='r1')
        p.add_argument('--export-only', action='store_true', help='只造数据集，不训练')
        p.set_defaults(func=run_train)
    else:
        p = sub.add_parser(step, help='S6a 整趟多尺度滑窗分割，写 seg.json')
        p.add_argument('trip')
        p.add_argument('--weights', required=True)
        p.add_argument('--imgsz', type=int, default=1024)
        p.add_argument('--conf', type=float, default=0.25)
        p.add_argument('--x0', type=int, default=None)
        p.add_argument('--x1', type=int, default=None)
        p.add_argument('--out', default=None,
                       help='默认：整趟写 数据/中间/<趟>/seg.json；只跑一段（--x0/--x1）写 数据/中间/_smoke/<趟>/seg.json')
        p.add_argument('--device', default='auto', help='auto：ultralytics 自动选（有 CUDA 用 CUDA）')
        p.set_defaults(func=run_segment)


def run(args):
    return args.func(args)
