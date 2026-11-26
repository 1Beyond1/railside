"""一键全合成脱敏 Demo：`railside demo`。

在几秒内生成 100% 纯程序合成、零客户隐私的 3 节编组地铁侧面线阵长图（T01 模板趟 + T02 巡检趟）、
模板 v1、对齐/识别/比对结果（包含缺失、转角、位移、异物 4 类真实由算法检出的告警）、SQLite 库与 HTML 报告。

默认输出到已 .gitignore 的 `数据/` 目录，也可通过 `--out` / `--data-dir` 指定到任意目录，绝不写入 Git 跟踪目录。
"""
import contextlib
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import align, compare, db, io, match, qc, report, stitch, template
from .coords import H, W
from .eval import synth
from .eval.frameset import FrameCopy
from .frames import Trip

SEED = 20261122
N_CARS = 3
CAR_W = 6000
HEAD_X = 2000
N_FRAMES = 12              # 1 帧空场 + 9 帧 3 节车厢 (18000 px) + 2 帧空场 = 24000 px
TOTAL_W = N_FRAMES * W

T02_DX = 120               # T02 相对 T01 的整体横向偏移（px）
T02_DY = 8                 # T02 相对 T01 的整体纵向偏移（px）
SX = 0.50                  # 合成标定比例（mm/px）
SY = 0.50

DEMO_TRIPS = ('T01', 'T02')
TEMPLATE_VER = 'v1'
MARKER = '.railside_demo.json'


def add_parser(sub, step):
    p = sub.add_parser(step, help='一键生成纯合成脱敏演示数据（两趟 + 模板 v1 + 4 类告警）')
    p.add_argument('--out', '--data-dir', dest='data_dir', default=None, metavar='目录',
                   help='输出数据目录；默认写入已忽略的 数据/ 目录')
    p.add_argument('--force', '-f', action='store_true', help='目标目录已存在演示/同趟数据时强制覆盖重建')
    p.add_argument('--seed', type=int, default=SEED, help=f'随机种子（默认 {SEED}）')
    p.set_defaults(func=run)


@contextlib.contextmanager
def _use_data_dir(data_dir: Path):
    """临时把 io.DATA / io.MID / io.OUT 指向目标目录，不修改 io.ROOT（仍可读 主数据/零件类型.csv）。"""
    old_data, old_mid, old_out = io.DATA, io.MID, io.OUT
    io.DATA = data_dir
    io.MID = data_dir / '中间'
    io.OUT = data_dir / '成品'
    try:
        yield
    finally:
        io.DATA, io.MID, io.OUT = old_data, old_mid, old_out


def _demo_targets(data_dir: Path):
    return [
        data_dir / '原始' / 'T01',
        data_dir / '原始' / 'T02',
        data_dir / '中间' / 'T01',
        data_dir / '中间' / 'T02',
        data_dir / '成品' / '模板' / TEMPLATE_VER,
        data_dir / '成品' / '报告' / 'T02',
    ]


def _demo_intact(data_dir: Path):
    required = [
        data_dir / MARKER,
        data_dir / '中间' / 'T01' / 'manifest.json',
        data_dir / '中间' / 'T02' / 'manifest.json',
        data_dir / '中间' / 'T02' / 'parts.json',
        data_dir / '中间' / 'T02' / 'compare.json',
        data_dir / '成品' / '模板' / TEMPLATE_VER / 'template.json',
        data_dir / '成品' / '库.sqlite',
    ]
    return all(p.is_file() for p in required)


def _prepare_dir(data_dir: Path, force: bool):
    """检查目标目录状态：
    - 无冲突：返回 'create'
    - 已存在完整演示数据且未传 --force：返回 'exists'（幂等跳过）
    - 已存在非演示数据且未传 --force：安全拒绝并退出
    - 传入 --force：清理旧演示产物后返回 'overwrite'
    """
    existing = [p for p in _demo_targets(data_dir) if p.exists()]
    marker = data_dir / MARKER
    if not existing and not marker.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return 'create'
    if not force:
        if _demo_intact(data_dir):
            return 'exists'
        shown = '、'.join(io.rel(p) for p in existing[:3])
        raise SystemExit(
            f'目标目录已存在数据（{shown}），未启用 --force 拒绝覆盖；'
            '请改用 --out 指定新目录，或确认后加 --force 重新生成。'
        )
    for p in existing:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    marker.unlink(missing_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return 'overwrite'


# ------------------------------------------------------------------ 合成长图与零件绘制

def _rect_poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _circle_poly(cx, cy, r, n=24):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [[round(cx + r * float(np.cos(a)), 1), round(cy + r * float(np.sin(a)), 1)] for a in ang]


def _hex_poly(cx, cy, r):
    return _circle_poly(cx, cy, r, n=6)


def _draw_anchor_plaque(canvas, x0, y0, size, rng, label):
    """在车体下部绘制高对比度、唯一的几何纹理定位靶块（供 S5 align 精配）。"""
    patch = canvas[y0:y0 + size, x0:x0 + size]
    patch[:] = 95
    # 独特伪随机几何方格 + 斜线 + 圆环，保证每个靶的自相关次峰极低
    cell = 32
    grid = rng.choice([45, 115, 195], size=(size // cell, size // cell))
    tex = cv2.resize(grid.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    patch[:] = cv2.GaussianBlur(tex, (5, 5), 1.0)
    cv2.rectangle(patch, (10, 10), (size - 11, size - 11), 220, 4)
    cv2.rectangle(patch, (26, 26), (size - 27, size - 27), 35, 3)
    cx, cy = size // 2, size // 2
    cv2.circle(patch, (cx, cy), 120, 235, 6)
    cv2.line(patch, (40, cy), (size - 40, cy), 25, 5)
    cv2.line(patch, (cx, 40), (cx, size - 40), 25, 5)
    cv2.putText(patch, label, (52, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 240, 4, cv2.LINE_AA)


def _draw_axle_end(canvas, x0, y0, x1, y1):
    """轴端装置：厚实同心圆法兰 + 右侧非对称传感器凸耳 + 非对称刻槽与螺栓。
    同心圆法兰保证 ±12° 旋转时未旋转 NCC 仍高于 match 门槛，非对称凸耳与刻槽确保 compare 精准测出 Δθ。
    """
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = min(x1 - x0, y1 - y0) // 2 - 6
    cv2.circle(canvas, (cx, cy), r, 32, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r - 5, 205, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r - 24, 65, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r - 38, 172, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r - 66, 98, -1, cv2.LINE_AA)
    # 强非对称特征：右侧矩形传感器凸耳 + 粗斜向定位槽
    cv2.rectangle(canvas, (cx + 16, cy - 22), (cx + r - 18, cy + 22), 235, -1)
    cv2.rectangle(canvas, (cx + 16, cy - 22), (cx + r - 18, cy + 22), 28, 3)
    cv2.line(canvas, (cx - int(r * 0.56), cy - int(r * 0.34)),
             (cx + int(r * 0.56), cy + int(r * 0.34)), 24, 14, cv2.LINE_AA)
    # 3 颗非对称分布螺栓头
    for deg in (25, 145, 255):
        rad = np.radians(deg)
        bx = int(round(cx + (r - 30) * np.cos(rad)))
        by = int(round(cy + (r - 30) * np.sin(rad)))
        cv2.circle(canvas, (bx, by), 10, 26, -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 6, 232, -1, cv2.LINE_AA)


def _draw_spring1(canvas, x0, y0, x1, y1):
    """一系螺旋弹簧：上下弹簧座 + 5 圈带明暗层次的斜向钢丝。"""
    cv2.rectangle(canvas, (x0 + 8, y0 + 6), (x1 - 8, y0 + 44), 168, -1)
    cv2.rectangle(canvas, (x0 + 8, y0 + 6), (x1 - 8, y0 + 44), 35, 3)
    cv2.rectangle(canvas, (x0 + 8, y1 - 44), (x1 - 8, y1 - 6), 168, -1)
    cv2.rectangle(canvas, (x0 + 8, y1 - 44), (x1 - 8, y1 - 6), 35, 3)
    y_top, y_bot = y0 + 56, y1 - 56
    n_coils = 5
    step = (y_bot - y_top) / n_coils
    for i in range(n_coils):
        ya = int(y_top + i * step + 18)
        yb = int(y_top + (i + 1) * step - 6)
        cv2.line(canvas, (x0 + 28, yb), (x1 - 28, ya), 28, 34, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 28, yb), (x1 - 28, ya), 186, 22, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 34, yb - 4), (x1 - 34, ya - 4), 232, 7, cv2.LINE_AA)


def _draw_airspring(canvas, x0, y0, x1, y1):
    """空气弹簧气囊：上下金属压盘 + 橡胶波纹气囊。"""
    cv2.rectangle(canvas, (x0 + 16, y0 + 8), (x1 - 16, y0 + 48), 180, -1)
    cv2.rectangle(canvas, (x0 + 16, y1 - 48), (x1 - 16, y1 - 8), 180, -1)
    mid_y = (y0 + y1) // 2
    for yc in (y0 + 108, mid_y, y1 - 108):
        cv2.ellipse(canvas, ((x0 + x1) // 2, yc), ((x1 - x0) // 2 - 22, 46), 0, 0, 360, 72, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, ((x0 + x1) // 2, yc), ((x1 - x0) // 2 - 22, 46), 0, 0, 360, 155, 4, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 48, yc - 14), (x1 - 48, yc - 14), 192, 6, cv2.LINE_AA)


def _draw_damper(canvas, x0, y0, x1, y1):
    """垂向减振器：上下吊耳 + 上部防尘筒（带条纹）+ 下部油缸。"""
    cx = (x0 + x1) // 2
    w = x1 - x0
    ym = y0 + int((y1 - y0) * 0.54)
    # 上下安装环
    for cy in (y0 + 36, y1 - 36):
        cv2.circle(canvas, (cx, cy), 30, 175, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 30, 30, 3, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 14, 42, -1, cv2.LINE_AA)
    # 上防尘套筒
    cv2.rectangle(canvas, (cx - w // 3, y0 + 62), (cx + w // 3, ym), 192, -1)
    cv2.rectangle(canvas, (cx - w // 3, y0 + 62), (cx + w // 3, ym), 32, 3)
    for yy in range(y0 + 88, ym - 16, 36):
        cv2.line(canvas, (cx - w // 3 + 6, yy), (cx + w // 3 - 6, yy + 14), 52, 8, cv2.LINE_AA)
    # 下工作缸体 + 纵高光
    cv2.rectangle(canvas, (cx - w // 4, ym), (cx + w // 4, y1 - 62), 124, -1)
    cv2.rectangle(canvas, (cx - w // 4, ym), (cx + w // 4, y1 - 62), 28, 3)
    cv2.line(canvas, (cx - 10, ym + 10), (cx - 10, y1 - 70), 215, 8)


def _draw_link(canvas, x0, y0, x1, y1):
    """拉杆/连杆：两端销轴耳环 + 粗连杆 + 中部调节螺套。"""
    cy = (y0 + y1) // 2
    cv2.rectangle(canvas, (x0 + 52, cy - 26), (x1 - 52, cy + 26), 165, -1)
    cv2.rectangle(canvas, (x0 + 52, cy - 26), (x1 - 52, cy + 26), 32, 3)
    cv2.line(canvas, (x0 + 60, cy - 10), (x1 - 60, cy - 10), 220, 6)
    cx = (x0 + x1) // 2
    cv2.rectangle(canvas, (cx - 48, cy - 38), (cx + 48, cy + 38), 195, -1)
    cv2.rectangle(canvas, (cx - 48, cy - 38), (cx + 48, cy + 38), 28, 3)
    cv2.line(canvas, (cx, cy - 34), (cx, cy + 34), 40, 5)
    for ex in (x0 + 48, x1 - 48):
        cv2.circle(canvas, (ex, cy), 42, 182, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ex, cy), 42, 30, 3, cv2.LINE_AA)
        cv2.circle(canvas, (ex, cy), 18, 42, -1, cv2.LINE_AA)


def _draw_bolt(canvas, x0, y0, x1, y1):
    """六角螺栓头：圆垫片 + 六角头 + 醒目防松划线。"""
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = max(16, min(x1 - x0, y1 - y0) // 2 - 3)
    cv2.circle(canvas, (cx, cy), r, 30, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r - 3, 148, -1, cv2.LINE_AA)
    pts = np.array(_hex_poly(cx, cy, max(8, r - 8)), dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 212)
    cv2.polylines(canvas, [pts], True, 26, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), max(4, r // 5), 54, -1, cv2.LINE_AA)
    # 醒目防松记号线（便于视觉与算法识别转角）
    cv2.line(canvas, (cx - r + 5, cy - 3), (cx + r - 5, cy + 3), 24, 4, cv2.LINE_AA)
    cv2.line(canvas, (cx - r + 7, cy - 3), (cx + r - 7, cy + 3), 245, 1, cv2.LINE_AA)


def _draw_equip(canvas, x0, y0, x1, y1, car):
    """车下设备箱：外框 + 散热百叶 + 检修门把手 + 铆钉。"""
    cv2.rectangle(canvas, (x0, y0), (x1, y1), 132, -1)
    cv2.rectangle(canvas, (x0 + 10, y0 + 10), (x1 - 10, y1 - 10), 156, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), 35, 6)
    mid_x = (x0 + x1) // 2
    cv2.line(canvas, (mid_x, y0 + 16), (mid_x, y1 - 16), 40, 5)
    # 左半：散热百叶
    for yy in range(y0 + 70, y1 - 70, 44):
        cv2.rectangle(canvas, (x0 + 50, yy), (mid_x - 45, yy + 16), 46, -1)
        cv2.line(canvas, (x0 + 50, yy + 18), (mid_x - 45, yy + 18), 208, 3)
    # 右半：检修门 + 门锁 + 铭牌文字
    cv2.rectangle(canvas, (mid_x + 40, y0 + 60), (x1 - 45, y1 - 60), 140, -1)
    cv2.rectangle(canvas, (mid_x + 40, y0 + 60), (x1 - 45, y1 - 60), 45, 4)
    cv2.rectangle(canvas, (x1 - 120, (y0 + y1) // 2 - 48), (x1 - 75, (y0 + y1) // 2 + 48), 52, -1)
    cv2.putText(canvas, f'INV-{car:02d}', (mid_x + 76, y0 + 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 225, 4, cv2.LINE_AA)


def _draw_tank(canvas, x0, y0, x1, y1):
    """风缸：带柱面明暗渐变的卧式圆柱罐体 + 两道固定抱箍。"""
    h = y1 - y0
    r = h // 2
    # 纵向圆柱光照渐变
    profile = np.clip(95 + 105 * np.sin(np.linspace(0.15, np.pi - 0.15, h)) ** 1.4, 45, 220).astype(np.uint8)
    mask = np.zeros((h, x1 - x0), np.uint8)
    cv2.rectangle(mask, (r, 0), (x1 - x0 - r, h), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (x1 - x0 - r, r), r, 255, -1)
    roi = canvas[y0:y1, x0:x1]
    roi[mask > 0] = np.broadcast_to(profile[:, None], roi.shape)[mask > 0]
    # 两道钢抱箍
    for sx in (x0 + int((x1 - x0) * 0.28), x0 + int((x1 - x0) * 0.72)):
        cv2.rectangle(canvas, (sx - 22, y0 + 6), (sx + 22, y1 - 6), 54, -1)
        cv2.line(canvas, (sx - 10, y0 + 12), (sx - 10, y1 - 12), 190, 5)
        cv2.circle(canvas, (sx, y0 + 36), 12, 220, -1)
        cv2.circle(canvas, (sx, y1 - 36), 12, 220, -1)


def _draw_mark(canvas, x0, y0, x1, y1, car):
    """车体标记/铭牌：高对比 T 形架车位标识 + 车厢编号。"""
    cv2.rectangle(canvas, (x0, y0), (x1, y1), 225, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), 32, 8)
    cv2.rectangle(canvas, (x0 + 18, y0 + 18), (x1 - 18, y1 - 18), 46, 3)
    # T 形标识
    tx = x0 + 115
    cv2.rectangle(canvas, (tx - 68, y0 + 58), (tx + 68, y0 + 96), 28, -1)
    cv2.rectangle(canvas, (tx - 20, y0 + 96), (tx + 20, y1 - 52), 28, -1)
    cv2.putText(canvas, f'M{car:02d}', (x0 + 235, y0 + 195), cv2.FONT_HERSHEY_SIMPLEX, 2.8, 28, 6, cv2.LINE_AA)


def _draw_bogie_chassis(canvas, bx0, by0, bx1, by1):
    """转向架构架与车轮结构背景（不含具体子件）。"""
    # 两对车轮（外圈轮辋 + 辐板 + 轴心）
    for wx in (bx0 + 350, bx0 + 1490):
        wy = by0 + 1040
        cv2.circle(canvas, (wx, wy), 330, 148, -1, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 330, 210, 10, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 268, 72, -1, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 190, 96, 8, cv2.LINE_AA)
    # 主纵梁与斜撑（铸钢深灰 + 倒角高光）
    cv2.rectangle(canvas, (bx0 + 40, by0 + 220), (bx1 - 40, by0 + 910), 62, -1)
    cv2.rectangle(canvas, (bx0 + 40, by0 + 220), (bx1 - 40, by0 + 910), 120, 6)
    cv2.rectangle(canvas, (bx0 + 120, by0 + 820), (bx1 - 120, by0 + 1220), 76, -1)
    cv2.rectangle(canvas, (bx0 + 120, by0 + 820), (bx1 - 120, by0 + 1220), 132, 5)
    # 构架顶部加强筋与铆钉
    cv2.line(canvas, (bx0 + 60, by0 + 250), (bx1 - 60, by0 + 250), 170, 8)
    for rx in range(bx0 + 100, bx1 - 90, 180):
        cv2.circle(canvas, (rx, by0 + 280), 9, 175, -1)
        cv2.circle(canvas, (rx, by0 + 870), 9, 145, -1)
    # 构架中央固定摇枕铸件结构（为父件中心块精修提供稳固几何基准）
    cx, cy = (bx0 + bx1) // 2, by0 + 730
    cv2.rectangle(canvas, (cx - 170, cy - 130), (cx + 170, cy + 130), 94, -1)
    cv2.rectangle(canvas, (cx - 170, cy - 130), (cx + 170, cy + 130), 185, 6)
    cv2.circle(canvas, (cx, cy), 68, 42, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 68, 205, 6, cv2.LINE_AA)
    cv2.line(canvas, (cx - 150, cy), (cx - 82, cy), 196, 8)
    cv2.line(canvas, (cx + 82, cy), (cx + 150, cy), 196, 8)
    cv2.line(canvas, (cx, cy - 115), (cx, cy - 78), 196, 8)
    cv2.line(canvas, (cx, cy + 78), (cx, cy + 115), 196, 8)


def _render_train_canvas(seed=SEED):
    """生成 100% 纯程序合成的 3 节编组列车长图画布（4096 × 24000 uint8）、定位靶与零件标注。"""
    rng = np.random.default_rng(seed)
    canvas = np.empty((H, TOTAL_W), dtype=np.uint8)

    # 0. 空场背景：沿纵向平缓变化（mean≈70, p99≈122），沿横向恒定（column_activity≈0）
    y = np.arange(H, dtype=np.float32)
    bg_col = 64.0 + 24.0 * np.sin(y / 420.0) + 12.0 * np.cos(y / 130.0)
    bg_col[3880:3980] += 36.0   # 远处钢轨反光带
    bg_col = np.clip(bg_col, 35, 130).astype(np.uint8)
    canvas[:] = bg_col[:, None]

    # 金属细颗粒底纹（tile 复用，快速且防止 flat/blur 误报）
    grain_tile = rng.integers(-9, 10, size=(512, 512), dtype=np.int16)

    def add_grain(x0, x1, y0, y1, scale=1.0):
        roi = canvas[y0:y1, x0:x1].astype(np.int16)
        gy = np.arange(y0, y1) & 511
        gx = np.arange(x0, x1) & 511
        g = grain_tile[gy[:, None], gx[None, :]]
        if scale != 1.0:
            g = np.round(g * scale).astype(np.int16)
        canvas[y0:y1, x0:x1] = np.clip(roi + g, 0, 255).astype(np.uint8)

    anchors = []
    labels = []
    next_id = 1
    foreign_pocket = None

    def add_part(typ, bbox, poly=None, parent=None):
        nonlocal next_id
        lid = next_id
        next_id += 1
        labels.append({
            'id': lid, 'type': typ, 'bbox': [int(v) for v in bbox],
            'poly': poly if poly is not None else _rect_poly(*bbox),
            'parent': parent, 'source': 'synthetic', 'verified': True,
        })
        return lid

    for car in range(1, N_CARS + 1):
        cx0 = HEAD_X + (car - 1) * CAR_W
        cx1 = cx0 + CAR_W
        # 车厢主体区间（留出车端暗带，形成标准车厢连接处断口）
        body_l = cx0 + (0 if car == 1 else 280)
        body_r = cx1 - (0 if car == N_CARS else 280)

        # 车端贯通道暗影背景
        canvas[:, cx0:cx1] = 28
        # 顶部亮铝合金裙板（stitch.find_car_gaps 依赖 y∈[48,560) 亮带 + 车厢连接处暗口）
        canvas[30:720, body_l:body_r] = 186
        # 裙板横向折线、板缝与车厢编号
        cv2.line(canvas, (body_l, 120), (body_r - 1, 120), 145, 6)
        cv2.line(canvas, (body_l, 680), (body_r - 1, 680), 82, 10)
        for sx in range(body_l + 600, body_r - 300, 900):
            cv2.line(canvas, (sx, 40), (sx, 710), 118, 4)
        cv2.putText(canvas, f'SYNTHETIC METRO CAR-{car:02d}', (body_l + 1950, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.2, 72, 8, cv2.LINE_AA)

        # 中部车体裙板下缘 & 线缆管路带
        canvas[720:1420, body_l:body_r] = 112
        for py in (780, 840, 1340):
            cv2.line(canvas, (body_l, py), (body_r - 1, py), 165, 8)
            cv2.line(canvas, (body_l, py + 10), (body_r - 1, py + 10), 58, 6)

        # 下部走行部阴影舱与钢轨带
        canvas[1420:3880, body_l:body_r] = 48
        canvas[3880:4096, cx0:cx1] = 86
        cv2.line(canvas, (cx0, 3915), (cx1 - 1, 3915), 182, 12)
        cv2.line(canvas, (cx0, 3955), (cx1 - 1, 3955), 42, 8)

        # 沿车身叠加细金属纹理与非周期纵纹，避免平块误报
        add_grain(cx0, cx1, 0, H, scale=1.0)
        for vx in range(body_l + 140, body_r - 140, 260):
            tone = int(rng.integers(68, 128))
            cv2.line(canvas, (vx, 1440), (vx, 2120), tone, 3)

        # 2 个独特高纹理定位靶（每节车左右各 1 个，共 6 个）
        for ai, ax in enumerate((cx0 + 900, cx0 + 4588), 1):
            ay, sz = 1480, 512
            aid = f'A{car}{ai:02d}'
            _draw_anchor_plaque(canvas, ax, ay, sz, rng, aid)
            anchors.append({'id': aid, 'car': car, 'x': ax, 'y': ay, 'w': sz, 'h': sz})

        # 车体中部设备（BODY 区）：铭牌、设备箱、风缸
        mark_box = [cx0 + 2720, 920, cx0 + 3240, 1220]
        _draw_mark(canvas, *mark_box, car)
        add_part('MARK', mark_box)

        equip_box = [cx0 + 2460, 1460, cx0 + 3540, 2360]
        _draw_equip(canvas, *equip_box, car)
        add_part('EQUIP', equip_box)

        tank_box = [cx0 + 2500, 2540, cx0 + 3500, 3200]
        _draw_tank(canvas, *tank_box)
        add_part('TANK', tank_box)

        # B1 转向架（左）
        b1 = [cx0 + 420, 2180, cx0 + 2260, 3780]
        _draw_bogie_chassis(canvas, *b1)
        b1_id = add_part('BOGIE', b1)

        bx0, by0 = b1[0], b1[1]
        ae1 = [bx0 + 220, by0 + 910, bx0 + 480, by0 + 1170]
        _draw_axle_end(canvas, *ae1)
        add_part('AXLE_END', ae1, _circle_poly((ae1[0] + ae1[2]) / 2, (ae1[1] + ae1[3]) / 2, 126), parent=b1_id)

        sp1 = [bx0 + 160, by0 + 420, bx0 + 410, by0 + 800]
        _draw_spring1(canvas, *sp1)
        add_part('SPRING1', sp1, parent=b1_id)

        dm1 = [bx0 + 465, by0 + 350, bx0 + 605, by0 + 860]
        _draw_damper(canvas, *dm1)
        add_part('DAMPER', dm1, parent=b1_id)

        lk1 = [bx0 + 820, by0 + 1000, bx0 + 1220, by0 + 1150]
        _draw_link(canvas, *lk1)
        add_part('LINK', lk1, parent=b1_id)

        bt1 = [bx0 + 700, by0 + 350, bx0 + 756, by0 + 406]
        _draw_bolt(canvas, *bt1)
        add_part('BOLT', bt1, _circle_poly((bt1[0] + bt1[2]) / 2, (bt1[1] + bt1[3]) / 2, 26), parent=b1_id)

        bt2 = [bx0 + 910, by0 + 350, bx0 + 966, by0 + 406]
        _draw_bolt(canvas, *bt2)
        add_part('BOLT', bt2, _circle_poly((bt2[0] + bt2[2]) / 2, (bt2[1] + bt2[3]) / 2, 26), parent=b1_id)

        # Car 3 的 B1 转向架右侧留出干净空腔，用于投放合成异物
        if car == 3:
            foreign_pocket = [bx0 + 1310, by0 + 430, bx0 + 1630, by0 + 770]

        # B2 转向架（右）
        b2 = [cx0 + 3740, 2180, cx0 + 5580, 3780]
        _draw_bogie_chassis(canvas, *b2)
        b2_id = add_part('BOGIE', b2)

        bx0, by0 = b2[0], b2[1]
        ae2 = [bx0 + 1360, by0 + 910, bx0 + 1620, by0 + 1170]
        _draw_axle_end(canvas, *ae2)
        add_part('AXLE_END', ae2, _circle_poly((ae2[0] + ae2[2]) / 2, (ae2[1] + ae2[3]) / 2, 126), parent=b2_id)

        as2 = [bx0 + 740, by0 + 240, bx0 + 1100, by0 + 540]
        _draw_airspring(canvas, *as2)
        add_part('AIRSPRING', as2, parent=b2_id)

        lk2 = [bx0 + 560, by0 + 1000, bx0 + 960, by0 + 1150]
        _draw_link(canvas, *lk2)
        add_part('LINK', lk2, parent=b2_id)

        bt3 = [bx0 + 430, by0 + 350, bx0 + 486, by0 + 406]
        _draw_bolt(canvas, *bt3)
        add_part('BOLT', bt3, _circle_poly((bt3[0] + bt3[2]) / 2, (bt3[1] + bt3[3]) / 2, 26), parent=b2_id)

        # 轻微终饰噪点使整节车厢更具真实相机颗粒感
        add_grain(body_l, body_r, 30, 3880, scale=0.45)

    return canvas, anchors, labels, foreign_pocket


def _write_frames(canvas, dst_dir: Path, quality=92):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in range(1, N_FRAMES + 1):
        frame = canvas[:, (f - 1) * W: f * W]
        io.imwrite(dst_dir / f'{f}.jpg', frame, quality=quality)


def _make_t02_canvas(t01_canvas, seed=SEED):
    """由 T01 画布平移 (T02_DX, T02_DY) 并施加轻微光照与相机噪声，生成二次过车基线长图。"""
    rng = np.random.default_rng(seed + 101)
    out = np.empty_like(t01_canvas)
    # 沿用空场背景列作为左右边缘填充
    bg = t01_canvas[:, :1]
    out[:] = bg
    # 平移主体区域
    src_y0, dst_y0 = (0, T02_DY) if T02_DY >= 0 else (-T02_DY, 0)
    h_copy = H - abs(T02_DY)
    w_copy = TOTAL_W - T02_DX
    shifted = t01_canvas[src_y0:src_y0 + h_copy, :w_copy].astype(np.float32)
    # 轻微光照变化（测试算法亮度归一化与稳健性）
    shifted = shifted * 0.96 + 5.0
    # 仅在车体范围内加极轻相机噪声，保持空场静止便于 stitch 精准裁边
    noise_tile = rng.normal(0.0, 1.5, size=(512, 512)).astype(np.float32)
    train_x0 = HEAD_X
    train_x1 = HEAD_X + N_CARS * CAR_W
    gy = np.arange(h_copy) & 511
    gx = np.arange(train_x1 - train_x0) & 511
    shifted[:, train_x0:train_x1] += noise_tile[gy[:, None], gx[None, :]]
    out[dst_y0:dst_y0 + h_copy, T02_DX:TOTAL_W] = np.clip(shifted, 0, 255).astype(np.uint8)
    # 顶部/底部因 dy 移出的极窄带用临近行补齐，避免黑边
    if T02_DY > 0:
        out[:T02_DY, T02_DX:] = out[T02_DY:T02_DY + 1, T02_DX:]
    return out


# ------------------------------------------------------------------ 标定、模板与 T01 基准数据

def _write_demo_calib(trip, manifest, sx_base):
    refs, curve = [], []
    for p in manifest['train_panels']:
        car = p['car']
        for z, frac in (('B1', 0.23), ('B2', 0.77)):
            x = round(p['x0'] + p['width_px'] * frac, 1)
            sx = round(sx_base + (car - 2) * 0.002, 5)
            curve.append({'x': x, 'mm_per_px': sx, 'n': 1})
            refs.append({
                'name': '合成标定尺', 'nominal_mm': 2300.0, 'measured_px': round(2300.0 / sx, 2),
                'x': x, 'car': car, 'zone': z, 'mm_per_px': sx, 'ok': True,
            })
    cal = {
        'trip': trip,
        'sx_curve': curve,
        'sx_mean_mm_per_px': round(float(np.mean([c['mm_per_px'] for c in curve])), 6),
        'sy_mm_per_px': SY,
        'sy_ratio_ba': round(sx_base / SY, 5),
        'refs': refs,
        'source': '全合成脱敏演示数据（纯程序生成，零客户数据）',
    }
    io.write_json(io.trip_dir(trip) / 'calib.json', cal)
    return cal


def _build_demo_template(t01_manifest, anchors_t01, labels_t01, calib_t01):
    """在 数据/成品/模板/v1 下生成完整模板，不写 主数据/零件实例_v1.csv（避免污染 Git 跟踪目录）。"""
    d = template.tpl_dir(TEMPLATE_VER)
    t = Trip.from_id('T01')
    head = t01_manifest['head_x']
    anchors_out = []
    for a in anchors_t01:
        patch = d / 'anchors' / f"{a['id']}.png"
        io.imwrite(patch, t.region(a['x'], a['x'] + a['w'], a['y'], a['y'] + a['h']))
        anchors_out.append({
            'id': a['id'], 'car': a['car'],
            'u': a['x'] - head, 'v': a['y'], 'w': a['w'], 'h': a['h'],
            'patch': io.rel(patch),
        })
    parts = template.build_parts(t, t01_manifest, labels_t01, head, d, calib_t01)
    tpl = {
        'version': TEMPLATE_VER, 'trip': 'T01', 'head_x': head,
        'train_panels': t01_manifest['train_panels'],
        'anchors': anchors_out, 'parts': parts, 'calib': calib_t01,
        'labels_sha256': None, 'synthetic_demo': True,
    }
    io.write_json(d / 'template.json', tpl)
    return tpl


def _write_t01_baseline_outputs(t01_manifest, tpl):
    """为模板趟 T01 补齐基准 align/seg/parts/compare，使 SQLite 与跨趟历史完整展示 T01+T02。"""
    d = io.trip_dir('T01')
    head = t01_manifest['head_x']
    x0, x1 = t01_manifest['train_x0'], t01_manifest['train_x1']
    io.write_json(d / 'align.json', {
        'trip': 'T01', 'template_ver': TEMPLATE_VER, 'template_trip': 'T01',
        'anchors': [
            {'id': a['id'], 'car': a['car'], 'x': a['u'] + head, 'y': a['v'],
             'u': a['u'], 'v': a['v'], 'score': 1.0, 'ok': True, 'u_fit_residual': 0.0}
            for a in tpl['anchors']
        ],
        'anchors_found': len(tpl['anchors']), 'anchors_total': len(tpl['anchors']),
        'g': [{'x': x0, 'u': x0 - head, 'dy': 0.0}, {'x': x1, 'u': x1 - head, 'dy': 0.0}],
        'blocks_total': 2, 'blocks_used': 2,
        'residual_px': {'median': 0.0, 'p95': 0.0, 'max': 0.0, 'n': len(tpl['anchors'])},
        'scale_range': [1.0, 1.0], 'offset_range_px': [0.0, 0.0],
        'u_trim0': 0, 'qc2': [], 'qc2_flagged': 0,
    })
    io.write_json(d / 'seg.json', {'model': {'name': 'none', 'sha256': None}, 'dets': []})
    inst = []
    items = []
    for q in tpl['parts']:
        b = q['bbox_uv']
        bxy = [b[0] + head, b[1], b[2] + head, b[3]]
        poly_xy = [[round(u + head, 1), round(v, 1)] for u, v in q['poly_uv']]
        cxy = [round(q['centroid_uv'][0] + head, 1), round(q['centroid_uv'][1], 1)]
        inst.append({
            'code': q['code'], 'type': q['type'], 'parent': q.get('parent'),
            'car': q.get('car'), 'zone': q.get('zone'),
            'bbox_uv': list(b), 'offset_uv': [0.0, 0.0],
            'bbox_xy': bxy, 'poly_xy': poly_xy, 'centroid_xy': cxy,
            'area_px': q['area_px'], 'angle_deg': q.get('angle_deg'),
            'match_score': 1.0, 'found_by': 'pixel', 'seg_score': None,
            'crop': q['crop'], 'mask': q['mask'],
            'src_frames': match.frames_of(bxy[0], bxy[2]),
        })
        items.append({
            'code': q['code'], 'type': q['type'], 'status': 'ok', 'bbox_xy': bxy,
            'ncc': 1.0, 'diff_area': 0.0, 'shift_mm': 0.0, 'shift_vec_mm': [0.0, 0.0],
            'dtheta_deg': 0.0, 'alarms': [],
        })
    io.write_json(d / 'parts.json', {'trip': 'T01', 'template_ver': TEMPLATE_VER, 'instances': inst, 'extra': []})
    io.write_json(d / 'compare.json', {'trip': 'T01', 'template_ver': TEMPLATE_VER, 'items': items, 'foreign': []})


def _inject_t02_defects(t02_base_dir: Path, t02_raw_dir: Path, tpl, foreign_pocket_t01, seed=SEED):
    """复用 railside.eval.synth 的 4 类像素级缺陷合成原语，向 T02 原图注入缺失/转角/位移/异物。"""
    rng = np.random.default_rng(seed + 202)
    types = template.load_types()
    head = tpl['head_x']
    by_code = {p['code']: p for p in tpl['parts']}

    def part_in_t02(code):
        q = by_code[code]
        b = q['bbox_uv']
        return {
            'code': code, 'type': q['type'], 'car': q['car'],
            'bbox_xy': [b[0] + head + T02_DX, b[1] + T02_DY, b[2] + head + T02_DX, b[3] + T02_DY],
            'poly_xy': [[u + head + T02_DX, v + T02_DY] for u, v in q['poly_uv']],
        }

    fc = FrameCopy(t02_base_dir, t02_raw_dir, quality=94)
    injected = []

    # 1. 缺失（missing）：抹除 1 号车 B1 螺栓
    p_miss = part_in_t02('C1-B1-BOLT-01')
    box, extra = synth.make_missing(fc, p_miss, {}, rng)
    injected.append({'kind': 'missing', 'code': p_miss['code'], 'bbox_xy': box, 'params': extra})

    # 2. 转角（rotated / dtheta）：1 号车 B2 轴端装置旋转 ±12°
    p_rot = part_in_t02('C1-B2-AXLE_END-01')
    box, extra = synth.make_rotate(fc, p_rot, {'deg': 12.0}, rng)
    injected.append({'kind': 'rotated', 'code': p_rot['code'], 'bbox_xy': box, 'params': extra})

    # 3. 位移（shifted / shift）：2 号车 B1 减振器沿纵向平移 2 倍阈值（16 mm ≈ 32 px）
    p_sh = part_in_t02('C2-B1-DAMPER-01')
    box, extra = synth.make_shift(fc, p_sh, {'k': 2.0, 'angle': 1.35}, rng, types, sx=SX, sy=SY)
    injected.append({'kind': 'shifted', 'code': p_sh['code'], 'bbox_xy': box, 'params': extra})

    # 4. 异物（foreign）：3 号车 B1 转向架空腔内投放高对比度异物块
    fp = [
        foreign_pocket_t01[0] + T02_DX, foreign_pocket_t01[1] + T02_DY,
        foreign_pocket_t01[2] + T02_DX, foreign_pocket_t01[3] + T02_DY,
    ]
    box, extra = synth.make_foreign(fc, {'bbox_xy': fp}, {'shape': 'blob'}, rng)
    injected.append({'kind': 'foreign', 'code': 'C3-B1-BOGIE-01', 'bbox_xy': box, 'params': extra})

    fc.write()
    return injected


# ------------------------------------------------------------------ 主流程

def generate(data_dir=None, force=False, seed=SEED):
    """生成完整全合成脱敏演示数据并返回摘要 dict。"""
    t0 = time.perf_counter()
    target = Path(data_dir).expanduser().resolve() if data_dir else io.DATA
    action = _prepare_dir(target, force=force)
    if action == 'exists':
        info = io.read_json(target / MARKER)
        return {
            'step': 'demo', 'status': 'exists', 'data_dir': io.rel(target),
            'trips': info.get('trips', list(DEMO_TRIPS)), 'template_ver': info.get('template_ver', TEMPLATE_VER),
            'cars': info.get('cars', N_CARS), 'alarms': info.get('alarms'),
            'elapsed_s': round(time.perf_counter() - t0, 2),
            'hint': '演示数据已存在，跳过重复生成；如需覆盖重建请加 --force，启动浏览请运行 uv run railside app',
        }

    with _use_data_dir(target):
        t01_raw = target / '原始' / 'T01' / 'camImg'
        t02_base = target / '原始' / 'T02' / '_base'
        t02_raw = target / '原始' / 'T02' / 'camImg'

        # 1. 合成 T01 基准长图与帧序列
        t01_canvas, anchors_t01, labels_t01, foreign_pocket = _render_train_canvas(seed=seed)
        _write_frames(t01_canvas, t01_raw, quality=92)

        # 2. T01 走真实质检、拼接、标定与模板构建（不碰 主数据/）
        qc.run(SimpleNamespace(src=str(t01_raw), trip='T01', camera='camImg'))
        stitch.run(SimpleNamespace(trip='T01', car_previews=False))
        t01_manifest = io.read_json(io.trip_dir('T01') / 'manifest.json')
        calib_t01 = _write_demo_calib('T01', t01_manifest, SX)
        tpl = _build_demo_template(t01_manifest, anchors_t01, labels_t01, calib_t01)
        _write_t01_baseline_outputs(t01_manifest, tpl)

        # 3. 合成 T02 二次过车长图并注入 4 类像素级缺陷
        t02_canvas = _make_t02_canvas(t01_canvas, seed=seed)
        del t01_canvas
        _write_frames(t02_canvas, t02_base, quality=94)
        del t02_canvas
        try:
            injected = _inject_t02_defects(t02_base, t02_raw, tpl, foreign_pocket, seed=seed)
        finally:
            shutil.rmtree(t02_base, ignore_errors=True)

        # 4. T02 走真实质检 → 拼接 → 标定 → 对齐 → 零件匹配 → 对比 → 入库 → 报告
        qc.run(SimpleNamespace(src=str(t02_raw), trip='T02', camera='camImg'))
        stitch.run(SimpleNamespace(trip='T02', car_previews=False))
        t02_manifest = io.read_json(io.trip_dir('T02') / 'manifest.json')
        _write_demo_calib('T02', t02_manifest, SX)
        al_res = align.run(SimpleNamespace(trip='T02', template=TEMPLATE_VER))
        io.write_json(io.trip_dir('T02') / 'seg.json', {'model': {'name': 'none', 'sha256': None}, 'dets': []})
        mt_res = match.run(SimpleNamespace(trip='T02', template=TEMPLATE_VER, seg=None, out=None))
        cmp_res = compare.run(SimpleNamespace(
            trip='T02', template=TEMPLATE_VER, parts=None, types='主数据/零件类型.csv', out=None,
            suggest=False, suggest_out='实验/E3', quantile=99.0, relax=1.5,
            foreign_diff=compare.FOREIGN_DIFF, foreign_min_area=compare.FOREIGN_MIN_AREA,
            foreign_min_score=compare.FOREIGN_MIN_SCORE,
        ))
        db_res = db.build(target / '成品' / '库.sqlite', list(DEMO_TRIPS), TEMPLATE_VER)
        rep_res = report.write_report('T02', target / '成品' / '报告' / 'T02', TEMPLATE_VER)
        db.append_agent_log('T02', 'demo', 'railside demo', {'seed': seed},
                            {'alarms': cmp_res['alarms'], 'foreign': cmp_res['foreign']}, 'ok',
                            db_path=target / '成品' / '库.sqlite')

        # 统计告警类型分布
        cmp_json = io.read_json(io.trip_dir('T02') / 'compare.json')
        alarm_types = {}
        for it in cmp_json.get('items') or []:
            for a in it.get('alarms') or []:
                t = a['type'] if isinstance(a, dict) else str(a)
                alarm_types[t] = alarm_types.get(t, 0) + 1
        if cmp_json.get('foreign'):
            alarm_types['foreign'] = len(cmp_json['foreign'])

        elapsed = round(time.perf_counter() - t0, 2)
        summary = {
            'step': 'demo',
            'status': 'overwritten' if action == 'overwrite' else 'created',
            'synthetic': True,
            'seed': seed,
            'data_dir': io.rel(target),
            'trips': list(DEMO_TRIPS),
            'cars': len(t02_manifest.get('train_panels') or []),
            'template_ver': TEMPLATE_VER,
            'parts': len(tpl['parts']),
            'align_residual_px': al_res['residual_px'],
            'matched': mt_res['found'],
            'alarms': cmp_res['alarms'],
            'alarm_types': alarm_types,
            'injected': [{'kind': d['kind'], 'code': d['code']} for d in injected],
            'db': db_res['db'],
            'report': rep_res['out'],
            'elapsed_s': elapsed,
            'next': ('uv run railside app' if target == (io.ROOT / '数据').resolve()
                     else f'RAILSIDE_DATA="{target}" uv run railside app'),
        }
        io.write_json(target / MARKER, {
            'demo': True, 'seed': seed, 'trips': list(DEMO_TRIPS), 'template_ver': TEMPLATE_VER,
            'cars': summary['cars'], 'alarms': summary['alarms'], 'alarm_types': alarm_types,
        })
        return summary


def run(args):
    return generate(data_dir=args.data_dir, force=args.force, seed=args.seed)
