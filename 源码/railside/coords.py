"""坐标换算（设计.md §2）。别处一律调用这里。

- 原图坐标 (f, c, r)：帧号 f 从 1 起，列 c、行 r 从 0 起，未翻转。
- 趟坐标 (x, y)：按时间顺序拼接、不翻转。x = (f-1)*W + c，y = r。
- 模板坐标 (u, v)：对齐到模板趟后的坐标，u = 0 是车头，往车尾增大。
- 显示：给人看的图水平翻转，车头在右边。
"""
import numpy as np

W = 2000  # 每帧列数
H = 4096  # 每帧行数


def raw_to_trip(f, c, r=0):
    return (np.asarray(f) - 1) * W + np.asarray(c), np.asarray(r)


def trip_to_raw(x, y=0):
    x = np.asarray(x)
    return x // W + 1, x % W, np.asarray(y)


def display_x(x, x0, x1):
    """趟坐标 x 在区间 [x0, x1) 的翻转显示图里的列号。"""
    return (x1 - 1) - np.asarray(x)


def from_display_x(xd, x0, x1):
    return (x1 - 1) - np.asarray(xd)


class AlignMap:
    """新趟趟坐标 x ↔ 模板坐标 u 的单调映射；dy = 新趟 y − 模板 v。

    由 align.json 的 g[]（x、u 采样点）和 dy[] 构造。采样点之外线性外推。
    """

    def __init__(self, xs, us, dys=None):
        xs, us = np.asarray(xs, float), np.asarray(us, float)
        order = np.argsort(xs)
        self.xs, self.us = xs[order], us[order]
        if np.any(np.diff(self.us) <= 0):
            raise ValueError('映射不单调')
        self.dys = np.zeros_like(self.xs) if dys is None else np.asarray(dys, float)[order]

    @classmethod
    def from_json(cls, align):
        g = align['g']
        return cls([p['x'] for p in g], [p['u'] for p in g], [p.get('dy', 0.0) for p in g])

    @staticmethod
    def _interp(q, a, b):
        q = np.asarray(q, float)
        out = np.interp(q, a, b)
        if len(a) >= 2:
            lo, hi = q < a[0], q > a[-1]
            k0 = (b[1] - b[0]) / (a[1] - a[0])
            k1 = (b[-1] - b[-2]) / (a[-1] - a[-2])
            out = np.where(lo, b[0] + (q - a[0]) * k0, out)
            out = np.where(hi, b[-1] + (q - a[-1]) * k1, out)
        return out

    def x_to_u(self, x):
        return self._interp(x, self.xs, self.us)

    def u_to_x(self, u):
        return self._interp(u, self.us, self.xs)

    def dy_at_x(self, x):
        return np.interp(np.asarray(x, float), self.xs, self.dys)

    def to_template(self, x, y):
        return self.x_to_u(x), np.asarray(y, float) - self.dy_at_x(x)

    def to_trip(self, u, v):
        x = self.u_to_x(u)
        return x, np.asarray(v, float) + self.dy_at_x(x)

    def scale_at_u(self, u):
        """模板 1 px 对应新趟多少 px（局部横向比例）。"""
        u = np.asarray(u, float)
        return (self.u_to_x(u + 50) - self.u_to_x(u - 50)) / 100.0

