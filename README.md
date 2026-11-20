# railside：地铁列车侧面视觉检测

输入地铁列车过车时线阵相机拍的侧面原始帧，输出整列长图、带编码的零件库，以及和模板车逐件比对后的告警，并提供本地网页浏览和智能体接口。

## 流程

| 步骤 | 命令 | 做什么 |
|---|---|---|
| 收录质检 | `railside ingest` | 检查帧序、缺帧、曝光和模糊 |
| 拼接 | `railside stitch` | 帧拼成整列长图，切瓦片 |
| 标定 | `railside calib` | 用已知车体尺寸求 mm/px |
| 模板车 | `railside template` / `label` | 建模板车：定位靶、零件多边形（SAM 2.1 框提示预标 + 人工核） |
| 对齐 | `railside align` | 新趟按定位靶对齐到模板车坐标 |
| 零件匹配 | `railside match`（可选 `segment`） | 逐件找到对应零件，给编码和匹配分 |
| 库与查询 | `railside db` / `query` | 写入 SQLite，按车、转向架、零件类型查 |
| 比对 | `railside compare` | 缺失、位移、转角、遮挡、异物告警 |
| 一键运行 | `railside run <原图目录> --trip <趟号>` | 从原始帧跑到报告 |
| 浏览 | `railside app` | 本地网页：整列浏览、零件详情、两趟对比 |
| 智能体 | `railside agent` / `mcp` | 把查询和比对开放成工具（MCP） |

## 安装

需要 Python 3.12+，用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
uv run railside --help
uv run pytest 源码/tests
```

分割和 SAM 预标依赖 `ultralytics`（在 `dl` 依赖组里，不随本仓库分发）。

## 目录

```
源码/railside/   代码包，每个步骤一个模块，cli.py 汇总成子命令
源码/tests/      测试（用合成数据，不需要真实图像）
主数据/          零件类型表和比对阈值
数据/            运行时生成（原图、中间结果、成品），不进仓库
```

## 数据

列车图像数据归所属单位所有，不在本仓库中，也不在许可范围内。

## 许可

本项目按 GNU AGPL-3.0 开源，见 `LICENSE`：可以自由使用、修改和分发，但基于本项目做的软件（包括以网络服务形式提供的）也必须按 AGPL-3.0 公开源代码。

如需在闭源产品或商业项目中使用，可另行取得商业授权，请通过 GitHub 联系作者。

Copyright (c) 2026 1Beyond1

`源码/railside/app/static/vendor/openseadragon/` 为 OpenSeadragon，按其自带许可（BSD-3-Clause）。
