简体中文 | [English](README.en.md)

# railside：地铁列车侧面视觉检测

输入地铁列车过车时线阵相机拍的侧面原始帧，输出整列长图、带编码的零件库，以及和模板车逐件比对后的告警，并提供本地网页浏览和智能体接口。

## 流程

| 步骤 | 命令 | 做什么 |
|---|---|---|
| 收录质检 | `railside ingest` | 检查帧序、缺帧、曝光和模糊 |
| 拼接 | `railside stitch` | 按时间顺序拼出整列车厢区间（`train_panels`），生成 `manifest.json` 和 1/8 预览图 |
| 切片 | `railside tiles` | 按趟坐标导出用于标注、训练和推理的重叠切片 |
| 标定 | `railside calib` | 用已知车体尺寸求 mm/px |
| 模板车 | `railside template` / `label` | 建模板车：定位靶、零件多边形（SAM 2.1 框提示预标 + 人工核） |
| 对齐 | `railside align` | 新趟按定位靶对齐到模板车坐标 |
| 分割训练与推理 | `railside train` / `segment` | 从模板趟标注构建并训练多尺度 YOLO-seg 模型，并对整趟滑窗推理（可选） |
| 零件匹配 | `railside match` | 逐件找到对应零件，给编码和匹配分 |
| 库与查询 | `railside db` / `query` | 写入 SQLite，按车、转向架、零件类型查 |
| 比对 | `railside compare` | 缺失、位移、转角、遮挡、异物告警 |
| 报告 | `railside report` | 生成单趟离线 HTML 报告（`index.html` + 局部图） |
| 一键运行 | `railside run <原图目录> --trip <趟号>` | 从原始帧跑到入库与报告 |
| 浏览 | `railside app` | 本地网页：整列 DZI 深度缩放浏览、零件详情、两趟对比（支持 `--export` 导出离线浏览包） |
| 智能体 | `railside agent` / `mcp` | 把查询、比对和流水线调用开放成智能体工具（MCP stdio） |

## 安装

需要 Python 3.12+，用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
uv run railside --help
uv run pytest 源码/tests
```

`pyproject.toml` 默认启用 `dev` 与 `dl` 依赖组（分割训练/推理与 SAM 预标依赖 `ultralytics`；模型权重不随本仓库分发）。如不需要深度学习依赖，可使用 `uv sync --no-group dl`。

## 目录

```text
源码/railside/   代码包，每个步骤一个模块，cli.py 汇总成子命令
源码/tests/      测试（用合成数据，不需要真实图像）
主数据/          零件类型表和比对阈值（零件类型.csv）
数据/            运行时生成（原图、中间结果、成品），不进仓库
```

## 数据

列车图像数据归所属单位所有，不在本仓库中，也不在许可范围内。

## 许可

本项目按 GNU AGPL-3.0 开源，见 [LICENSE](LICENSE)：可以自由使用、修改和分发，但基于本项目做的软件（包括以网络服务形式提供的）也必须按 AGPL-3.0 公开源代码。

如需在闭源产品或商业项目中使用，可另行取得商业授权，请通过 GitHub 联系作者。

Copyright (c) 2026 1Beyond1

`源码/railside/app/static/vendor/openseadragon/` 为 OpenSeadragon 6.1.1，按其自带许可（BSD-3-Clause，见 [LICENSE.txt](源码/railside/app/static/vendor/openseadragon/LICENSE.txt)）。
