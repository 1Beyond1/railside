# S10 应用：走行部侧面巡检（本地网页）

```
uv run railside app [--host 127.0.0.1] [--port 8765]
```

浏览器打开 `http://127.0.0.1:8765/`。如果没有真实数据，可先运行 `uv run railside demo` 生成纯合成脱敏示例数据，再运行 `uv run railside app` 启动浏览。

离线浏览包（别人电脑上零安装看结果）：

```
uv run railside app --export <输出目录> [--trip T02]
```

导出接口结果（`data/data.js`）、半分辨率长图瓦片、每个零件两侧局部图，双击包里的 `index.html` 就能看。只读：没有“处理新趟”和“智能体”。T02 约 126 MB。后端 FastAPI + uvicorn，前端是 `static/` 下的纯 HTML/JS/CSS，不用构建。

## 页面

| 区域 | 内容 |
|---|---|
| 顶栏 | 趟次切换（`数据/中间/*/manifest.json`，圆点颜色是质检结论）；状态条：质检、编组、对齐残差、零件数、告警数 |
| 车厢导航条 | 整车缩略图（翻转，车头在右），按 `train_panels` 分节；点「第 k 节」跳到该节车头端，点或拖动其他位置平移；蓝框是当前视野；底部短线是告警位置 |
| 长图与对比 | 上下双屏 OpenSeadragon 深度缩放浏览，支持「模板对比（默认：上半本趟识别、下半标准模板按 `align.json` 分段映射精准联动，支持叠加模板框与同步高亮）/ 原图对比 / 单屏」切换及拖动分隔条调比例；左下读数双屏联动 |
| 叠加 | 零件有轮廓画多边形，父件画框；按状态着色：正常（绿）、告警（橙）、缺失（红虚线）、模板标准框（青蓝）、定位靶（紫虚线）。点框打开零件详情 |
| 侧栏·告警 | 本趟全部告警（对比告警、缺失、异物），支持按类型筛选与「上一条/下一条」一键巡检（`[` / `]`）；点一条跳到并双屏高亮该零件 |
| 侧栏·零件 | 全部零件检索与筛选（按编码/名称、车厢、类型、状态），支持一键定位 |
| 侧栏·质检对齐 | `qc.json` 结论与理由、拼接与车厢表、`align.json` 残差等摘要、标定/分割/识别/对比概况、流水线文件是否齐 |
| 侧栏·处理新趟 | 填原图目录和趟次号，后台跑 `railside run <目录> --trip <趟>`，轮询显示状态和日志尾部；完成后趟次列表自动刷新 |
| 侧栏·智能体 | 对话框；每条消息起一个 `railside agent --trip <趟> --message <文本>` 子进程，输出显示为回复 |
| 零件与异物详情 | 编码、类型、父件（可点）、车/区、几何、对比指标（触发告警的高亮）、模板与本趟局部图并排（含异物局部对比图，点击支持大图卷帘/并排/按住切换对比）、跨趟历史 |

键盘：`1`–`6` 跳到第 k 节车，`H` 适应高度，`0` 看整列，`+`/`-` 缩放，`[`/`]` 切换告警，`B` 折叠/展开右侧栏，`Esc` 关弹窗/详情。

截图：

<!-- 截图位：整列浏览 + 零件框 -->
<!-- 截图位：零件详情（模板/本趟并排） -->
<!-- 截图位：告警列表 -->
<!-- 截图位：质检与对齐 -->
<!-- 截图位：处理新趟 / 智能体对话 -->

## 接口

出错一律返回 `{"error": "<中文说明>"}`：400 参数不对，404 趟次/零件/瓦片不存在，501 要调的步骤还没实现。

坐标约定：`xy` 是趟坐标框 `[x0, y0, x1, y1]`（不翻转）；`rect` 是翻转显示图里的 `[xd, y, w, h]`，`xd = train_x1 - 1 - x`（`coords.display_x`）。

| 方法 | 路径 | 返回 |
|---|---|---|
| GET | `/api/trips` | `trips[]`：`trip`、`n_frames`、`train_x0`、`train_x1`、`n_cars`、`verdict`、`files[]` |
| GET | `/api/trips/{趟}` | 车内范围、`head_x`、显示图 `width`/`height`、`dzi`（url、tile、overlap、levels）、`cars[]`（`car`、`x0`、`x1`、`xd`、`w`） |
| GET | `/api/trips/{趟}/dzi.dzi` | DZI 描述（XML，512 px 瓦片，重叠 1） |
| GET | `/api/trips/{趟}/dzi_files/{层}/{列}_{行}.jpg` | 瓦片 JPEG |
| GET | `/api/trips/{趟}/overview.jpg?w=&h=` | 整车缩略条（导航条用） |
| GET | `/api/trips/{趟}/overlay` | `source`（`template`/`parts`/`annotation`/null）、`message`、`counts`、`boxes[]`（`code`、`type`、`type_name`、`parent`、`car`、`zone`、`status`、`alarms[]`、`xy`、`rect`）、`anchors[]`、`template_ver`、`template_trip` |
| GET | `/api/trips/{趟}/alarms` | `rows[]`（同 boxes，另有 `kind`=`part`/`foreign`、`metrics`）、`types`（告警类型计数）、`message` |
| GET | `/api/trips/{趟}/parts/{编码}` | `record`、`template`（版本、模板趟、零件几何）、`instance`、`compare`、`crops`、`history`（`source`、`note`、`rows[]`） |
| GET | `/api/trips/{趟}/parts/{编码}/crop.jpg?side=template\|trip` | 局部图 JPEG（翻转，框出零件） |
| GET | `/api/trips/{趟}/foreign/{序号}/crop.jpg?side=template\|trip` | 异物局部对比图 JPEG（翻转，框出模板同位置或本趟异物区域） |
| GET | `/api/trips/{趟}/summary` | `qc`、`stitch`、`calib`、`align`、`seg`、`parts`、`compare` 摘要（缺哪个文件就没有哪个键）、`files[]` |
| POST | `/api/run` | 体 `{src, trip, template?: "v1", weights?: null}`；起 `railside run`，返回任务。权重可填项目相对路径或绝对路径，空值使用纯模板匹配；不存在的权重文件返回 400 |
| POST | `/api/agent` | 体 `{trip, message}`；起 `railside agent`，返回任务 |
| GET | `/api/jobs`、`/api/jobs/{id}` | 任务：`id`、`kind`、`trip`、`status`（`running`/`ok`/`partial`/`failed`）、`rc`、`elapsed`、`log[]`（尾部）；智能体任务结束后带 `reply` |

接口文档（自动生成）：`/api/docs`。

## 数据来源与退化

- 零件框：模板趟（`template.json` 的 `trip`）叠模板零件，`x = u + head_x`；其他趟叠 `parts.json` 的 `bbox_xy`（缺时用 `bbox_uv` 经 `align.json` 映射）。状态：`found_by == 'missing'` 为缺失，`compare.json` 的 `alarms` 非空为告警，其余正常。模板还没有零件且没有 `parts.json` 时，叠 `标注/模板<趟>/零件.json` 的人工标注（编码记为 `#<id>`）。
- `alarms[]` 的元素可以是字符串，也可以是带 `type` 的对象。
- 跨趟历史：先调 `railside.query.part(编码)`，取返回里的 `history`/`instances`/`rows` 列表；模块不存在、出错或形状不认识时，直接读各趟 JSON。
- 智能体回复：子进程最后一行是 JSON 且带 `reply`（或 `text`/`answer`/`message`）时取它，否则显示全部输出。
- 哪个文件缺了，对应接口返回空列表和 `message`，页面显示空状态提示。

## 瓦片

金字塔按 DZI 约定：最高层 = `ceil(log2(max(宽, 高)))`，第 l 层缩小 `2^(最高层 − l)` 倍。

- 缩小 ≥ 8 倍的层从 `预览/整车_1-8.jpg` 缩放裁切（它已翻转、已裁到车内）。
- 更细的层用 `Trip.region(..., step=s)` 从原图取再翻转；原图帧按趟缓存 32 帧。
- 生成后缓存到 `数据/中间/<趟>/dzi/<train_x0>-<train_x1>/<层>/<列>_<行>.jpg`；重跑 stitch 改了车内范围会自动换目录。删掉 `dzi/` 可重建。

实测（本机 i7-13700H，T02，首次生成，含 HTTP）：全分辨率层 35–75 ms，1/2、1/4 层 10–110 ms，取自预览的层首张 120–170 ms（该层缩放一次）、之后 2–20 ms；已缓存的瓦片约 20 ms。

## 文件

| 文件 | 内容 |
|---|---|
| `__init__.py` | `add_parser`/`run`：`railside app` |
| `server.py` | FastAPI 路由 |
| `data.py` | 读各步 JSON：趟列表、零件框、告警、详情、历史、概况、局部图 |
| `dzi.py` | DZI 金字塔与瓦片缓存 |
| `jobs.py` | `run`/`agent` 子进程任务 |
| `static/` | `index.html`、`app.css`、`app.js`；`vendor/openseadragon/`：OpenSeadragon 6.1.1（BSD-3-Clause，见其 `VERSION.md`、`LICENSE.txt`） |

