# 智能体工具的 MCP 接入

`uv run railside mcp` 起一个 MCP 服务（stdio），把 `railside.agent` 的那套工具原样暴露出去，
Claude Code 这类客户端可以直接接。工具实现、留痕和 `railside agent` 完全一样：
每次调用写一条 `railside.db.append_agent_log`（`数据/中间/<趟>/agent_log.jsonl`，库在时同时写库）。

## 暴露的工具

| 工具 | 参数 | 作用 |
|---|---|---|
| `qc_summary` | `trip?` | 质检结论（S1）：verdict、帧数、缺帧、有标记的帧 |
| `align_summary` | `trip?` | 对齐摘要（S5）：定位靶、残差中位/p95、局部比例、质检第二段标记 |
| `calib_summary` | `trip?` | 标定摘要（S3）：横向/纵向 mm/px、尺寸来源 |
| `alarms` | `trip?, type?` | 告警列表，按严重度排序；`type` 取 missing/foreign/ncc/shift/dtheta/area/qc |
| `part_history` | `code, trip?` | 一个零件编码的跨趟历史（识别方式、匹配分、NCC、位移、转角、告警） |
| `sql` | `text` | 只读 SQL（SELECT/WITH），查 `数据/成品/库.sqlite` |
| `run_step` | `step, trip?, src?` | 跑流程里的一步（ingest/stitch/calib/align/match/compare/db/report） |
| `run_trip` | `src?, trip?, weights?` | 跑整趟：原图 → 质检 → … → 入库 → 报告 |
| `report` | `trip?` | 出这一趟的 HTML 报告 |

`trip` 都可不给，但 MCP 服务本身没有"当前趟"，不给会报"没有指定趟次"。

## 接入 Claude Code

用虚拟环境里的入口（推荐，工作目录无关，包是 editable 安装，项目根自动识别）：

```
claude mcp add railside -- "D:\path\to\railside\.venv\Scripts\railside.exe" mcp
```

或者走 uv（每次启动会先同步环境，慢一点）：

```
claude mcp add railside -- uv run --project "D:\path\to\railside" railside mcp
```

项目挪了位置、或要用别的数据目录时，加一个环境变量 `RAILSIDE_ROOT=<项目根>`
（`railside.io` 优先读它，没有才按包路径推断）。Windows 下在 `claude mcp add` 后用
`claude mcp get railside` 看配置，需要时在 `~/.claude.json` 里给这条服务的 `env` 加 `RAILSIDE_ROOT`。

## 自查

```
uv run python - <<'PY'
import asyncio, sys
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=['-m', 'railside.cli', 'mcp'])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            print((await s.call_tool('alarms', {'trip': 'T02'})).content[0].text)

asyncio.run(main())
PY

同一件事有测试兜底：`源码/tests/test_agent.py::test_mcp_stdio_lists_tools_and_queries`
（起服务、列工具、查一次 T02 的告警）。

## 注意

- 服务用 stdout 传协议，任何 `print` 都会弄坏连接；工具的输出只走 MCP 返回值，日志只写文件。
- 只读 SQL 由 `railside.query.sql` 把关（只许 SELECT/WITH，库以只读方式打开）。
- 跑步骤、跑整趟是真的起子进程（`python -m railside.cli <步骤>`），耗时和命令行一样。