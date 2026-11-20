"""后台子进程：处理新趟（railside run）和智能体对话（railside agent）。

任务只存在内存里（本地单用户应用）；输出逐行收进环形缓冲，页面轮询看尾部。
"""
import collections
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import threading
import time

from .. import cli, io

LOG_LINES = 400
_ids = itertools.count(1)
_lock = threading.Lock()
JOBS = {}


class Unavailable(Exception):
    """要调的步骤还没实现（模块不存在）。"""


def step_ready(step):
    mod = cli.STEPS.get(step)
    if mod is None:
        return False
    try:
        return importlib.util.find_spec(f'railside.{mod}') is not None
    except (ImportError, ValueError):
        return False


def command(step, argv):
    """子进程命令；与 `uv run railside <步骤>` 等价，用当前解释器，不依赖 PATH。"""
    return [sys.executable, '-m', 'railside.cli', step, *argv]


class Job:
    def __init__(self, kind, trip, cmd, meta=None):
        self.id = f'{kind}-{next(_ids)}'
        self.kind, self.trip, self.cmd, self.meta = kind, trip, cmd, meta or {}
        self.status, self.rc = 'running', None
        self.lines = collections.deque(maxlen=LOG_LINES)
        self.started, self.ended = time.time(), None
        self.proc = None

    def view(self, tail=200):
        lines = list(self.lines)[-tail:]
        out = {'id': self.id, 'kind': self.kind, 'trip': self.trip, 'status': self.status, 'rc': self.rc,
               'cmd': ' '.join(self.cmd), 'meta': self.meta, 'started': self.started, 'ended': self.ended,
               'elapsed': round((self.ended or time.time()) - self.started, 1), 'log': lines}
        if self.kind == 'agent' and self.status != 'running':
            out['reply'] = agent_reply(list(self.lines), self.rc)
        return out


def start(kind, step, argv, trip, meta=None):
    if not step_ready(step):
        raise Unavailable(f'railside {step} 还没实现（缺模块 railside.{cli.STEPS.get(step, step)}），装好后再试')
    job = Job(kind, trip, command(step, argv), meta)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'RAILSIDE_ROOT': str(io.ROOT)}
    try:
        job.proc = subprocess.Popen(job.cmd, cwd=str(io.ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        job.lines.append(f'启动失败：{e}')
        job.status, job.rc, job.ended = 'failed', None, time.time()
    else:
        threading.Thread(target=_pump, args=(job,), daemon=True).start()
    with _lock:
        JOBS[job.id] = job
    return job


def _pump(job):
    for line in job.proc.stdout:
        job.lines.append(line.rstrip('\r\n'))
    job.rc = job.proc.wait()
    job.ended = time.time()
    job.status = 'ok' if job.rc == 0 else 'failed'
    if job.kind == 'run' and job.rc == 2:
        for line in reversed(job.lines):
            try:
                summary = json.loads(line)
            except ValueError:
                continue
            if isinstance(summary, dict) and summary.get('step') == 'run' and summary.get('status') == 'partial':
                job.status = 'partial'
            break


def get(job_id):
    return JOBS.get(job_id)


def all_jobs():
    with _lock:
        return sorted(JOBS.values(), key=lambda j: j.started, reverse=True)


def agent_reply(lines, rc):
    """智能体输出转成一条回复：末行是 JSON 且带 reply/text/answer/message 时取它，否则用全部输出。"""
    text = '\n'.join(lines).strip()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            break
        if isinstance(obj, dict):
            for k in ('reply', 'text', 'answer', 'message'):
                if isinstance(obj.get(k), str):
                    return obj[k]
        break
    if rc not in (0, None) and not text:
        return f'智能体退出码 {rc}，没有输出'
    return text or '（智能体没有输出）'

