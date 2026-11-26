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


class Conflict(Exception):
    """同一趟已有运行中的处理任务。"""


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
        self._lock = threading.Lock()
        self.status, self.rc = 'running', None
        self.lines = collections.deque(maxlen=LOG_LINES)
        self.started, self.ended = time.time(), None
        self.proc = None

    def view(self, tail=200):
        with self._lock:
            lines_all = list(self.lines)
            status, rc, ended = self.status, self.rc, self.ended
        lines = lines_all[-tail:] if tail else []
        out = {'id': self.id, 'kind': self.kind, 'trip': self.trip, 'status': status, 'rc': rc,
               'cmd': ' '.join(self.cmd), 'meta': self.meta, 'started': self.started, 'ended': ended,
               'elapsed': round((ended or time.time()) - self.started, 1), 'log': lines}
        if self.kind == 'agent' and status != 'running':
            out['reply'] = agent_reply(lines_all, rc)
        return out


def start(kind, step, argv, trip, meta=None):
    if not step_ready(step):
        raise Unavailable(f'railside {step} 还没实现（缺模块 railside.{cli.STEPS.get(step, step)}），装好后再试')
    with _lock:
        if kind == 'run':
            for existing in JOBS.values():
                with existing._lock:
                    if existing.kind == 'run' and existing.trip == trip and existing.status == 'running':
                        raise Conflict(f'趟次 {trip} 已有正在运行的处理任务（{existing.id}），请等待完成或先取消')
        job = Job(kind, trip, command(step, argv), meta)
        JOBS[job.id] = job
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
           'RAILSIDE_ROOT': str(io.ROOT), 'RAILSIDE_DATA': str(io.DATA)}
    try:
        proc = subprocess.Popen(job.cmd, cwd=str(io.ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        with job._lock:
            job.lines.append(f'启动失败：{e}')
            job.status, job.rc, job.ended = 'failed', None, time.time()
    else:
        with job._lock:
            job.proc = proc
        threading.Thread(target=_pump, args=(job,), daemon=True).start()
    return job


def _pump(job):
    stdout = job.proc.stdout
    try:
        if stdout is not None:
            for line in stdout:
                with job._lock:
                    job.lines.append(line.rstrip('\r\n'))
    finally:
        if stdout is not None:
            try:
                stdout.close()
            except OSError:
                pass
        rc = job.proc.wait()
        with job._lock:
            job.rc = rc
            if job.ended is None:
                job.ended = time.time()
            if job.status == 'running':
                new_status = 'ok' if rc == 0 else 'failed'
                if job.kind == 'run' and rc == 2:
                    for line in reversed(job.lines):
                        try:
                            summary = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(summary, dict) and summary.get('step') == 'run' and summary.get('status') == 'partial':
                            new_status = 'partial'
                            break
                job.status = new_status


def cancel(job_id):
    job = get(job_id)
    if job is None:
        return None
    with job._lock:
        if job.status != 'running':
            return job
        proc = job.proc
        job.status = 'cancelled'
        job.ended = time.time()
        job.lines.append('任务已取消')
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
    return job


def shutdown():
    """服务退出时终止所有仍在运行的后台任务子进程。"""
    with _lock:
        ids = [j.id for j in JOBS.values() if j.status == 'running']
    for jid in ids:
        cancel(jid)


def get(job_id):
    with _lock:
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

