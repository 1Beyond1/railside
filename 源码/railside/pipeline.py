"""S9 一键流程：原图 → 质检 → 拼接 → 标定 → 对齐 → 分割 → 身份 → 对比 → 入库 → 报告。

每步用一个子进程跑 `python -m railside.cli <步骤> ...`（和环境隔离，某步的依赖坏了不拖累别步），
取 stdout 最后一行的 JSON 摘要。缺模块、缺上游输入的步骤记跳过并说明，不崩；
质检 verdict 为 fail 就停下上报。核心步骤被跳过时，报告仍生成，但总状态为 partial。
每步写一条 railside.db.append_agent_log，最后一行打总摘要。

用法：uv run railside run <原图目录> --trip <新趟号> [--weights <分割权重>] [--template v<n>，默认最新]
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import cli, db, io

# 步骤顺序 = 设计.md §3 的 S1–S8，外加 S7 的入库、出报告。
ORDER = ('ingest', 'stitch', 'calib', 'align', 'segment', 'match', 'compare', 'db', 'report')
NEEDS_WEIGHTS = ('segment',)  # 只有给了 --weights 才跑
# 另一路在写的步骤：按 <趟> 加可选 --template 调，参数不认就去掉重试。
TRY_TEMPLATE = ('calib', 'align', 'match', 'compare')


def step_argv(step, trip, src=None, weights=None, template=None):
    """一组步骤参数（不含步骤名本身）。"""
    if step == 'ingest':
        if src is None:
            raise ValueError('ingest 需要原图目录')
        return [str(src), '--trip', trip]
    if step == 'stitch':
        return [trip]
    if step in TRY_TEMPLATE:
        return [trip, '--template', str(template or db.latest_template())]
    if step == 'segment':
        if not weights:
            raise ValueError('segment 需要 --weights')
        return [trip, '--weights', str(weights)]
    if step == 'db':
        template = template or db.latest_template()
        return ['build'] + (['--template', str(template)] if template else [])
    if step == 'report':
        return [trip]
    raise ValueError(f'不认识的步骤：{step}')


def available(step):
    """步骤模块在不在。返回 (可用, 原因)。"""
    mod = cli.STEPS.get(step)
    if not mod:
        return False, f'cli.STEPS 里没有 {step}'
    try:
        ok = importlib.util.find_spec(f'railside.{mod}') is not None
    except (ImportError, ValueError):
        ok = False
    return ok, ('' if ok else f'模块 railside.{mod} 还没实现')


def _template(trip, default):
    """使用本次指定的模板，不能让上次运行的 align.json 覆盖它。"""
    ver = default
    tp = io.OUT / '模板' / str(ver) / 'template.json'
    return ver, (io.read_json(tp) if tp.is_file() else None)


def prereq(step, trip, template=None):
    """这一步要的输入在不在。返回跳过原因；None 表示可以跑。

    缺输入的步骤不该让整趟停在半路（设计.md §4.1 的文件本来就允许缺）：
    记跳过并说明缺什么，后面的入库、报告照样出，报告里那节写“未运行”。
    """
    d = io.trip_dir(trip)
    template = template or db.latest_template()
    if step in ('align', 'match') and not template:
        return '数据/成品/模板 下没有模板，先跑 template build'
    if step == 'align':
        ver, tpl = _template(trip, template)
        if tpl is None:
            return f'模板 {ver} 不存在（{io.rel(io.OUT / "模板" / str(ver))}），先跑 template build'
        if not (d / 'manifest.json').is_file():
            return '缺 manifest.json，先跑 stitch'
    elif step == 'match':
        if not (d / 'align.json').is_file():
            return '缺 align.json（S5 没跑或失败），跳过 S6b 身份匹配'
        aligned_ver = io.read_json(d / 'align.json').get('template_ver')
        if aligned_ver != template:
            return f'align.json 的模板 {aligned_ver} 与本次指定的 {template} 不一致，先重跑 align'
        ver, tpl = _template(trip, template)
        if tpl is None:
            return f'模板 {ver} 不存在，跳过 S6b 身份匹配'
        if not tpl.get('parts'):
            return f'模板 {ver} 里没有零件（零件标注还没导入），跳过 S6b 身份匹配'
    elif step == 'compare':
        if not (d / 'parts.json').is_file():
            return '缺 parts.json（S6b 没跑或跳过），跳过 S8 对比'
    elif step == 'report':
        if not (d / 'manifest.json').is_file():
            return '缺 manifest.json，先跑 stitch'
    return None


def _skip(step, why):
    return {'step': step, 'status': 'skipped', 'seconds': 0.0, 'argv': [], 'summary': None, 'reason': why}


def _spawn(argv):
    """跑一个子进程：python -m railside.cli <argv>。返回 (rc, stdout, stderr)。测试里 monkeypatch 它。"""
    cmd = [sys.executable, '-m', 'railside.cli', *argv]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'RAILSIDE_ROOT': str(io.ROOT)}
    p = subprocess.run(cmd, cwd=str(io.ROOT), env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    return p.returncode, p.stdout, p.stderr


def last_json(text):
    """stdout 里最后一行能解析的 JSON（各步的摘要）。"""
    for line in reversed((text or '').splitlines()):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def _tail(text, n=3):
    lines = [x.strip() for x in (text or '').strip().splitlines() if x.strip()]
    return '；'.join(lines[-n:]) if lines else ''


def run_step(step, argv):
    """跑一步。返回 {step, status(ok/skipped/failed), seconds, argv, summary, reason}。"""
    ok, why = available(step)
    if not ok:
        return {'step': step, 'status': 'skipped', 'seconds': 0.0, 'argv': list(argv), 'summary': None, 'reason': why}
    t0 = time.time()
    rc, out, err = _spawn([step, *argv])
    used = list(argv)
    # 步骤模块还没定型：--template 不认就去掉重试一次
    if rc != 0 and '--template' in used and 'unrecognized arguments' in (err or ''):
        k = used.index('--template')
        used = used[:k] + used[k + 2:]
        rc, out, err = _spawn([step, *used])
    seconds = round(time.time() - t0, 2)
    summary = last_json(out)
    if rc != 0:
        reason = _tail(err) or _tail(out) or f'退出码 {rc}'
        return {'step': step, 'status': 'failed', 'seconds': seconds, 'argv': used, 'summary': summary,
                'reason': f'退出码 {rc}：{reason}'}
    return {'step': step, 'status': 'ok', 'seconds': seconds, 'argv': used, 'summary': summary,
            'reason': None if summary is not None else '没取到 JSON 摘要'}


def _say(msg):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print(msg, flush=True)


def _log(trip, r):
    db.append_agent_log(trip, r['step'], 'pipeline.run_step', {'argv': r['argv']},
                        {'status': r['status'], 'summary': r.get('summary'), 'reason': r.get('reason'),
                         'seconds': r['seconds']}, r['status'])


def archive_previous(trip):
    """整趟重跑前归档派生产物；保留原图、标注、权重和历史调用日志。"""
    d = io.trip_dir(trip)
    names = ('trip.json', 'qc.json', 'manifest.json', 'calib.json', 'align.json',
             'seg.json', 'parts.json', 'compare.json', 'parts', '预览', 'dzi')
    paths = [(d / name, name) for name in names if (d / name).exists()]
    report = io.OUT / '报告' / trip
    if report.exists():
        paths.append((report, 'report'))
    if not paths:
        return None
    backup = d / 'previous_runs' / uuid.uuid4().hex
    backup.mkdir(parents=True)
    for path, name in paths:
        path.rename(backup / name)
    return io.rel(backup)


def run_trip(src, trip, weights=None, template=None):
    """从原图跑到报告。返回总摘要（dict）。template 不给时取 数据/成品/模板 下最新的。"""
    template = template or db.latest_template()
    t0 = time.time()
    previous_run = archive_previous(trip)
    steps = []
    for i, step in enumerate(ORDER, 1):
        ok, why = available(step)
        if not ok:
            r = _skip(step, why)
        elif step in NEEDS_WEIGHTS and not weights:
            r = _skip(step, '没给 --weights（分割权重），跳过 S6a')
        else:
            why = prereq(step, trip, template)
            r = _skip(step, why) if why else run_step(
                step, step_argv(step, trip, src=src, weights=weights, template=template))
        if r['status'] == 'ok' and step == 'ingest':
            verdict = (r.get('summary') or {}).get('verdict')
            if verdict == 'fail':
                r['status'] = 'failed'
                r['reason'] = '质检结论 fail：' + ('；'.join((r['summary'] or {}).get('reasons') or []) or '见 qc.json')
        steps.append(r)
        _log(trip, r)
        _say(f"[{i}/{len(ORDER)}] {step}: {r['status']}" + (f"（{r['reason']}）" if r.get('reason') else '')
             + f" {r['seconds']}s")
        if r['status'] == 'failed':
            break
    report = next((s['summary'].get('out') for s in steps if s['step'] == 'report' and s['status'] == 'ok'
                   and s.get('summary')), None)
    failed = next((s for s in steps if s['status'] == 'failed'), None)
    incomplete = [s for s in steps if s['status'] != 'ok'
                  and (s['step'] not in NEEDS_WEIGHTS or weights)]
    status = 'failed' if failed else ('partial' if incomplete else 'ok')
    reason = (failed['reason'] if failed else
              ('未完成步骤：' + '；'.join(f"{s['step']}（{s['reason']}）" for s in incomplete)
               if incomplete else None))
    out = {
        'step': 'run', 'trip': trip, 'src': str(src), 'template': template, 'weights': weights,
        'previous_run': previous_run,
        'status': status, 'reason': reason, 'incomplete_steps': [s['step'] for s in incomplete],
        'qc_verdict': next((s['summary'].get('verdict') for s in steps if s['step'] == 'ingest' and s.get('summary')), None),
        'steps': [{'step': s['step'], 'status': s['status'], 'seconds': s['seconds'],
                   'reason': s['reason'], 'summary': s.get('summary')} for s in steps],
        'report': report, 'elapsed_s': round(time.time() - t0, 1),
    }
    db.append_agent_log(trip, 'run', 'pipeline.run', {'src': str(src), 'weights': weights, 'template': template},
                        {k: out[k] for k in ('status', 'reason', 'qc_verdict', 'report', 'elapsed_s', 'incomplete_steps')}
                        | {'steps': [[s['step'], s['status'], s['seconds']] for s in steps]}, out['status'])
    return out


def add_parser(sub, step):
    p = sub.add_parser(step, help='S9 一键流程：原图 → 质检 → … → 入库 → 报告')
    p.add_argument('src', help='原图目录（含 1.jpg..N.jpg，如 .../camImg）')
    p.add_argument('--trip', required=True, help='新趟号，如 T09')
    p.add_argument('--weights', default=None, help='分割权重；不给就跳过 S6a')
    p.add_argument('--template', default=None, help='模板版本；默认 数据/成品/模板 下最新的')
    p.set_defaults(func=run)


def run(args):
    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f'run: 原图目录不存在：{src}')
    summary = run_trip(src, args.trip, args.weights, args.template)
    io.emit(summary)
    if summary['status'] != 'ok':
        raise SystemExit(2 if summary['status'] == 'partial' else 1)
    return None
