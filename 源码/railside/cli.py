"""命令行入口：uv run railside <步骤> ...

每个步骤模块提供 add_parser(sub) 和 run(args)；run 返回摘要 dict，由这里打成一行 JSON。
"""
import argparse
import importlib
import sys

from . import io

# 步骤名 → 模块名。模块按需导入，启动快，某步缺依赖也不拖累别的步骤。
STEPS = {
    'ingest': 'qc',
    'stitch': 'stitch',
    'tiles': 'tiles',
    'calib': 'calib',
    'template': 'template',
    'label': 'label',
    'align': 'align',
    'train': 'segment',
    'segment': 'segment',
    'match': 'match',
    'compare': 'compare',
    'db': 'db',
    'query': 'query',
    'report': 'report',
    'run': 'pipeline',
    'agent': 'agent',
    'mcp': 'agent',
    'app': 'app',
    'demo': 'demo',
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog='railside', description='地铁列车侧面视觉检测')
    sub = ap.add_subparsers(dest='step', required=True)
    want = argv[0] if argv and not argv[0].startswith('-') else None
    for step, mod in STEPS.items():
        if want and step != want:
            sub.add_parser(step)
            continue
        try:
            m = importlib.import_module(f'.{mod}', __package__)
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith(__package__ + '.'):
                sub.add_parser(step, help='（未实现）')
                continue
            raise
        m.add_parser(sub, step)
    args = ap.parse_args(argv)
    if not hasattr(args, 'func'):
        ap.error(f'{args.step} 还没实现')
    summary = args.func(args)
    if summary is not None:
        io.emit(summary)


if __name__ == '__main__':
    main()
