"""S10 应用：`railside app [--host 127.0.0.1] [--port 8765]` 起本地网页。说明见同目录 README.md。"""


def add_parser(sub, step):
    p = sub.add_parser(step, help='S10 本地网页应用（长图浏览、零件框、告警、处理新趟、智能体对话）')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--export', metavar='目录', help='不起服务，导出离线浏览包（双击 index.html 看，只读）')
    p.add_argument('--trip', action='append', help='导出哪几趟，可重复；默认所有跑完 match 的趟')
    p.add_argument('--max-level', type=int, default=None, help='长图瓦片导到第几层；默认比原分辨率低一层（半分辨率）')
    p.set_defaults(func=run)


def run(args):
    import sys

    if args.export:
        from .export import export
        return export(args.export, args.trip, args.max_level)

    import uvicorn

    from .server import create_app
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    url = f'http://{args.host}:{args.port}/'
    print(f'railside 应用已启动：{url}（Ctrl+C 退出）', flush=True)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level='warning')
    return {'step': 'app', 'url': url, 'status': 'stopped'}
