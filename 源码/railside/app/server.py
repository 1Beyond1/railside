"""FastAPI 路由。接口一览见同目录 README.md。出错一律返回 {"error": "..."}。"""
from contextlib import asynccontextmanager
import hmac
import ipaddress
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data, dzi, jobs
from .. import io

STATIC = Path(__file__).parent / 'static'
TILE_CACHE = {'Cache-Control': 'public, max-age=86400'}
ALLOWED_WEIGHT_EXTS = {'.pt', '.onnx'}


class Unauthorized(Exception):
    pass


class Forbidden(Exception):
    pass


def is_loopback_host(host: str) -> bool:
    h = (host or '').strip().lower().strip('[]')
    if h in ('localhost', '127.0.0.1', '::1'):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _is_unc_path(raw: str) -> bool:
    s = (raw or '').strip().strip('"').strip("'")
    return s.startswith('\\\\') or s.startswith('//')


def _allowed_models_dirs() -> list[Path]:
    dirs = []
    env_dirs = (os.environ.get('RAILSIDE_MODELS_DIR') or '').strip()
    if env_dirs:
        for part in env_dirs.split(os.pathsep):
            if part.strip():
                dirs.append(Path(part.strip()).expanduser().resolve())
    dirs.append((io.ROOT / 'models').resolve())
    return dirs


def _validate_weights(raw: str) -> str:
    if _is_unc_path(raw):
        raise data.BadRequest('分割权重路径不允许使用网络共享/UNC 路径')
    s = raw.strip().strip('"').strip("'")
    wp = Path(s)
    wp = wp if wp.is_absolute() else io.ROOT / wp
    if wp.suffix.lower() not in ALLOWED_WEIGHT_EXTS:
        raise data.BadRequest(f'分割权重仅支持 .pt 或 .onnx 文件：{wp.name}')
    rp = wp.resolve()
    allowed = _allowed_models_dirs()
    if not any(rp.is_relative_to(d) for d in allowed):
        raise data.BadRequest(f'分割权重必须位于允许的模型目录内（models/ 或 RAILSIDE_MODELS_DIR）：{wp}')
    if not rp.is_file():
        raise data.BadRequest(f'分割权重文件不存在：{wp}')
    return str(rp)


class RunReq(BaseModel):
    src: str
    trip: str
    weights: str | None = None
    template: str | None = None


class AgentReq(BaseModel):
    trip: str
    message: str


def _err(status):
    def handler(_req: Request, exc: Exception):
        return JSONResponse({'error': str(exc)}, status_code=status)
    return handler


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    jobs.shutdown()


def create_app(host: str = '127.0.0.1', token: str | None = None):
    is_local = is_loopback_host(host)
    expected_token = (token if token is not None else (os.environ.get('RAILSIDE_APP_TOKEN') or '')).strip()

    def _check_write_auth(request: Request):
        if is_local and not expected_token:
            return
        if not is_local and not expected_token:
            raise Forbidden('非回环地址监听未配置 RAILSIDE_APP_TOKEN，已禁止写操作；请设置环境变量 RAILSIDE_APP_TOKEN 后重启')
        auth = request.headers.get('authorization') or ''
        supplied = ''
        if auth.lower().startswith('bearer '):
            supplied = auth[7:].strip()
        if not supplied:
            supplied = (request.headers.get('x-railside-token') or '').strip()
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            raise Unauthorized('需要有效的访问令牌（Authorization: Bearer 或 X-Railside-Token）')

    app = FastAPI(title='railside 走行部侧面巡检', docs_url='/api/docs', redoc_url=None,
                  openapi_url='/api/openapi.json', lifespan=_lifespan)
    app.add_exception_handler(data.BadRequest, _err(400))
    app.add_exception_handler(Unauthorized, _err(401))
    app.add_exception_handler(Forbidden, _err(403))
    app.add_exception_handler(data.NotFound, _err(404))
    app.add_exception_handler(dzi.TileMissing, _err(404))
    app.add_exception_handler(jobs.Conflict, _err(409))
    app.add_exception_handler(jobs.Unavailable, _err(501))

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/trips')
    def trips():
        return {'trips': data.list_trips()}

    @app.get('/api/trips/{trip}')
    def trip_info(trip: str):
        return data.trip_info(trip)

    @app.get('/api/trips/{trip}/summary')
    def summary(trip: str):
        return data.summary(trip)

    @app.get('/api/trips/{trip}/dzi.dzi')
    def dzi_xml(trip: str):
        data.check_trip(trip)
        return Response(dzi.pyramid(trip).dzi_xml(), media_type='application/xml')

    @app.get('/api/trips/{trip}/dzi_files/{level}/{tile}')
    def dzi_tile(trip: str, level: int, tile: str):
        data.check_trip(trip)
        try:
            col, row = (int(v) for v in tile.removesuffix('.jpg').split('_'))
        except ValueError:
            raise data.BadRequest(f'瓦片名应为 <列>_<行>.jpg：{tile}')
        return Response(dzi.tile_bytes(trip, level, col, row), media_type='image/jpeg', headers=TILE_CACHE)

    @app.get('/api/trips/{trip}/overview.jpg')
    def overview(trip: str, w: int = 2400, h: int = 64):
        data.check_trip(trip)
        w, h = min(max(w, 16), 8000), min(max(h, 8), 512)
        return Response(dzi.overview_bytes(trip, w, h), media_type='image/jpeg', headers=TILE_CACHE)

    @app.get('/api/trips/{trip}/overlay')
    def overlay(trip: str):
        return data.overlay(trip)

    @app.get('/api/trips/{trip}/alarms')
    def alarms(trip: str):
        return data.alarms(trip)

    @app.get('/api/trips/{trip}/parts/{code}')
    def part(trip: str, code: str):
        return data.part_detail(trip, code)

    @app.get('/api/trips/{trip}/parts/{code}/crop.jpg')
    def part_crop(trip: str, code: str, side: str = 'trip'):
        return Response(data.crop(trip, code, side), media_type='image/jpeg', headers=TILE_CACHE)

    @app.get('/api/trips/{trip}/foreign/{idx}/crop.jpg')
    def foreign_crop(trip: str, idx: int, side: str = 'trip'):
        return Response(data.foreign_crop(trip, idx, side), media_type='image/jpeg', headers=TILE_CACHE)

    @app.get('/api/jobs')
    def job_list():
        return {'jobs': [j.view(tail=40) for j in jobs.all_jobs()]}

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        j = jobs.get(job_id)
        if j is None:
            raise data.NotFound(f'没有任务 {job_id}')
        return j.view()

    @app.delete('/api/jobs/{job_id}')
    def cancel_job(job_id: str, request: Request):
        _check_write_auth(request)
        j = jobs.cancel(job_id)
        if j is None:
            raise data.NotFound(f'没有任务 {job_id}')
        return j.view()

    @app.post('/api/run')
    def run_trip(req: RunReq, request: Request):
        _check_write_auth(request)
        trip = req.trip.strip()
        if not data.TRIP_RE.match(trip):
            raise data.BadRequest(f'趟次号不合法：{trip!r}（如 T03）')
        if _is_unc_path(req.src):
            raise data.BadRequest('原图目录不允许使用网络共享/UNC 路径')
        src = Path(req.src.strip().strip('"').strip("'"))
        if not src.is_dir():
            raise data.BadRequest(f'原图目录不存在：{src}')
        template = (req.template or '').strip()  # 空：交给 railside run 取最新的模板
        if template and not data.TRIP_RE.fullmatch(template):
            raise data.BadRequest(f'模板版本不合法：{template!r}（如 v1）')
        weights_raw = (req.weights or '').strip().strip('"').strip("'")
        weights = _validate_weights(weights_raw) if weights_raw else ''
        argv = [str(src), '--trip', trip] + (['--template', template] if template else [])
        if weights:
            argv += ['--weights', weights]
        return jobs.start('run', 'run', argv, trip,
                          {'src': str(src), 'weights': weights or None, 'template': template or None}).view()

    @app.post('/api/agent')
    def agent(req: AgentReq, request: Request):
        _check_write_auth(request)
        msg = req.message.strip()
        if not msg:
            raise data.BadRequest('消息是空的')
        data.check_trip(req.trip)
        return jobs.start('agent', 'agent', ['--trip', req.trip, '--message', msg], req.trip, {'message': msg}).view()

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app

