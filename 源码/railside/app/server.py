"""FastAPI 路由。接口一览见同目录 README.md。出错一律返回 {"error": "..."}。"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data, dzi, jobs
from .. import io

STATIC = Path(__file__).parent / 'static'
TILE_CACHE = {'Cache-Control': 'public, max-age=86400'}


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


def create_app():
    app = FastAPI(title='railside 走行部侧面巡检', docs_url='/api/docs', redoc_url=None, openapi_url='/api/openapi.json')
    app.add_exception_handler(data.BadRequest, _err(400))
    app.add_exception_handler(data.NotFound, _err(404))
    app.add_exception_handler(dzi.TileMissing, _err(404))
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

    @app.post('/api/run')
    def run_trip(req: RunReq):
        trip = req.trip.strip()
        if not data.TRIP_RE.match(trip):
            raise data.BadRequest(f'趟次号不合法：{trip!r}（如 T03）')
        src = Path(req.src.strip().strip('"'))
        if not src.is_dir():
            raise data.BadRequest(f'原图目录不存在：{src}')
        template = (req.template or '').strip()  # 空：交给 railside run 取最新的模板
        if template and not data.TRIP_RE.fullmatch(template):
            raise data.BadRequest(f'模板版本不合法：{template!r}（如 v1）')
        weights = (req.weights or '').strip().strip('"')
        if weights:
            wp = Path(weights)
            wp = wp if wp.is_absolute() else io.ROOT / wp
            if not wp.is_file():
                raise data.BadRequest(f'分割权重文件不存在：{wp}')
            weights = str(wp.resolve())
        argv = [str(src), '--trip', trip] + (['--template', template] if template else [])
        if weights:
            argv += ['--weights', weights]
        return jobs.start('run', 'run', argv, trip,
                          {'src': str(src), 'weights': weights or None, 'template': template or None}).view()

    @app.post('/api/agent')
    def agent(req: AgentReq):
        msg = req.message.strip()
        if not msg:
            raise data.BadRequest('消息是空的')
        data.check_trip(req.trip)
        return jobs.start('agent', 'agent', ['--trip', req.trip, '--message', msg], req.trip, {'message': msg}).view()

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app

