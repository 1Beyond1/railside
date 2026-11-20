"""对齐考题：人工标两趟对应点（设计.md §5“对齐”格）。

程序只在第 1 趟挑点（角点、避开定位靶），第 2 趟由人看图点同一处；第 2 趟的图框中心加了随机偏移，
框的位置不提示答案。标好的点写 标注/评测/对应点.csv，score.py 拿它量对齐误差。

  python -m railside.eval.pairs            # 切图（第一次）并打开标注网页 http://127.0.0.1:8765
  python -m railside.eval.pairs --n 40     # 候选点数（标够 30 对即可，看不清的跳过）
"""
import argparse
import csv
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .. import io
from ..coords import AlignMap, H
from ..frames import Trip

SEED = 20261119
C1 = 128          # 第 1 趟小图半宽（点在正中）
C2 = 224          # 第 2 趟大图半宽
JITTER = 120      # 第 2 趟框中心随机偏移，px
OUT = io.ROOT / '数据' / '中间' / '评测' / '对应点'
CSV = io.ROOT / '标注' / '评测' / '对应点.csv'
FIELDS = ['id', 'x_t01', 'y_t01', 'x_t02', 'y_t02', 'note']


def _anchor_boxes(tpl, head):
    return [(a['u'] + head - 200, a['v'] - 200, a['u'] + a['w'] + head + 200, a['v'] + a['h'] + 200)
            for a in tpl['anchors']]


def _distinct(b, px, py, r=32, search=16, excl=6):
    """(px, py) 处 (2r+1)² 的块在 ±search 内和自己以外的位置最多有多像（NCC）。直边、平面会很像。"""
    pat = b[py - r:py + r + 1, px - r:px + r + 1]
    win = b[py - r - search:py + r + search + 1, px - r - search:px + r + search + 1]
    if pat.shape != (2 * r + 1, 2 * r + 1) or win.shape != (2 * (r + search) + 1,) * 2 or pat.std() < 15:
        return 1.0
    m = cv2.matchTemplate(win, pat, cv2.TM_CCOEFF_NORMED)
    cv2.circle(m, (search, search), excl, -1.0, -1)
    return float(m.max())


def _corner(t1, x, y):
    """(x, y) 附近 600×600 里离中心最近、两个方向都定得住的角点；没有返回 None。

    先模糊（σ=3）压掉颗粒噪声：噪声两趟不同，不能当对应点；再要求块在周围找不到相像的位置（排除直边）。"""
    x0, y0 = int(x) - 300, max(0, int(y) - 300)
    im = t1.region(x0, x0 + 600, y0, min(H, y0 + 600))
    b = cv2.GaussianBlur(im.astype(np.float32), (0, 0), 3)
    pts = cv2.goodFeaturesToTrack(b, 60, 0.01, 20, blockSize=15)
    if pts is None:
        return None
    pts = pts.reshape(-1, 2)
    for px, py in sorted(pts, key=lambda p: np.hypot(p[0] - 300, p[1] - 300)):
        px, py = int(round(px)), int(round(py))
        if not (C1 <= px < 600 - C1 and C1 <= py < im.shape[0] - C1):
            continue
        if _distinct(b, px, py) < 0.7:
            return px + x0, py + y0
    return None


def prepare(n):
    man = io.read_json(io.trip_dir('T01') / 'manifest.json')
    align = io.read_json(io.trip_dir('T02') / 'align.json')
    tpl = io.read_json(io.ROOT / '数据' / '成品' / '模板' / align['template_ver'] / 'template.json')
    head = man['head_x']
    amap = AlignMap.from_json(align)
    t1, t2 = Trip.from_id('T01'), Trip.from_id('T02')
    boxes = _anchor_boxes(tpl, head)
    rng = np.random.default_rng(SEED)
    a, b = man['train_x0'] + 3000, man['train_x1'] - 3000
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('P*_t0?.png'):
        old.unlink()
    items = []
    for k in range(n):
        for _ in range(150):
            x = a + (b - a) * (k + rng.uniform(0.1, 0.9)) / n
            y = rng.uniform(500, H - 500)
            p = _corner(t1, x, y)
            if p and not any(bx0 <= p[0] <= bx1 and by0 <= p[1] <= by1 for bx0, by0, bx1, by1 in boxes):
                break
        else:
            continue
        x1, y1 = p
        u, v = x1 - head, y1
        xe, ye = (float(z) for z in amap.to_trip(u, v))
        cx = int(round(xe + rng.uniform(-JITTER, JITTER)))
        cy = int(np.clip(round(ye + rng.uniform(-JITTER, JITTER)), C2, H - C2))
        o1 = (x1 - C1, y1 - C1)
        o2 = (cx - C2, cy - C2)
        pid = f'P{len(items) + 1:02d}'
        io.imwrite(OUT / f'{pid}_t01.png', t1.region(o1[0], o1[0] + 2 * C1, o1[1], o1[1] + 2 * C1))
        io.imwrite(OUT / f'{pid}_t02.png', t2.region(o2[0], o2[0] + 2 * C2, o2[1], o2[1] + 2 * C2))
        items.append({'id': pid, 'x_t01': x1, 'y_t01': y1, 'o1': o1, 'o2': o2})
    io.write_json(OUT / 'pairs.json', {'seed': SEED, 'items': items, 'code_ver': io.code_ver(), 'created_at': io.now()})
    return items


def read_rows(path=CSV):
    if not Path(path).is_file():
        return []
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write_rows(rows, path=CSV):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r['id']):
            w.writerow({k: r.get(k, '') for k in FIELDS})


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>对应点标注</title>
<style>
body{font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif;margin:16px;background:#f4f4f2;color:#222}
.row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.box{background:#fff;border:1px solid #ccc;padding:8px}
canvas{display:block;image-rendering:pixelated;cursor:crosshair}
h1{font-size:18px;margin:0 0 8px} .muted{color:#666} kbd{border:1px solid #aaa;border-radius:3px;padding:0 4px;background:#fff}
button{font:inherit;padding:4px 12px;margin-right:6px}
#bar{margin:8px 0}
</style>
<h1>对应点标注 <span id="prog" class="muted"></span></h1>
<div class="muted">左边第 1 趟，红十字是要找的点。在右边第 2 趟图上点<b>同一个位置</b>（同一个螺栓角、同一条边的拐点）。
右边两个放大图用来细调：在下面那张放大图上点一下、或按方向键（每次 0.5 px）。<kbd>Enter</kbd> 存并下一个，<kbd>S</kbd> 看不清、跳过，<kbd>B</kbd> 上一个。</div>
<div id="bar"><button id="prev">上一个 (B)</button><button id="skip">跳过 (S)</button><button id="ok">存并下一个 (Enter)</button>
<span id="msg" class="muted"></span></div>
<div class="row">
 <div class="box"><div>第 1 趟（×2）</div><canvas id="c1" width="512" height="512"></canvas></div>
 <div class="box"><div>第 2 趟（×1.5，点这里）</div><canvas id="c2" width="672" height="672"></canvas></div>
 <div class="box"><div>放大对照（×8，上：第 1 趟，下：第 2 趟）</div><canvas id="z1" width="256" height="256"></canvas><div style="height:8px"></div><canvas id="z2" width="256" height="256"></canvas></div>
</div>
<script>
const S1=2,S2=1.5,Z=8,C1=%C1%,C2=%C2%;
let items=[],done={},i=0,pt=null,im1=new Image(),im2=new Image();
const $=id=>document.getElementById(id);
function cross(g,x,y,r,col){g.strokeStyle=col;g.lineWidth=1.5;g.beginPath();g.moveTo(x-r,y);g.lineTo(x-3,y);g.moveTo(x+3,y);g.lineTo(x+r,y);g.moveTo(x,y-r);g.lineTo(x,y-3);g.moveTo(x,y+3);g.lineTo(x,y+r);g.stroke();}
function zoom(cv,img,cx,cy){const g=cv.getContext('2d');g.imageSmoothingEnabled=false;g.fillStyle='#000';g.fillRect(0,0,256,256);
 const w=256/Z;g.drawImage(img,cx-w/2,cy-w/2,w,w,0,0,256,256);cross(g,128,128,30,'#e00');}
function draw(){
 const g1=$('c1').getContext('2d');g1.imageSmoothingEnabled=false;g1.drawImage(im1,0,0,512,512);cross(g1,(C1+.5)*S1,(C1+.5)*S1,24,'#e00');
 const g2=$('c2').getContext('2d');g2.imageSmoothingEnabled=false;g2.drawImage(im2,0,0,672,672);
 zoom($('z1'),im1,C1+.5,C1+.5);
 if(pt){cross(g2,pt[0]*S2,pt[1]*S2,24,'#0a0');zoom($('z2'),im2,pt[0],pt[1]);}
 else{const g=$('z2').getContext('2d');g.fillStyle='#ddd';g.fillRect(0,0,256,256);}
 const n=Object.values(done).filter(r=>r.x_t02!=='').length;
 $('prog').textContent=`第 ${i+1}/${items.length} 个 · 已标 ${n} 对（够 30 对就可以停）`;
}
function load(){const it=items[i];const r=done[it.id];
 pt=r&&r.x_t02!==''?[+r.x_t02-it.o2[0]+.5,+r.y_t02-it.o2[1]+.5]:null;
 let k=0;const go=()=>{if(++k==2)draw();};im1.onload=go;im2.onload=go;
 im1.src=`img/${it.id}_t01.png`;im2.src=`img/${it.id}_t02.png`;$('msg').textContent=r?(r.x_t02===''?'之前跳过了':'之前标过'):'';}
async function save(skip){const it=items[i];
 if(!skip&&!pt){$('msg').textContent='先在右图上点一下';return;}
 const row={id:it.id,x_t01:it.x_t01,y_t01:it.y_t01,x_t02:skip?'':(it.o2[0]+pt[0]-.5).toFixed(1),y_t02:skip?'':(it.o2[1]+pt[1]-.5).toFixed(1),note:skip?'skip':''};
 const r=await fetch('save',{method:'POST',body:JSON.stringify(row)});
 if(!r.ok){$('msg').textContent='保存失败';return;}
 done[it.id]=row;if(i<items.length-1){i++;load();}else{draw();$('msg').textContent='全部看完了，可以关掉这个页面';}}
// 右图：按住拖动，绿十字跟着走，松手停
let drag=null;
const c2=$('c2'),z2=$('z2');
const at=e=>{const b=c2.getBoundingClientRect();pt=[(e.clientX-b.left)/S2,(e.clientY-b.top)/S2];draw();};
c2.onpointerdown=e=>{c2.setPointerCapture(e.pointerId);drag='c2';at(e);};
c2.onpointermove=e=>{if(drag==='c2')at(e);};
// 放大图：按住拖图像（十字固定在中心），只点不拖就跳到点的位置
z2.onpointerdown=e=>{if(!pt)return;z2.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,p:pt.slice(),moved:false};};
z2.onpointermove=e=>{if(!drag||drag==='c2')return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;if(drag.moved){pt=[drag.p[0]-dx/Z,drag.p[1]-dy/Z];draw();}};
z2.onpointerup=e=>{if(drag&&drag!=='c2'&&!drag.moved){const b=z2.getBoundingClientRect();pt=[drag.p[0]+(e.clientX-b.left-128)/Z,drag.p[1]+(e.clientY-b.top-128)/Z];draw();}drag=null;};
c2.onpointerup=()=>{drag=null;};
$('ok').onclick=()=>save(false);$('skip').onclick=()=>save(true);$('prev').onclick=()=>{if(i>0){i--;load();}};
document.onkeydown=e=>{const d={ArrowLeft:[-.5,0],ArrowRight:[.5,0],ArrowUp:[0,-.5],ArrowDown:[0,.5]}[e.key];
 if(d&&pt){pt=[pt[0]+d[0],pt[1]+d[1]];draw();e.preventDefault();}
 else if(e.key==='Enter')save(false);else if(e.key==='s'||e.key==='S')save(true);else if(e.key==='b'||e.key==='B')$('prev').onclick();};
(async()=>{const s=await (await fetch('state')).json();items=s.items;for(const r of s.rows)done[r.id]=r;
 i=items.findIndex(it=>!done[it.id]);if(i<0)i=0;load();})();
</script></html>'''


def serve(port, items, csv_path=CSV, open_browser=True):
    lock = threading.Lock()
    page = PAGE.replace('%C1%', str(C1)).replace('%C2%', str(C2)).encode('utf-8')

    class H_(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                return self._send(page, 'text/html; charset=utf-8')
            if self.path == '/state':
                return self._send(json.dumps({'items': items, 'rows': read_rows(csv_path)}).encode(), 'application/json')
            if self.path.startswith('/img/'):
                name = Path(self.path[5:]).name
                p = OUT / name
                if p.suffix == '.png' and p.is_file():
                    return self._send(p.read_bytes(), 'image/png')
            self._send(b'not found', 'text/plain', 404)

        def do_POST(self):
            if self.path != '/save':
                return self._send(b'not found', 'text/plain', 404)
            row = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            ids = {it['id'] for it in items}
            if row.get('id') not in ids:
                return self._send(b'bad id', 'text/plain', 400)
            with lock:
                rows = {r['id']: r for r in read_rows(csv_path)}
                rows[row['id']] = {k: row.get(k, '') for k in FIELDS}
                write_rows(rows.values(), csv_path)
            self._send(b'ok', 'text/plain')

    srv = ThreadingHTTPServer(('127.0.0.1', port), H_)
    url = f'http://127.0.0.1:{port}/'
    print(f'标注页面：{url}（标完关掉页面，在这里按 Ctrl+C 结束）', flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(prog='python -m railside.eval.pairs', description=__doc__.splitlines()[0])
    ap.add_argument('--n', type=int, default=40, help='候选点数')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--redo', action='store_true', help='重新挑点切图（已标的点不删）')
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args(argv)
    pj = OUT / 'pairs.json'
    items = prepare(a.n) if a.redo or not pj.is_file() else io.read_json(pj)['items']
    print(f'候选 {len(items)} 个点，切图在 {io.rel(OUT)}', flush=True)
    serve(a.port, items, open_browser=not a.no_browser)


if __name__ == '__main__':
    main()
