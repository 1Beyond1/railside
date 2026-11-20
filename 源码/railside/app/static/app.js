'use strict';
/* 走行部侧面巡检 前端。坐标：rect/xd 都是翻转显示图的像素（车头在右）；x 是趟坐标（不翻转）。 */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NS = 'http://www.w3.org/2000/svg';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 0) => (v == null || Number.isNaN(+v)) ? '—' : Number(v).toLocaleString('zh-CN', { maximumFractionDigits: d });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const enc = encodeURIComponent;
/* 离线浏览包（railside app --export）：接口结果、瓦片、局部图都预先导出成静态文件，双击 index.html 打开，只读。 */
const OFF = window.RS_STATIC || null;

async function api(path, opts) {
  if (OFF) {
    const key = path.split('?')[0];
    if (opts?.method || !(key in OFF.api)) {
      const e = new Error('离线浏览包只读，这个功能要用完整应用（双击“启动应用.cmd”）');
      e.status = 404;
      throw e;
    }
    return structuredClone(OFF.api[key]);
  }
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* 非 JSON */ }
  if (!r.ok) {
    const e = new Error(body?.error || body?.detail?.[0]?.msg || `请求失败（HTTP ${r.status}）`);
    e.status = r.status;
    throw e;
  }
  return body;
}
const post = (path, obj) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

const STATUS = { ok: '正常', alarm: '告警', missing: '缺失' };
const VERDICT = { pass: '通过', warn: '警告', fail: '失败' };
const ALARM = {
  ncc: '外观', appearance: '外观', diff_area: '外观差异', area: '面积变化', area_change: '面积变化',
  shift: '位移', shift_mm: '位移', dtheta: '转角', dtheta_deg: '转角', rotation: '转角', missing: '缺失', foreign: '异物',
};
const alarmLabel = a => ALARM[a] || a;
const FOUND = { parent: '父件定位', pixel: '像素匹配', topology: '拓扑补漏', missing: '未找到', template: '模板' };
const JOB = { running: '运行中', ok: '完成', partial: '部分完成', failed: '失败' };

const S = {
  trips: [], trip: null, info: null, ov: null, alarms: null, summary: null,
  byCode: new Map(), filter: 'all', sel: null, tok: 0, dtok: 0, jobsTimer: null, seenRunning: new Set(),
};

/* ---------------- 浏览器与叠加层 ---------------- */

const viewer = OpenSeadragon({
  element: $('#viewer'),
  prefixUrl: OFF ? 'static/vendor/openseadragon/images/' : '/static/vendor/openseadragon/images/',
  showNavigationControl: false, showNavigator: false,
  visibilityRatio: 1, constrainDuringPan: true, minZoomImageRatio: 0.95, maxZoomPixelRatio: 3,
  zoomPerScroll: 1.35, animationTime: 0.5, springStiffness: 9,
  imageLoaderLimit: 8, timeout: 60000,
  gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, flickEnabled: false },
});

const svg = document.createElementNS(NS, 'svg');
svg.setAttribute('class', 'ov hide-anchor');
const root = document.createElementNS(NS, 'g');
const gAnc = document.createElementNS(NS, 'g');
const gBox = document.createElementNS(NS, 'g');
const gSel = document.createElementNS(NS, 'g');
root.append(gAnc, gBox, gSel);
svg.append(root);
viewer.canvas.appendChild(svg);

function ready() { return S.info && viewer.world.getItemCount() > 0; }

function syncOverlay() {
  if (!ready()) return;
  const vp = viewer.viewport;
  const p0 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
  const p1 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(10000, 0));
  root.setAttribute('transform', `translate(${p0.x} ${p0.y}) scale(${(p1.x - p0.x) / 10000})`);
  syncCarbarView();
}
['animation', 'viewport-change', 'update-viewport', 'resize'].forEach(ev => viewer.addHandler(ev, syncOverlay));

viewer.addHandler('open', () => { jumpCar(1, true); syncOverlay(); });
viewer.addHandler('open-failed', e => showViewerEmpty(`长图打不开：${e.message || '未知错误'}`));

viewer.addHandler('canvas-click', e => {
  if (!e.quick) return;
  const code = e.originalEvent.target?.dataset?.code;
  if (code) { e.preventDefaultAction = true; selectPart(code, false); }
});

function showViewerEmpty(text) { const el = $('#viewerEmpty'); el.textContent = text; el.hidden = false; }

function rectEl(r, cls, code) {
  const el = document.createElementNS(NS, 'rect');
  el.setAttribute('x', r[0]); el.setAttribute('y', r[1]);
  el.setAttribute('width', Math.max(r[2], 1)); el.setAttribute('height', Math.max(r[3], 1));
  el.setAttribute('class', cls);
  if (code) el.dataset.code = code;
  return el;
}

function renderOverlay() {
  gBox.replaceChildren(); gAnc.replaceChildren(); gSel.replaceChildren(); S.byCode.clear();
  const ov = S.ov;
  const counts = ov?.counts || { ok: 0, alarm: 0, missing: 0 };
  $('#cntOk').textContent = counts.ok; $('#cntAlarm').textContent = counts.alarm; $('#cntMissing').textContent = counts.missing;
  $('#cntAnchor').textContent = ov?.anchors?.length || 0;
  const note = $('#overlayNote');
  note.hidden = !ov?.message; note.textContent = ov?.message || '';
  if (!ov) return;
  const order = { ok: 0, missing: 1, alarm: 2 };
  const frag = document.createDocumentFragment();
  for (const b of [...ov.boxes].sort((a, b) => order[a.status] - order[b.status])) {
    S.byCode.set(b.code, b);
    frag.append(rectEl(b.rect, `st-${b.status}`, b.code));
  }
  gBox.append(frag);
  const fa = document.createDocumentFragment();
  for (const a of ov.anchors || []) fa.append(rectEl(a.rect, a.ok === false ? 'anc bad' : 'anc'));
  gAnc.append(fa);
  markSel();
}

$('#layerCtrl').addEventListener('change', e => {
  const chip = e.target.closest('.lchip');
  if (chip) svg.classList.toggle(`hide-${chip.dataset.k}`, !e.target.checked);
});

/* 视图操作 */
function aspect() { const el = viewer.container; return el.clientWidth / Math.max(el.clientHeight, 1); }

function fitHeightAt(xd, immediately = false) {
  const H = S.info.height, w = H * aspect();
  viewer.viewport.fitBounds(viewer.viewport.imageToViewportRectangle(xd - w / 2, 0, w, H), immediately);
}

function viewImageRect() {
  const vp = viewer.viewport;
  return vp.viewportToImageRectangle(vp.getBounds(true));
}

function panToXd(xd, immediately = false) {
  const b = viewImageRect();
  if (b.width >= S.info.width * 0.5) return fitHeightAt(xd, immediately);
  viewer.viewport.panTo(viewer.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(xd, b.y + b.height / 2)), immediately);
}

function jumpCar(k, immediately = false) {
  const c = S.info?.cars.find(c => c.car === k);
  if (!c) return;
  const w = S.info.height * aspect();
  fitHeightAt(c.xd + c.w - w / 2 - w * 0.02, immediately); // 车头一端（右端）贴着视口右缘
}

function flyTo([x, y, w, h]) {
  const a = aspect();
  let tw = Math.max(w * 5, 1600), th = Math.max(h * 5, 1000);
  if (tw / th > a) th = tw / a; else tw = th * a;
  viewer.viewport.fitBounds(viewer.viewport.imageToViewportRectangle(x + w / 2 - tw / 2, y + h / 2 - th / 2, tw, th));
}

$('#btnZoomIn').onclick = () => { viewer.viewport.zoomBy(1.6); viewer.viewport.applyConstraints(); };
$('#btnZoomOut').onclick = () => { viewer.viewport.zoomBy(1 / 1.6); viewer.viewport.applyConstraints(); };
$('#btnFitH').onclick = () => { if (ready()) { const b = viewImageRect(); fitHeightAt(b.x + b.width / 2); } };
$('#btnHome').onclick = () => viewer.viewport.goHome();
$('#btnFull').onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen(); else $('.viewer-wrap').requestFullscreen();
};

/* 读数与悬停提示 */
const viewerEl = $('#viewer');
viewerEl.addEventListener('mousemove', e => {
  if (!ready()) return;
  const box = viewerEl.getBoundingClientRect();
  const p = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(e.clientX - box.left, e.clientY - box.top));
  const I = S.info, xd = Math.floor(p.x), y = Math.floor(p.y);
  const scale = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
  const ratio = scale >= 1 ? `${fmt(scale, 1)}:1` : `1:${fmt(1 / scale, 1)}`;
  if (xd < 0 || xd >= I.width || y < 0 || y >= I.height) {
    $('#readout').textContent = `比例 ${ratio}`;
  } else {
    const x = I.train_x1 - 1 - xd;
    const car = I.cars.find(c => x >= c.x0 && x < c.x1);
    $('#readout').textContent =
      `趟坐标 x ${x}  y ${y}   帧 ${Math.floor(x / I.frame_w) + 1} 列 ${x % I.frame_w}   第 ${car?.car ?? '—'} 节   比例 ${ratio}`;
  }
  const tip = $('#tip'), code = e.target?.dataset?.code;
  const b = code && S.byCode.get(code);
  if (b) {
    const wrap = $('.viewer-wrap').getBoundingClientRect();
    tip.innerHTML = `<span class="mono">${esc(b.code)}</span> · ${esc(b.type_name || b.type || '')} · ${STATUS[b.status]}`;
    tip.style.left = `${e.clientX - wrap.left + 14}px`; tip.style.top = `${e.clientY - wrap.top + 14}px`;
    tip.hidden = false;
  } else tip.hidden = true;
});
viewerEl.addEventListener('mouseleave', () => { $('#tip').hidden = true; });

/* ---------------- 车厢导航条 ---------------- */

function renderCarbar() {
  const I = S.info, bar = $('#carbar');
  const W = I.width;
  const dpr = window.devicePixelRatio || 1;
  $('#carbarImg').src = OFF ? `data/${enc(I.trip)}/overview.jpg` : `/api/trips/${enc(I.trip)}/overview.jpg?w=${Math.min(8000, Math.round(bar.clientWidth * dpr))}&h=${Math.round(56 * dpr)}`;
  $('#carbarCars').innerHTML = [...I.cars].sort((a, b) => a.xd - b.xd).map(c =>
    `<div class="car-seg" data-car="${c.car}" style="left:${c.xd / W * 100}%;width:${c.w / W * 100}%">` +
    `<span title="跳到第 ${c.car} 节车（键 ${c.car}）">第 ${c.car} 节</span></div>`).join('');
  $('#carbarTitle').textContent = `${I.trip} · ${I.cars.length} 节编组 · 车内 ${fmt(W)} px`;
  renderTicks();
}

function renderTicks() {
  const W = S.info?.width;
  const rows = (S.alarms?.rows || []).filter(r => r.rect);
  $('#carbarTicks').innerHTML = W ? rows.map(r =>
    `<i class="${r.status === 'missing' ? 'missing' : 'alarm'}" style="left:${(r.rect[0] + r.rect[2] / 2) / W * 100}%"></i>`).join('') : '';
}

function syncCarbarView() {
  const W = S.info.width, b = viewImageRect();
  const l = Math.max(0, b.x / W), r = Math.min(1, (b.x + b.width) / W);
  const v = $('#carbarView');
  v.style.left = `${l * 100}%`; v.style.width = `${Math.max(0, r - l) * 100}%`;
  const cx = S.info.train_x1 - 1 - (b.x + b.width / 2);
  const cur = S.info.cars.find(c => cx >= c.x0 && cx < c.x1)?.car;
  $$('.car-seg').forEach(s => s.classList.toggle('cur', +s.dataset.car === cur));
}

(() => {
  const bar = $('#carbar');
  let drag = false;
  const at = e => { const r = bar.getBoundingClientRect(); return (e.clientX - r.left) / r.width * S.info.width; };
  bar.addEventListener('pointerdown', e => {
    if (!ready()) return;
    const label = e.target.closest('.car-seg span');
    if (label) { jumpCar(+label.parentElement.dataset.car); return; }
    drag = true; bar.setPointerCapture(e.pointerId); panToXd(at(e));
  });
  bar.addEventListener('pointermove', e => { if (drag) panToXd(at(e), true); });
  bar.addEventListener('pointerup', () => { drag = false; });
})();

/* ---------------- 趟次 ---------------- */

async function loadTrips(keep) {
  let trips;
  try { ({ trips } = await api('/api/trips')); } catch (e) { showViewerEmpty(`读不到趟次列表：${e.message}`); return; }
  S.trips = trips;
  renderTripSwitch();
  if (keep && S.trip) return;
  if (!trips.length) {
    showViewerEmpty('还没有可浏览的趟次。到右侧「处理新趟」填原图目录，跑完拼接后这里会出现。');
    setTab('process');
    return;
  }
  const want = decodeURIComponent(location.hash.slice(1));
  await selectTrip(trips.some(t => t.trip === want) ? want : trips[0].trip);
}

function renderTripSwitch() {
  $('#tripSwitch').innerHTML = S.trips.map(t =>
    `<button data-trip="${esc(t.trip)}" class="${t.trip === S.trip ? 'on' : ''}" title="${t.n_cars} 节 · 质检 ${VERDICT[t.verdict] || '未运行'}">` +
    `<i class="dot ${esc(t.verdict || '')}"></i>${esc(t.trip)}</button>`).join('');
}
$('#tripSwitch').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b && b.dataset.trip !== S.trip) selectTrip(b.dataset.trip);
});

/* 离线包的长图：只导到 maxLevel 层，瓦片边长、重叠按倍数放大，图像坐标仍是原分辨率（零件框不用换算） */
function offlineTiles(trip) {
  const t = OFF.tiles[trip];
  return {
    width: t.width, height: t.height, tileSize: t.tileSize, tileOverlap: t.overlap, minLevel: 0, maxLevel: t.maxLevel,
    getTileUrl: (level, x, y) => `data/${enc(trip)}/tiles/${level}/${x}_${y}.jpg`,
  };
}

async function selectTrip(trip) {
  const tok = ++S.tok;
  S.trip = trip; S.sel = null; S.filter = 'all'; S.ov = S.alarms = S.summary = null;
  history.replaceState(null, '', `#${enc(trip)}`);
  renderTripSwitch(); closeDetail(); renderOverlay(); renderAlarms(); renderStatus();
  $('#summaryBody').innerHTML = '<div class="empty">读取中…</div>';
  try {
    S.info = await api(`/api/trips/${enc(trip)}`);
  } catch (e) { showViewerEmpty(e.message); return; }
  if (tok !== S.tok) return;
  $('#viewerEmpty').hidden = true;
  viewer.open(OFF ? offlineTiles(trip) : S.info.dzi.url);
  renderCarbar();
  const [ov, al, sm] = await Promise.allSettled([
    api(`/api/trips/${enc(trip)}/overlay`), api(`/api/trips/${enc(trip)}/alarms`), api(`/api/trips/${enc(trip)}/summary`)]);
  if (tok !== S.tok) return;
  S.ov = ov.value || { boxes: [], anchors: [], counts: { ok: 0, alarm: 0, missing: 0 }, message: `零件框读取失败：${ov.reason?.message}` };
  S.alarms = al.value || { rows: [], types: {}, message: `告警读取失败：${al.reason?.message}` };
  S.summary = sm.value || null;
  renderOverlay(); renderAlarms(); renderTicks(); renderStatus();
  renderSummary(sm.value, sm.reason);
}

function renderStatus() {
  const sm = S.summary, ov = S.ov, al = S.alarms;
  const chips = [];
  if (sm?.qc) {
    const v = sm.qc.verdict;
    chips.push(`<span class="chip ${v === 'pass' ? 'ok' : v === 'warn' ? 'warn' : 'bad'}">质检 <b>${VERDICT[v] || v}</b></span>`);
  } else if (sm) chips.push('<span class="chip">质检 <b>未运行</b></span>');
  if (S.info) chips.push(`<span class="chip opt">编组 <b>${S.info.cars.length} 节</b></span>`);
  if (sm?.align) {
    const r = sm.align.residual_px;
    const v = r && typeof r === 'object' ? r.p95 : r;
    chips.push(`<span class="chip opt">对齐残差 p95 <b>${fmt(v, 1)} px</b></span>`);
  } else if (ov?.template_trip && ov.template_trip === S.trip) chips.push('<span class="chip opt">对齐 <b>模板趟</b></span>');
  else if (sm) chips.push('<span class="chip opt">对齐 <b>未运行</b></span>');
  if (ov) chips.push(`<span class="chip">零件 <b>${ov.boxes.length}</b></span>`);
  if (al) {
    const n = al.rows.length;
    chips.push(`<span class="chip ${n ? 'warn' : ''}">告警 <b>${al.message && !n ? '—' : n}</b></span>`);
  }
  $('#statusStrip').innerHTML = chips.join('');
}

/* ---------------- 侧栏：页签 ---------------- */

function setTab(name) {
  $$('.side-tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  $$('.pane').forEach(p => p.classList.toggle('on', p.id === `pane-${name}`));
  if (name === 'process') refreshJobs();
}
$$('.side-tabs button').forEach(b => { b.onclick = () => { closeDetail(); setTab(b.dataset.tab); }; });

/* ---------------- 告警 ---------------- */

function carOf(xy) {
  if (!xy || !S.info) return null;
  const x = (xy[0] + xy[2]) / 2;
  return S.info.cars.find(c => x >= c.x0 && x < c.x1)?.car ?? null;
}

function metricLine(r) {
  const m = r.metrics || {}, out = [];
  if (m.ncc != null) out.push(`NCC ${fmt(m.ncc, 3)}`);
  if (m.shift_mm != null) out.push(`偏移 ${fmt(m.shift_mm, 1)} mm`);
  if (m.dtheta_deg != null) out.push(`Δθ ${fmt(m.dtheta_deg, 1)}°`);
  if (m.diff_area != null) out.push(`差异 ${fmt(m.diff_area)} px²`);
  if (m.area != null) out.push(`面积 ${fmt(m.area)}`);
  return out.join(' · ');
}

function renderAlarms() {
  const A = S.alarms, list = $('#alarmList'), filters = $('#alarmFilters');
  $('#alarmBadge').textContent = A?.rows.length || '';
  if (!A) { filters.innerHTML = ''; list.innerHTML = '<div class="empty">读取中…</div>'; return; }
  if (!A.rows.length) {
    filters.innerHTML = '';
    list.innerHTML = A.message
      ? `<div class="empty"><b>暂无告警数据</b><small>${esc(A.message)}</small></div>`
      : '<div class="empty"><b>本趟没有告警</b><small>全部零件的对比指标都在阈值内</small></div>';
    return;
  }
  const types = Object.entries(A.types).sort((a, b) => b[1] - a[1]);
  filters.innerHTML = [['all', A.rows.length], ...types].map(([t, n]) =>
    `<button data-f="${esc(t)}" class="${S.filter === t ? 'on' : ''}">${t === 'all' ? '全部' : esc(alarmLabel(t))}<em>${n}</em></button>`).join('');
  const rows = A.rows.filter(r => S.filter === 'all' || r.alarms.includes(S.filter));
  list.innerHTML = (A.message ? `<div class="note" style="margin:0 0 6px">${esc(A.message)}</div>` : '') + rows.map(r => {
    const car = carOf(r.xy);
    const sub = [car ? `第 ${car} 节` : null, metricLine(r) || null, r.rect ? null : '无位置'].filter(Boolean).join(' · ');
    return `<div class="item ${r.code === S.sel ? 'sel' : ''}" data-code="${esc(r.code)}">` +
      `<i class="bar ${r.status}"></i>` +
      `<div><div class="l1"><span class="mono">${esc(r.code)}</span><span>${esc(r.type_name || r.type || '')}</span></div>` +
      `<div class="l2">${esc(sub)}</div></div>` +
      `<div class="tags">${r.alarms.map(a => `<span class="tag ${a === 'missing' ? 'missing' : 'alarm'}">${esc(alarmLabel(a))}</span>`).join('')}</div></div>`;
  }).join('');
}
$('#alarmFilters').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b) { S.filter = b.dataset.f; renderAlarms(); }
});
$('#alarmList').addEventListener('click', e => {
  const it = e.target.closest('.item');
  if (it) selectPart(it.dataset.code, true);
});

/* ---------------- 零件详情 ---------------- */

function findRow(code) { return S.byCode.get(code) || S.alarms?.rows.find(r => r.code === code); }

function markSel() {
  $$('rect.sel', gBox).forEach(r => r.classList.remove('sel'));
  gSel.replaceChildren();
  $$('#alarmList .item').forEach(i => i.classList.toggle('sel', i.dataset.code === S.sel));
  if (!S.sel) return;
  const el = gBox.querySelector(`rect[data-code="${CSS.escape(S.sel)}"]`);
  if (el) { el.classList.add('sel'); gBox.appendChild(el); return; }
  const r = findRow(S.sel);
  if (r?.rect) gSel.append(rectEl(r.rect, `st-${r.status} sel`, null));
}

function selectPart(code, jump) {
  S.sel = code; markSel();
  const r = findRow(code);
  if (jump && r?.rect) flyTo(r.rect);
  openDetail(code, r);
}

function closeDetail() {
  $('#detail').hidden = true; S.dtok++;
  if (S.sel) { S.sel = null; markSel(); }
}
$('#detailClose').onclick = closeDetail;
$('#dLocate').onclick = () => { const r = S.sel && findRow(S.sel); if (r?.rect) flyTo(r.rect); };

async function openDetail(code, row) {
  const tok = ++S.dtok;
  $('#detail').hidden = false;
  $('#dCode').textContent = code;
  const st = row?.status || 'ok';
  $('#dStatus').className = `pill ${st}`; $('#dStatus').textContent = STATUS[st];
  $('#dLocate').hidden = !row?.rect;
  const body = $('#detailBody');
  if (row?.kind === 'foreign') { body.innerHTML = foreignDetail(row); return; }
  body.innerHTML = '<div class="empty">读取中…</div>';
  try {
    const d = await api(`/api/trips/${enc(S.trip)}/parts/${enc(code)}`);
    if (tok === S.dtok) body.innerHTML = partDetail(d);
  } catch (e) {
    if (tok === S.dtok) body.innerHTML = `<div class="empty"><b>读不到零件详情</b><small>${esc(e.message)}</small></div>`;
  }
}

function foreignDetail(r) {
  return `<div class="sec-title">异物</div><dl class="kv">` +
    `<dt>面积</dt><dd class="mono">${fmt(r.metrics?.area)}</dd>` +
    `<dt>位置</dt><dd class="mono">${r.xy ? `x ${fmt(r.xy[0])}–${fmt(r.xy[2])}  y ${fmt(r.xy[1])}–${fmt(r.xy[3])}` : '无（缺对齐映射）'}</dd>` +
    `<dt>车厢</dt><dd>${carOf(r.xy) ? `第 ${carOf(r.xy)} 节` : '—'}</dd></dl>` +
    '<div class="note">零件框外的大片差异，需人工复核。</div>';
}

const kv = rows => `<dl class="kv">${rows.filter(r => r[1] != null && r[1] !== '').map(([k, v, mono]) =>
  `<dt>${k}</dt><dd${mono ? ' class="mono"' : ''}>${v}</dd>`).join('')}</dl>`;

const METRICS = [
  ['ncc', 'NCC（外观相似度）', 3, ['ncc', 'appearance']],
  ['diff_area', '差异面积 px²', 0, ['diff_area', 'area', 'area_change', 'appearance']],
  ['shift_mm', '质心偏移 mm', 2, ['shift', 'shift_mm']],
  ['dtheta_deg', '转角 Δθ °', 2, ['dtheta', 'dtheta_deg', 'rotation']],
];

function partDetail(d) {
  const r = d.record, tp = d.template?.part || {}, inst = d.instance || {}, c = d.compare;
  const crop = side => OFF ? `data/${enc(S.trip)}/crops/${enc(r.code)}_${side}.jpg`
    : `/api/trips/${enc(S.trip)}/parts/${enc(r.code)}/crop.jpg?side=${side}`;
  const fig = (ok, side, cap, sub, why) => `<figure>${ok ? `<img src="${crop(side)}" alt="${cap}" loading="lazy">` : `<div class="noimg">${why}</div>`}` +
    `<figcaption><span>${cap}</span><span class="mono">${esc(sub)}</span></figcaption></figure>`;
  let h = '<div class="pair">' +
    fig(d.crops.template, 'template', '模板', d.template ? `${d.template.version} · ${d.template.trip}` : '—', '模板里没有这个零件') +
    fig(d.crops.trip, 'trip', '本趟', S.trip, r.status === 'missing' ? '未找到（缺失候选）' : '没有位置') + '</div>';

  h += '<div class="sec-title">对比指标</div>';
  if (c) {
    const al = new Set(r.alarms);
    h += '<div class="metrics">' + METRICS.map(([k, label, dg, keys]) =>
      `<div class="metric ${keys.some(x => al.has(x)) ? 'hit' : ''}"><small>${label}</small><b>${fmt(c[k], dg)}</b></div>`).join('') + '</div>';
    h += r.alarms.length
      ? `<div class="note">告警：${r.alarms.map(a => `<span class="tag ${a === 'missing' ? 'missing' : 'alarm'}">${esc(alarmLabel(a))}</span>`).join(' ')}</div>`
      : '<div class="note">全部指标在阈值内。</div>';
  } else {
    h += `<div class="note">${r.status === 'missing' ? '零件缺失，没有对比指标。' : '本趟还没有对比结果（compare.json 不存在）。'}</div>`;
  }

  const xy = r.xy;
  const size = tp.size_mm ? `${fmt(tp.size_mm[0], 1)} × ${fmt(tp.size_mm[1], 1)} mm` : null;
  const parent = r.parent ? (S.byCode.has(r.parent) ? `<a href="#" class="mono" data-goto="${esc(r.parent)}">${esc(r.parent)}</a>` : `<span class="mono">${esc(r.parent)}</span>`) : '—';
  h += '<div class="sec-title">零件</div>' + kv([
    ['类型', `${esc(r.type_name || '—')}${r.type && r.type !== r.type_name ? ` <span class="mono" style="color:var(--faint)">${esc(r.type)}</span>` : ''}`],
    ['父件', parent],
    ['车 / 区', r.car != null ? `第 ${r.car} 节${r.zone ? ` · ${esc(r.zone)}` : ''}` : (carOf(xy) ? `第 ${carOf(xy)} 节` : null)],
    ['位置（趟）', xy ? `x ${fmt(xy[0])}–${fmt(xy[2])}<br>y ${fmt(xy[1])}–${fmt(xy[3])}` : '—', true],
    ['外框', xy ? `${fmt(xy[2] - xy[0])} × ${fmt(xy[3] - xy[1])} px` : null, true],
    ['尺寸', size, true],
    ['面积', tp.area_px != null ? `${fmt(tp.area_px)} px²` : (inst.area_mm2 != null ? `${fmt(inst.area_mm2, 1)} mm²` : null), true],
    ['主方向', tp.angle_deg != null ? `${fmt(tp.angle_deg, 1)}°` : (inst.angle_deg != null ? `${fmt(inst.angle_deg, 1)}°` : null), true],
    ['模板质心', tp.centroid_uv ? `u ${fmt(tp.centroid_uv[0], 1)}  v ${fmt(tp.centroid_uv[1], 1)}` : null, true],
    ['找到方式', inst.found_by ? esc(FOUND[inst.found_by] || inst.found_by) : null],
    ['匹配分', inst.match_score != null ? fmt(inst.match_score, 3) : null, true],
    ['来源帧', inst.src_frames ? esc(JSON.stringify(inst.src_frames)) : null, true],
  ]);

  const hs = d.history;
  h += '<div class="sec-title">跨趟历史</div>';
  if (hs?.rows?.length) {
    h += '<table class="t"><thead><tr><th>趟次</th><th>状态</th><th>找到</th><th class="num">NCC</th><th class="num">偏移mm</th><th class="num">Δθ°</th></tr></thead><tbody>' +
      hs.rows.map(x => `<tr class="${x.trip === S.trip ? 'cur' : ''}"><td>${esc(x.trip)}</td>` +
        `<td>${x.status ? `<span class="pill ${x.status}">${STATUS[x.status] || esc(x.status)}</span>` : '—'}</td>` +
        `<td>${esc(FOUND[x.found_by] || x.found_by || '—')}</td><td class="num">${fmt(x.ncc, 3)}</td>` +
        `<td class="num">${fmt(x.shift_mm, 2)}</td><td class="num">${fmt(x.dtheta_deg, 2)}</td></tr>`).join('') + '</tbody></table>';
  } else h += '<div class="note">没有其他趟的记录。</div>';
  if (hs) h += `<div class="note">数据来源：${esc(hs.source)}${hs.note ? `（${esc(hs.note)}）` : ''}</div>`;
  return h;
}
$('#detailBody').addEventListener('click', e => {
  const a = e.target.closest('[data-goto]');
  if (a) { e.preventDefault(); selectPart(a.dataset.goto, true); }
});

/* ---------------- 质检与对齐 ---------------- */

const empty = (t, s) => `<div class="empty"><b>${t}</b><small>${s}</small></div>`;

function renderSummary(sm, err) {
  const el = $('#summaryBody');
  if (!sm) { el.innerHTML = empty('读不到概况', esc(err?.message || '')); return; }
  let h = '<div class="sec-title">S1 收录质检</div>';
  if (sm.qc) {
    const q = sm.qc, v = q.verdict;
    h += `<div class="verdict ${esc(v)}"><span class="big">${VERDICT[v] || esc(v)}</span>` +
      `<small>${fmt(q.frames)} 帧 · 缺帧 ${q.missing.length} · 有标记的帧 ${q.flagged.length}${q.readonly ? ' · 原图只读' : ''}</small></div>`;
    if (q.reasons.length) h += `<ul class="reasons">${q.reasons.map(r => `<li>${esc(typeof r === 'string' ? r : JSON.stringify(r))}</li>`).join('')}</ul>`;
    else h += '<div class="note">没有异常理由：无缺帧、无过暗过曝、无模糊。</div>';
    if (q.missing.length) h += `<div class="note">缺帧：<span class="mono">${esc(q.missing.join(', '))}</span></div>`;
    if (q.flagged.length) {
      h += '<table class="t" style="margin-top:8px"><thead><tr><th>帧</th><th>标记</th></tr></thead><tbody>' +
        q.flagged.slice(0, 30).map(f => `<tr><td>${f.f}</td><td>${esc(f.flags.join(', '))}</td></tr>`).join('') + '</tbody></table>';
    }
  } else h += empty('未运行', 'qc.json 不存在，跑 railside ingest');

  h += '<div class="sec-title">S2 拼接</div>';
  if (sm.stitch) {
    const s = sm.stitch;
    h += kv([
      ['车内范围', `x ${fmt(s.train_x0)} – ${fmt(s.train_x1)}`, true],
      ['车头', `x ${fmt(s.head_x)}`, true],
      ['车内帧', s.train_frames ? `${s.train_frames[0]} – ${s.train_frames[1]}` : null, true],
      ['接缝', `${s.n_seams} 处 · 异常 <b style="color:${s.bad_seams ? 'var(--missing)' : 'var(--ok)'}">${s.bad_seams ?? '—'}</b>`],
    ]);
    h += '<table class="t" style="margin-top:8px"><thead><tr><th>车</th><th class="num">x0</th><th class="num">x1</th><th class="num">宽 px</th></tr></thead><tbody>' +
      s.cars.map(c => `<tr data-car="${c.car}"><td>第 ${c.car} 节</td><td class="num">${fmt(c.x0)}</td><td class="num">${fmt(c.x1)}</td><td class="num">${fmt(c.width_px)}</td></tr>`).join('') +
      '</tbody></table><div class="note">点一行跳到该节车。车号按过车先后，1 号车在右（车头）。</div>';
  }

  h += '<div class="sec-title">S5 对齐</div>';
  if (sm.align) {
    const a = sm.align, r = a.residual_px;
    const res = r && typeof r === 'object' ? `中位 ${fmt(r.median, 2)} · p95 ${fmt(r.p95, 2)} · 最大 ${fmt(r.max, 2)} px` : `${fmt(r, 2)} px`;
    h += kv([
      ['模板版本', esc(a.template_ver), true],
      ['定位靶', a.anchors_total != null ? `找到 ${a.anchors_found} / ${a.anchors_total}` : `${a.n_anchors} 个`],
      ['靶点残差', res, true],
      ['横向比例', a.scale_range ? `${fmt(a.scale_range[0], 4)} – ${fmt(a.scale_range[1], 4)}` : null, true],
      ['横向偏移', a.offset_range_px ? `${fmt(a.offset_range_px[0], 1)} – ${fmt(a.offset_range_px[1], 1)} px` : null, true],
      ['加密块', a.blocks_total != null ? `${a.blocks_used} / ${a.blocks_total}` : null, true],
      ['映射采样点', a.n_g, true],
      ['二段质检', a.qc2_flagged != null ? `${a.qc2_flagged} 处标记` : null],
    ]);
  } else if (S.ov?.template_trip === S.trip) {
    h += '<div class="note">本趟是模板趟（u = x − head_x），不需要对齐。</div>';
  } else h += empty('未运行', 'align.json 不存在，跑 railside align');

  h += '<div class="sec-title">S3 标定 · S6 识别 · S8 对比</div>';
  h += kv([
    ['标定', sm.calib ? `${fmt(sm.calib.sx_min, 3)}–${fmt(sm.calib.sx_max, 3)} mm/px · ${sm.calib.n_refs} 个参照` : '未运行'],
    ['分割', sm.seg ? `${fmt(sm.seg.n_dets)} 个检测 · ${esc(typeof sm.seg.model === 'object' ? (sm.seg.model?.name || '') : (sm.seg.model || ''))}` : '未运行'],
    ['身份', sm.parts ? `${fmt(sm.parts.n)} 个实例 · ${Object.entries(sm.parts.found_by).map(([k, n]) => `${esc(FOUND[k] || k)} ${n}`).join(' / ')}` : '未运行'],
    ['对比', sm.compare ? `${fmt(sm.compare.n_items)} 个零件 · 告警 ${sm.compare.n_alarm_parts} · 异物 ${sm.compare.n_foreign}` : '未运行'],
  ]);

  h += '<div class="sec-title">流水线文件</div><div class="files">' +
    sm.files.map(f => `<div class="${f.exists ? 'yes' : 'no'}" title="${esc(f.file)}"><i></i>${esc(f.step)}</div>`).join('') + '</div>';
  el.innerHTML = h;
}
$('#summaryBody').addEventListener('click', e => {
  const tr = e.target.closest('tr[data-car]');
  if (tr) jumpCar(+tr.dataset.car);
});

/* ---------------- 处理新趟 ---------------- */

$('#runForm').addEventListener('submit', async e => {
  e.preventDefault();
  const f = e.target, btn = f.querySelector('button'), msg = $('#runMsg');
  msg.textContent = ''; btn.disabled = true;
  try {
    await post('/api/run', { src: f.src.value, trip: f.trip.value,
      template: f.elements.namedItem('template').value,
      weights: f.elements.namedItem('weights').value.trim() || null });
    refreshJobs();
  } catch (err) {
    msg.textContent = err.message;
  } finally { btn.disabled = false; }
});

async function refreshJobs() {
  clearTimeout(S.jobsTimer);
  let jobs = [];
  try { ({ jobs } = await api('/api/jobs')); } catch { /* 下次再试 */ }
  const runs = jobs.filter(j => j.kind === 'run');
  const el = $('#jobList');
  el.innerHTML = runs.length ? runs.map(j =>
    `<div class="job"><div class="job-head"><span class="pill ${j.status}">${JOB[j.status]}</span>` +
    `<span class="mono">${esc(j.trip)}</span><span class="meta">${fmt(j.elapsed, 1)} s${j.rc != null ? ` · 退出码 ${j.rc}` : ''}</span></div>` +
    `<pre>${esc(j.log.join('\n') || '（还没有输出）')}</pre></div>`).join('')
    : '<div class="empty"><b>还没有任务</b><small>填上面的表单开始处理一趟新数据</small></div>';
  $$('#jobList pre').forEach(p => { p.scrollTop = p.scrollHeight; });
  let finished = false;
  for (const j of runs) {
    if (j.status === 'running') S.seenRunning.add(j.id);
    else if (S.seenRunning.delete(j.id) && ['ok', 'partial'].includes(j.status)) finished = true;
  }
  if (finished) loadTrips(true);
  if (runs.some(j => j.status === 'running')) S.jobsTimer = setTimeout(refreshJobs, 1500);
}

/* ---------------- 智能体对话 ---------------- */

function addMsg(cls, text, who) {
  const m = document.createElement('div');
  m.className = `msg ${cls}`;
  m.innerHTML = (who ? `<span class="who">${esc(who)}</span>` : '') + `<span class="txt">${esc(text)}</span>`;
  $('#chatLog').append(m);
  $('#chatLog').scrollTop = $('#chatLog').scrollHeight;
  return m;
}
addMsg('sys', '智能体会调用流水线各步、查库、汇总告警，写成文字结论。对话针对顶栏选中的趟次。');

$('#chatText').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('#chatForm').requestSubmit(); }
});

$('#chatForm').addEventListener('submit', async e => {
  e.preventDefault();
  const ta = $('#chatText'), text = ta.value.trim();
  if (!text || !S.trip) return;
  ta.value = '';
  const btn = $('#chatSend'); btn.disabled = true;
  addMsg('me', text, `我 · ${S.trip}`);
  const pend = addMsg('bot pending', '思考中…', '智能体');
  try {
    let j = await post('/api/agent', { trip: S.trip, message: text });
    while (j.status === 'running') {
      await sleep(1200);
      j = await api(`/api/jobs/${enc(j.id)}`);
      const last = j.log.filter(Boolean).slice(-1)[0];
      pend.querySelector('.txt').textContent = `思考中… ${fmt(j.elapsed, 0)} s${last ? `\n${last.slice(0, 160)}` : ''}`;
    }
    pend.classList.remove('pending');
    if (j.status === 'failed') {
      pend.classList.add('err');
      pend.querySelector('.txt').textContent = `智能体出错（退出码 ${j.rc ?? '—'}）：\n${j.reply}`;
    } else pend.querySelector('.txt').textContent = j.reply;
  } catch (err) {
    pend.remove();
    addMsg('sys err', err.status === 501 ? `智能体还没接上：${err.message}` : `发送失败：${err.message}`);
  } finally {
    btn.disabled = false;
    $('#chatLog').scrollTop = $('#chatLog').scrollHeight;
  }
});

/* ---------------- 键盘 ---------------- */

document.addEventListener('keydown', e => {
  if (e.target.closest('input, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') closeDetail();
  else if (!ready()) return;
  else if (/^[1-9]$/.test(e.key)) jumpCar(+e.key);
  else if (e.key === 'h' || e.key === 'H') $('#btnFitH').click();
  else if (e.key === '0') viewer.viewport.goHome();
  else if (e.key === '+' || e.key === '=') $('#btnZoomIn').click();
  else if (e.key === '-') $('#btnZoomOut').click();
});

if (OFF) {   // 离线包没有后端：不显示处理新趟和智能体
  $$('.side-tabs button[data-tab="process"], .side-tabs button[data-tab="chat"], #pane-process, #pane-chat')
    .forEach(el => { el.hidden = true; });
  $('.brand-text small').textContent = '离线浏览包 · 只读';
}
loadTrips();

