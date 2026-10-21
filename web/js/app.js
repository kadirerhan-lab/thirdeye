// ThirdEye arayüz denetleyicisi.
import { preprocessAsync, detectAll, noduleFromTrack, noduleFromManual, syntheticTrainingCase, PATCH } from './pipeline.js';
import { segmentFromSeed, ellipseMask, contourOf, DEFAULT_DETECT } from './detect.js';
import { featureRows } from './features.js';
import { TIRADS, TR_INFO, scoreTirads, suggestTirads, recommendation } from './tirads.js';
import { fuseRisk, modelReliability } from './risk.js';
import { generateCase } from './synth.js';
import { hasTF, trainModel, loadModel, deleteModel, exportModel, predict } from './model.js';

// Gerçek klinik veriyle (Stanford) ön-eğitilmiş model sitede varsa, yerel model yokken kullanılır.
async function loadPretrained() {
  if (!hasTF()) return null;
  try {
    const res = await fetch('models/stanford/meta.json', { cache: 'no-cache' });
    if (!res.ok) return null;
    const meta = await res.json();
    const model = await window.tf.loadLayersModel('models/stanford/model.json');
    return { model, meta };
  } catch { return null; }
}
const modelName = (meta) => (meta?.origin === 'pretrained' ? `Stanford modeli v${meta.version}` : `Yerel model v${meta.version}`);
import * as store from './store.js';

const $ = (id) => document.getElementById(id);
const tick = () => new Promise((r) => setTimeout(r, 0));
// Dosya indirme yalnızca gömülü (sandbox) önizlemelerde kapalıdır.
// Telefon ve tabletlerde daha küçük çözünürlük ve daha az kare: işlem süresi ve bellek kabaca yarıya iner.
const MOBILE = (typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches) || /Android|iPhone|iPad|Mobi/i.test(navigator.userAgent);
const MAX_W = MOBILE ? 480 : 640;
const CAN_DOWNLOAD = !/claude(usercontent)?\.(ai|com)$/.test(location.hostname);
const fmtPct = (p, d = 1) => `%${(p * 100).toFixed(d).replace('.', ',')}`;
const fmtNum = (v, d = 1) => v.toFixed(d).replace('.', ',');
const TRACK_COLORS = ['#f0ae2e', '#5fb3e8', '#c58af0', '#6fd39a', '#f07a6a'];
const SENS = [
  { label: 'çok düşük', hypo: 0.7 }, { label: 'düşük', hypo: 0.74 }, { label: 'orta', hypo: 0.78 },
  { label: 'yüksek', hypo: 0.82 }, { label: 'çok yüksek', hypo: 0.86 },
];

const state = {
  src: null, prep: null, mmPerPx: 0.1, det: null, tracks: [], selTrack: null, nodule: null,
  suggestion: null, sel: null, edited: new Set(), tr: null, rec: null,
  model: null, meta: null, prediction: null, fused: null, truth: null,
  cur: 0, playing: false, timer: null, view: 'clean', drag: null, step: 'source',
};

// ---------- yardımcılar ----------
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function busy(text, p = null) {
  const b = $('busy');
  if (text == null) { b.hidden = true; return; }
  b.hidden = false;
  $('busyText').textContent = text;
  $('busyBar').style.width = p == null ? '100%' : `${Math.round(p * 100)}%`;
}
function hint(text, warn = false) { const h = $('hint'); h.textContent = text; h.classList.toggle('is-warn', warn); }

function download(name, blob) {
  const a = el('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---------- adım gezinme ----------
function goStep(step) {
  state.step = step;
  for (const b of document.querySelectorAll('.step')) {
    const on = b.dataset.step === step;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on);
  }
  for (const p of document.querySelectorAll('.panel')) p.hidden = p.id !== `panel-${step}`;
}
function markSteps() {
  const done = { source: !!state.prep, nodule: !!state.nodule, tirads: !!state.tr, score: !!state.fused, learn: !!state.meta };
  for (const b of document.querySelectorAll('.step')) b.classList.toggle('is-done', !!done[b.dataset.step]);
}

// ---------- kaynak yükleme ----------
async function extractVideoFrames(file, n) {
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = url;
  const once = (ev) => new Promise((res, rej) => {
    const t = setTimeout(() => rej(new Error('Video zaman aşımına uğradı.')), 15000);
    v.addEventListener(ev, () => { clearTimeout(t); res(); }, { once: true });
    v.addEventListener('error', () => { clearTimeout(t); rej(new Error('Bu video biçimi tarayıcıda açılamadı. MP4 (H.264) veya WebM deneyin.')); }, { once: true });
  });
  await once('loadedmetadata');
  if (!Number.isFinite(v.duration)) { // MediaRecorder ile kaydedilmiş WebM'lerde süre bilinmeyebilir
    v.currentTime = 1e6; await once('seeked'); v.currentTime = 0; await once('seeked');
  }
  const dur = v.duration;
  const scale = Math.min(1, MAX_W / v.videoWidth);
  const W = Math.round(v.videoWidth * scale), H = Math.round(v.videoHeight * scale);
  const c = el('canvas', { width: W, height: H });
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const frames = [];
  const count = Math.max(1, Math.min(n, Math.floor(dur * 30)));
  for (let i = 0; i < count; i++) {
    v.currentTime = Math.min(dur - 0.001, ((i + 0.5) * dur) / count);
    await once('seeked');
    ctx.drawImage(v, 0, 0, W, H);
    frames.push(ctx.getImageData(0, 0, W, H));
    busy(`Videodan kare alınıyor (${i + 1}/${count})`, (i + 1) / count);
  }
  URL.revokeObjectURL(url);
  return frames;
}

async function decodeImages(files) {
  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, 'tr', { numeric: true }));
  let W = 0, H = 0;
  const frames = [];
  for (const f of sorted) {
    const bmp = await createImageBitmap(f);
    if (!W) { const s = Math.min(1, MAX_W / bmp.width); W = Math.round(bmp.width * s); H = Math.round(bmp.height * s); }
    const c = el('canvas', { width: W, height: H });
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0, W, H);
    frames.push(ctx.getImageData(0, 0, W, H));
  }
  return frames;
}

async function loadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  try {
    busy('Dosya okunuyor…', 0.02);
    const video = files.find((f) => f.type.startsWith('video/') || /\.(mp4|webm|mov|m4v|ogv)$/i.test(f.name));
    const images = files.filter((f) => f.type.startsWith('image/'));
    let frames, name;
    if (video) { frames = await extractVideoFrames(video, +$('framesSelect').value); name = video.name; }
    else if (images.length) { frames = await decodeImages(images); name = images.length > 1 ? `${images.length} görüntü` : images[0].name; }
    else throw new Error('Desteklenen bir video veya görüntü bulunamadı.');
    state.truth = null;
    await ingest(frames, { name, kind: video ? 'video' : 'image' });
  } catch (e) {
    busy(null); hint(e.message, true);
  }
}

async function loadSample(kind) {
  busy('Sentetik tarama üretiliyor…', 0.05);
  await tick();
  const seed = kind === 'malignant' ? 7 : kind === 'benign' ? 11 : 100 + Math.floor(Math.random() * 1e6);
  const c = generateCase({ seed, preset: kind === 'random' ? 'random' : kind, frames: +$('framesSelect').value, w: MAX_W, h: Math.round(MAX_W * 0.75) });
  state.truth = c.truth;
  const names = { malignant: 'Sentetik · şüpheli görünüm', benign: 'Sentetik · iyi huylu görünüm', random: `Sentetik · rastgele #${seed}` };
  await ingest(c.frames, { name: names[kind], kind: 'synthetic', mmPerPx: c.mmPerPx, seed });
}

async function ingest(frames, src) {
  stopPlay();
  Object.assign(state, { src, prep: null, det: null, tracks: [], selTrack: null, nodule: null, suggestion: null, sel: null, tr: null, prediction: null, fused: null, cur: 0 });
  state.edited.clear();
  busy('Ön işleme: görüntü alanı ve ekran işaretleri…', 0.1);
  await tick();
  const prep = await preprocessAsync(frames, { clean: $('optClean').checked, onProgress: (p) => busy('Ön işleme: görüntü alanı ve ekran işaretleri…', 0.1 + 0.2 * p) });
  state.prep = prep;
  if (src.mmPerPx) {
    state.mmPerPx = src.mmPerPx;
    $('depthInput').value = ((src.mmPerPx * prep.h) / 10).toFixed(1);
  } else state.mmPerPx = (+$('depthInput').value * 10) / prep.h;
  $('viewerEmpty').hidden = true;
  const n = prep.cleaned.length;
  $('frameSlider').max = n - 1; $('frameSlider').disabled = n < 2; $('playBtn').disabled = n < 2;
  renderSourceFacts();
  render();
  await runDetection();
  goStep('nodule');
}

async function runDetection() {
  const { prep } = state;
  if (!prep) return;
  const cfg = { ...DEFAULT_DETECT, hypo: SENS[+$('sensSlider').value].hypo };
  busy('Nodül adayları taranıyor…', 0);
  state.det = await detectAll(prep.grays, prep.w, prep.h, cfg, (p) => busy(`Nodül adayları taranıyor (${Math.round(p * 100)}%)`, p));
  state.tracks = state.det.tracks;
  if (state.tracks.length) {
    busy('Nodül sınırı belirleniyor…', 0.95);
    await selectTrack(state.tracks[0]);
    busy(null);
    hint('En olası aday seçildi. Başka bir adaya geçebilir, nodüle tıklayabilir veya sürükleyerek elips çizebilirsiniz.');
  } else {
    busy(null);
    state.nodule = null;
    renderCandidates(); renderAll();
    hint('Otomatik aday bulunamadı. Nodülün içine tıklayın veya sürükleyerek elips çizin.', true);
  }
}

function renderSourceFacts() {
  const { prep, src } = state;
  $('fSource').textContent = src ? src.name : '–';
  $('fFrames').textContent = prep ? String(prep.cleaned.length) : '–';
  $('fSize').textContent = prep ? `${prep.w}×${prep.h} px` : '–';
  $('fScale').textContent = prep ? `${fmtNum(state.mmPerPx, 3)} mm/px` : '–';
  $('fOverlay').textContent = prep ? (prep.overlayMasks ? `${fmtNum(prep.overlayRatio * 100, 2)}% piksel` : 'kapalı') : '–';
}

// ---------- nodül seçimi ----------
let selectToken = 0;
async function selectTrack(t) {
  const { prep } = state;
  const token = ++selectToken;
  state.selTrack = t;
  const nod = await noduleFromTrack(t, prep.grays, prep.w, prep.h, state.mmPerPx, state.det.preCache);
  if (token !== selectToken) return; // bu arada başka bir aday seçildi
  state.nodule = nod;
  if (!state.nodule) { hint('Bu aday için özellik çıkarılamadı.', true); return; }
  state.cur = state.nodule.bestFrame;
  afterNodule();
}

function manualNodule(mask) {
  const { prep } = state;
  state.selTrack = null;
  const pc = state.det?.preCache;
  state.nodule = noduleFromManual(state.cur, mask, prep.grays, prep.w, prep.h, state.mmPerPx, pc);
  if (!state.nodule) { hint('Bu bölgeden özellik çıkarılamadı; daha büyük bir alan seçin.', true); return; }
  hint(`Elle işaretlenen nodül ${state.nodule.frameMasks.size} kareye yayıldı.`);
  state.cur = state.nodule.bestFrame;
  afterNodule();
}

function afterNodule() {
  const f = state.nodule.features;
  state.suggestion = suggestTirads(f);
  state.sel = structuredClone(state.suggestion.sel);
  state.edited.clear();
  renderCandidates();
  renderFeatures();
  renderTiradsForm();
  updateScore();
  $('addCaseBtn').disabled = !$('labelSelect').value;
  renderTruthHint();
}

// ---------- skor ----------
function updateScore() {
  if (!state.nodule || !state.sel) return;
  const f = state.nodule.features;
  state.tr = scoreTirads(state.sel);
  state.rec = recommendation(state.tr.category, f.maxDiamMm);
  state.prediction = null;
  if (state.model && state.meta && hasTF()) {
    try { state.prediction = predict(state.model, state.meta, f, state.nodule.patch.data); } catch (e) { console.warn(e); }
  }
  state.fused = fuseRisk(state.tr.category, state.prediction?.p ?? null, state.meta);
  renderTiradsResult();
  renderScore();
  render();
  markSteps();
}

// ---------- görüntüleyici ----------
const view = $('view');
const vctx = view.getContext('2d');
const fcan = document.createElement('canvas');
const fctx = fcan.getContext('2d');
const tintCache = new Map();

function render() {
  const { prep } = state;
  if (!prep) return;
  const { w, h } = prep;
  if (view.width !== w || view.height !== h) { view.width = w; view.height = h; fcan.width = w; fcan.height = h; tintCache.clear(); }
  const f = state.cur;
  const showOrig = state.view === 'orig';
  const src = showOrig ? prep.orig[f] : prep.cleaned[f];
  fctx.putImageData(new ImageData(src.data, w, h), 0, 0);
  vctx.drawImage(fcan, 0, 0);
  const lw = Math.max(1, w / Math.max(200, view.clientWidth || w));

  if (showOrig && prep.overlayMasks) {
    if (!tintCache.has(f)) {
      const m = prep.overlayMasks[f];
      const img = new ImageData(w, h);
      for (let i = 0; i < w * h; i++) if (m[i]) { img.data[i * 4] = 255; img.data[i * 4 + 1] = 40; img.data[i * 4 + 2] = 60; img.data[i * 4 + 3] = 170; }
      const c = el('canvas', { width: w, height: h });
      c.getContext('2d').putImageData(img, 0, 0);
      tintCache.set(f, c);
    }
    vctx.drawImage(tintCache.get(f), 0, 0);
  }

  if ($('optContours').checked) {
    state.tracks.forEach((t, k) => {
      if (state.nodule && t === state.selTrack) return;
      const it = t.items.find((i) => i.frame === f);
      if (!it) return;
      it.cand._poly ||= contourOf(it.cand, 72);
      strokePoly(it.cand._poly, TRACK_COLORS[k % TRACK_COLORS.length], 1.2 * lw, [4 * lw, 3 * lw]);
      label(`A${t.id}`, it.cand.x0, it.cand.y0 - 4 * lw, TRACK_COLORS[k % TRACK_COLORS.length], lw);
    });
  }
  const nod = state.nodule;
  if (nod) {
    const cm = nod.frameMasks.get(f);
    if (f === nod.bestFrame && $('optContours').checked) strokePoly(nod.features.contour, '#f0ae2e', 2 * lw);
    else if (cm && $('optContours').checked) { cm._poly ||= contourOf(cm, 72); strokePoly(cm._poly, '#f0ae2e', 1.6 * lw); }
    if (f === nod.bestFrame && $('optSaliency').checked && state.prediction) drawSaliency(nod.patch.box, state.prediction.saliency);
    if (f === nod.bestFrame && $('optMeasure').checked) drawMeasure(nod.features, lw);
  }
  if (state.drag?.moved) {
    const d = state.drag;
    vctx.save();
    vctx.strokeStyle = '#5fb3e8'; vctx.lineWidth = 1.5 * lw; vctx.setLineDash([5 * lw, 4 * lw]);
    vctx.beginPath();
    vctx.ellipse((d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2, Math.abs(d.x1 - d.x0) / 2, Math.abs(d.y1 - d.y0) / 2, 0, 0, 2 * Math.PI);
    vctx.stroke(); vctx.restore();
  }
  $('frameSlider').value = f;
  $('frameLabel').textContent = `${f + 1} / ${prep.cleaned.length}`;
  const badge = $('frameBadge');
  badge.hidden = !nod;
  if (nod) badge.textContent = f === nod.bestFrame ? 'En büyük kesit · ölçüm karesi' : nod.frameMasks.has(f) ? 'Nodül izleniyor' : 'Nodül bu karede yok';
}

function strokePoly(pts, color, width, dash = []) {
  if (!pts || pts.length < 3) return;
  vctx.save();
  vctx.strokeStyle = color; vctx.lineWidth = width; vctx.setLineDash(dash);
  vctx.beginPath();
  pts.forEach((p, i) => (i ? vctx.lineTo(p.x, p.y) : vctx.moveTo(p.x, p.y)));
  vctx.closePath(); vctx.stroke(); vctx.restore();
}
function label(text, x, y, color, lw) {
  vctx.save();
  vctx.font = `${Math.round(11 * lw)}px "IBM Plex Mono", monospace`;
  vctx.fillStyle = color; vctx.fillText(text, x, Math.max(10 * lw, y));
  vctx.restore();
}
function drawMeasure(f, lw) {
  const { x0, y0, x1, y1 } = f.bbox;
  const cx = f.centroid.x, cy = f.centroid.y;
  vctx.save();
  vctx.strokeStyle = 'rgba(240,240,240,0.9)'; vctx.lineWidth = 1.2 * lw; vctx.setLineDash([3 * lw, 3 * lw]);
  vctx.beginPath(); vctx.moveTo(x0, cy); vctx.lineTo(x1, cy); vctx.moveTo(cx, y0); vctx.lineTo(cx, y1); vctx.stroke();
  vctx.setLineDash([]);
  const tickL = 5 * lw;
  vctx.beginPath();
  for (const [x, y, hor] of [[x0, cy, false], [x1, cy, false], [cx, y0, true], [cx, y1, true]]) {
    if (hor) { vctx.moveTo(x - tickL, y); vctx.lineTo(x + tickL, y); } else { vctx.moveTo(x, y - tickL); vctx.lineTo(x, y + tickL); }
  }
  vctx.stroke();
  vctx.font = `${Math.round(12 * lw)}px "IBM Plex Mono", monospace`;
  const tag = (t, x, y) => {
    const m = vctx.measureText(t);
    vctx.fillStyle = 'rgba(3,5,7,0.75)'; vctx.fillRect(x - 3 * lw, y - 12 * lw, m.width + 6 * lw, 16 * lw);
    vctx.fillStyle = '#f0ae2e'; vctx.fillText(t, x, y);
  };
  tag(`T ${fmtNum(f.widthMm)} mm`, x1 + 6 * lw, cy + 4 * lw);
  tag(`AP ${fmtNum(f.heightMm)} mm`, cx + 6 * lw, y1 + 16 * lw);
  vctx.restore();
}
function drawSaliency(box, sal) {
  const c = el('canvas', { width: PATCH, height: PATCH });
  const cx = c.getContext('2d');
  const img = cx.createImageData(PATCH, PATCH);
  for (let i = 0; i < PATCH * PATCH; i++) {
    const v = Math.min(1, sal[i] * 1.6);
    img.data[i * 4] = 255; img.data[i * 4 + 1] = Math.round(200 * (1 - v) + 40 * v); img.data[i * 4 + 2] = 40;
    img.data[i * 4 + 3] = Math.round(200 * v);
  }
  cx.putImageData(img, 0, 0);
  vctx.save();
  vctx.imageSmoothingEnabled = true;
  vctx.filter = 'blur(2px)';
  vctx.drawImage(c, box.x, box.y, box.side, box.side);
  vctx.restore();
}

// Tuval "object-fit: contain" ile yerleştiği için görüntü kenarlarında boşluk olabilir.
function toImg(e) {
  const r = view.getBoundingClientRect();
  const s = Math.min(r.width / view.width, r.height / view.height);
  const ox = (r.width - view.width * s) / 2, oy = (r.height - view.height * s) / 2;
  return { x: (e.clientX - r.left - ox) / s, y: (e.clientY - r.top - oy) / s };
}
view.addEventListener('pointerdown', (e) => {
  if (!state.prep) return;
  const p = toImg(e);
  state.drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, moved: false };
  view.setPointerCapture(e.pointerId);
  stopPlay();
});
view.addEventListener('pointermove', (e) => {
  if (!state.drag) return;
  const p = toImg(e);
  state.drag.x1 = p.x; state.drag.y1 = p.y;
  if (Math.hypot(p.x - state.drag.x0, p.y - state.drag.y0) > 6) state.drag.moved = true;
  if (state.drag.moved) render();
});
view.addEventListener('pointerup', () => {
  const d = state.drag;
  state.drag = null;
  if (!d || !state.prep) return;
  const { prep } = state;
  if (d.moved) {
    const m = ellipseMask(prep.w, prep.h, d.x0, d.y0, d.x1, d.y1);
    if (m) manualNodule(m); else render();
    return;
  }
  const pre = state.det?.preCache ? state.det.preCache(state.cur) : null;
  const r = segmentFromSeed(prep.grays[state.cur], prep.w, prep.h, d.x0, d.y0, 0.6, pre);
  if (!r.ok) { hint(r.reason, true); render(); return; }
  manualNodule(r.mask);
});

function setFrame(f) { if (!state.prep) return; state.cur = Math.max(0, Math.min(state.prep.cleaned.length - 1, f)); render(); }
function stopPlay() { state.playing = false; clearInterval(state.timer); $('playIcon').setAttribute('d', 'M6 4 L16 10 L6 16 Z'); $('playBtn').setAttribute('aria-label', 'Oynat'); }
function togglePlay() {
  if (!state.prep || state.prep.cleaned.length < 2) return;
  if (state.playing) return stopPlay();
  state.playing = true;
  $('playIcon').setAttribute('d', 'M5 4 H8.5 V16 H5 Z M11.5 4 H15 V16 H11.5 Z');
  $('playBtn').setAttribute('aria-label', 'Duraklat');
  state.timer = setInterval(() => setFrame((state.cur + 1) % state.prep.cleaned.length), 90);
}

// ---------- adaylar ----------
function thumb(frameIdx, cm, color) {
  const { prep } = state;
  const c = el('canvas', { width: 64, height: 64 });
  const ctx = c.getContext('2d');
  const side = Math.max(cm.bw, cm.bh) * 1.6;
  const sx = cm.x0 + cm.bw / 2 - side / 2, sy = cm.y0 + cm.bh / 2 - side / 2;
  const src = el('canvas', { width: prep.w, height: prep.h });
  src.getContext('2d').putImageData(new ImageData(prep.cleaned[frameIdx].data, prep.w, prep.h), 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 64, 64);
  ctx.drawImage(src, sx, sy, side, side, 0, 0, 64, 64);
  cm._poly ||= contourOf(cm, 72);
  ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  ctx.beginPath();
  cm._poly.forEach((p, i) => { const x = ((p.x - sx) / side) * 64, y = ((p.y - sy) / side) * 64; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.stroke();
  return c;
}

function renderCandidates() {
  const list = $('candList');
  list.innerHTML = '';
  if (!state.prep) { list.append(el('p', { class: 'empty' }, 'Kayıt yüklendiğinde nodül adayları burada listelenir.')); return; }
  const F = state.prep.cleaned.length;
  state.tracks.forEach((t, k) => {
    const color = TRACK_COLORS[k % TRACK_COLORS.length];
    const diam = Math.max(t.best.bw, t.best.bh) * state.mmPerPx;
    const on = state.nodule && state.selTrack === t;
    list.append(el('button', { type: 'button', class: `cand${on ? ' is-on' : ''}`, onclick: () => { stopPlay(); selectTrack(t); } },
      thumb(t.bestFrame, t.best, color),
      el('div', {},
        el('div', { class: 'cand-title' }, el('span', { class: 'swatch', style: `background:${color}` }), `Aday A${t.id}`,
          t.vesselLike ? el('span', { class: 'chip warn' }, 'sabit yapı: damar / gölge') : null,
          t.best.polarity === 'hyper' ? el('span', { class: 'chip' }, 'hiperekoik') : null),
        el('div', { class: 'cand-meta' }, `≈${fmtNum(diam)} mm · ${F > 1 ? `${t.n}/${F} karede görüldü` : 'tek görüntü'} · en büyük kesit: kare ${t.bestFrame + 1}`)),
      el('div', { class: 'cand-conf' }, el('b', {}, fmtPct(t.confidence, 0)), 'güven')));
  });
  if (state.nodule && !state.selTrack) {
    list.append(el('div', { class: 'cand is-on' },
      thumb(state.nodule.bestFrame, state.nodule.frameMasks.get(state.nodule.bestFrame), '#f0ae2e'),
      el('div', {}, el('div', { class: 'cand-title' }, 'Elle işaretlenen nodül'),
        el('div', { class: 'cand-meta' }, `${state.nodule.frameMasks.size} karede izlendi · en büyük kesit: kare ${state.nodule.bestFrame + 1}`)),
      el('div', { class: 'cand-conf' }, el('b', {}, 'hekim'), 'seçimi')));
  }
  if (!list.children.length) list.append(el('p', { class: 'empty' }, 'Aday bulunamadı. Görüntüde nodüle tıklayın veya elips çizin.'));
}

function renderFeatures() {
  const box = $('featureTables'), sum = $('noduleSummary');
  box.innerHTML = '';
  if (!state.nodule) { sum.hidden = true; return; }
  const f = state.nodule.features;
  sum.hidden = false;
  sum.innerHTML = '';
  const cell = (k, v) => el('div', {}, el('span', {}, k), el('b', {}, v));
  sum.append(
    cell('En büyük çap', `${fmtNum(f.maxDiamMm)} mm`),
    cell('T × AP', `${fmtNum(f.widthMm)} × ${fmtNum(f.heightMm)}`),
    cell('AP / T', fmtNum(f.apRatio, 2)),
    cell('Görüldüğü kare', String(f.framesSeen ?? 1)),
  );
  featureRows(f).forEach(([group, rows], i) => {
    box.append(el('details', { open: i < 2 },
      el('summary', {}, group),
      el('table', { class: 'ftable' }, el('tbody', {}, rows.map(([k, v]) => el('tr', {}, el('td', {}, k), el('td', {}, v)))))));
  });
}

// ---------- TI-RADS ----------
function renderTiradsForm() {
  const form = $('tiradsForm');
  form.innerHTML = '';
  if (!state.nodule) { form.append(el('p', { class: 'empty' }, 'Önce bir nodül seçin.')); return; }
  const { sel, suggestion } = state;
  for (const [field, def] of Object.entries(TIRADS)) {
    const sugg = suggestion.sel[field];
    const isSugg = (id) => (Array.isArray(sugg) ? sugg.includes(id) : sugg === id);
    const fs = el('fieldset', {},
      el('legend', {}, def.label,
        el('small', {}, `AI: ${suggestion.why[field]} · güven ${fmtPct(suggestion.conf[field], 0)}`),
        state.edited.has(field) ? el('span', { class: 'edited-tag' }, 'hekim düzeltti') : null));
    const opts = el('div', { class: 'opts' });
    for (const o of def.options) {
      const id = `tr-${field}-${o.id}`;
      const checked = def.multi ? sel[field].includes(o.id) : sel[field] === o.id;
      const input = el('input', { type: def.multi ? 'checkbox' : 'radio', name: field, id, value: o.id, checked });
      input.addEventListener('change', () => {
        if (def.multi) {
          let cur = [...form.querySelectorAll(`input[name="${field}"]:checked`)].map((x) => x.value);
          if (o.id === 'none' && input.checked) cur = ['none'];
          else if (o.id !== 'none') cur = cur.filter((x) => x !== 'none');
          if (!cur.length) cur = ['none'];
          sel[field] = cur;
        } else sel[field] = o.id;
        const same = JSON.stringify([].concat(sel[field]).sort()) === JSON.stringify([].concat(sugg).sort());
        if (same) state.edited.delete(field); else state.edited.add(field);
        renderTiradsForm();
        updateScore();
      });
      opts.append(el('span', { class: 'opt' }, input,
        el('label', { for: id }, o.label, el('span', { class: 'pts' }, `${o.pts} p`), isSugg(o.id) ? el('span', { class: 'ai-tag' }, 'AI') : null)));
    }
    fs.append(opts);
    form.append(fs);
  }
}

function renderTiradsResult() {
  const box = $('tiradsResult');
  if (!state.tr) { box.hidden = true; return; }
  const { tr, rec } = state;
  const n = tr.category.slice(2);
  box.hidden = false;
  box.innerHTML = '';
  box.append(...[
    el('div', { class: 'tr-line' },
      el('span', { class: 'tr-badge', style: `--trc: var(--tr${n})` }, tr.category),
      el('span', {}, el('b', {}, tr.info.name)),
      el('span', { class: 'tr-points' }, `${tr.points} puan`),
      el('span', { class: 'fine' }, `Kategori malignite oranı ${fmtPct(tr.info.risk)}`)),
    el('p', { class: `rec${rec.action === 'fna' ? ' fna' : ''}` }, rec.text),
    tr.note ? el('p', { class: 'fine' }, tr.note) : null,
    el('p', { class: 'fine' }, state.edited.size ? `Hekim ${state.edited.size} tanımlayıcıyı düzeltti; bu düzeltmeler vaka veri setine eklendiğinde saklanır.` : 'Tüm tanımlayıcılar AI önerisiyle aynı. Ekstratiroidal uzanım ve kuyruklu yıldız artefaktı otomatik değerlendirilmez.'),
  ].filter(Boolean));
}

// ---------- skor paneli ----------
function renderScore() {
  const body = $('scoreBody');
  body.innerHTML = '';
  $('exportRow').hidden = !(state.fused && CAN_DOWNLOAD);
  if (!state.fused) { body.append(el('p', { class: 'empty' }, 'TI-RADS değerlendirmesi tamamlandığında skor burada görünür.')); return; }
  const { fused, tr, prediction, meta, rec } = state;
  const scaleMax = 0.6;
  const pos = (p) => `${Math.min(100, (Math.min(p, scaleMax) / scaleMax) * 100)}%`;
  const scale = el('div', { class: 'risk-scale', 'aria-hidden': 'true' }, el('div', { class: 'track' }), el('div', { class: 'mark', style: `left:${pos(fused.final)}` }));
  for (const [k, v] of Object.entries(TR_INFO)) if (k !== 'TR1') scale.append(el('span', { class: 'tick', style: `left:${pos(v.risk)}` }, k));
  scale.append(el('span', { class: 'tick', style: 'left:97%' }, '≥%60'));

  body.append(
    el('div', { class: 'score-hero' },
      el('div', { class: 'score-num' }, el('small', {}, '%'), `${fmtNum(fused.final * 100)}`),
      el('div', {}, el('span', { class: `band ${fused.band.id}` }, fused.band.label),
        el('p', { class: 'fine' }, 'Tahmini malignite olasılığı. TI-RADS riski, öğrenen modelin doğrulanmış güvenilirliği oranında güncellenir.'))),
    scale,
  );

  const rows = [
    [`TI-RADS referans riski (${tr.category}, ${tr.points} puan)`, 'ACR TI-RADS çok merkezli doğrulama verisi', fmtPct(fused.prior)],
  ];
  if (prediction) {
    rows.push(['Görüntü modeli olasılığı', `${modelName(meta)} · dengeli eğitim seti (%50 prevalans)`, fmtPct(prediction.p)]);
    rows.push(['Model güvenilirlik ağırlığı (w)', `Doğrulama AUC ${fmtNum(meta.metrics.auc ?? 0, 2)} · ${meta.nTrain + meta.nVal} vaka (${meta.nClinical ?? 0} klinik)${(meta.nClinical ?? 0) < 30 ? ' · 30 klinik vakaya kadar yarım ağırlık' : ''}`, fmtNum(fused.w, 2)]);
  } else {
    rows.push(['Görüntü modeli', state.model ? 'Tahmin üretilemedi' : 'Henüz eğitilmedi; skor yalnızca TI-RADS’a dayanıyor', '–']);
  }
  rows.push(['Nihai skor', fused.w > 0 ? 'logit(TI-RADS) + w × model kanıtı' : 'TI-RADS ile aynı', fmtPct(fused.final)]);
  body.append(el('table', { class: 'breakdown' }, el('tbody', {}, rows.map((r, i) => el('tr', { class: i === rows.length - 1 ? 'total' : null },
    el('td', {}, r[0], el('span', { class: 'sub' }, r[1])), el('td', {}, r[2]))))));

  body.append(el('div', { class: 'note' }, el('b', {}, 'Öneri: '), rec.text));

  if (prediction) {
    const top = prediction.contributions.slice(0, 6);
    const mx = Math.max(0.05, ...top.map((c) => Math.abs(c.delta)));
    body.append(el('h3', {}, 'Modelin kararını etkileyen özellikler'),
      el('div', { class: 'contrib' }, top.map((c) => el('div', { class: 'contrib-row' },
        el('span', {}, c.label),
        el('span', { class: 'bar' }, el('i', { class: c.delta >= 0 ? 'up' : 'down', style: `width:${(Math.abs(c.delta) / mx) * 50}%` })),
        el('span', { class: 'v' }, `${c.delta >= 0 ? '+' : '−'}${fmtNum(Math.abs(c.delta), 2)}`)))),
      el('p', { class: 'fine' }, 'Değer: özellik eğitim ortalamasına çekildiğinde log-olasılıktaki değişim. Kırmızı riski artırır, yeşil azaltır. Görüntüde “AI dikkat haritası” modelin yama üzerindeki duyarlılığını gösterir.'));
  } else {
    body.append(el('button', { type: 'button', class: 'btn', onclick: () => goStep('learn') }, 'Öğrenme sekmesinde modeli eğit'));
  }
}

// ---------- rapor ----------
function snapshot() {
  const keep = state.cur;
  state.cur = state.nodule.bestFrame;
  render();
  const url = view.toDataURL('image/png');
  state.cur = keep; render();
  return url;
}
function resultJson() {
  const f = { ...state.nodule.features };
  delete f.contour;
  return {
    generatedAt: new Date().toISOString(), app: 'ThirdEye 0.1 (araştırma prototipi)',
    source: state.src, frames: state.prep.cleaned.length, mmPerPx: state.mmPerPx,
    nodule: { source: state.nodule.source, bestFrame: state.nodule.bestFrame + 1, framesSeen: state.nodule.frameMasks.size, features: f },
    tirads: { aiSuggestion: state.suggestion.sel, final: state.sel, editedFields: [...state.edited], ...state.tr, recommendation: state.rec },
    model: state.meta ? { version: state.meta.version, auc: state.meta.metrics.auc, pModel: state.prediction?.p ?? null, weight: state.fused.w } : null,
    risk: { prior: state.fused.prior, final: state.fused.final, band: state.fused.band.label },
  };
}
function exportReport() {
  const j = resultJson();
  const f = state.nodule.features;
  const img = snapshot();
  const rows = featureRows(f).flatMap(([g, r]) => [`<tr><th colspan="2">${g}</th></tr>`, ...r.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)]).join('');
  const html = `<!doctype html><html lang="tr"><meta charset="utf-8"><title>ThirdEye raporu</title>
<style>body{font:14px/1.5 system-ui,sans-serif;max-width:820px;margin:32px auto;padding:0 16px;color:#16212b}h1{font-size:22px;margin:0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:4px 0;text-align:left}td:last-child{text-align:right;font-family:monospace}th{padding-top:12px}.big{font-size:40px;font-weight:600}.warn{font-size:12px;color:#666;border-top:1px solid #ddd;padding-top:8px;margin-top:24px}img{max-width:100%;border-radius:6px}</style>
<h1>ThirdEye · Tiroid nodülü analiz raporu</h1><p>${new Date().toLocaleString('tr-TR')} · Kaynak: ${state.src.name}</p>
<img src="${img}" alt="Ölçüm karesi">
<p class="big">${fmtPct(j.risk.final)} <small>${j.risk.band}</small></p>
<p><b>${j.tirads.category}</b> (${j.tirads.points} puan, ${j.tirads.info.name}) · ${state.rec.text}</p>
<p>Hekim düzeltmesi: ${j.tirads.editedFields.length ? j.tirads.editedFields.join(', ') : 'yok'} · Model: ${j.model ? `v${j.model.version}, p=${fmtPct(j.model.pModel ?? 0)}, w=${fmtNum(j.model.weight, 2)}` : 'eğitilmemiş'}</p>
<table>${rows}</table>
<p class="warn">Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz.</p></html>`;
  download(`thirdeye-rapor-${Date.now()}.html`, new Blob([html], { type: 'text/html' }));
}

// ---------- öğrenme ----------
function renderTruthHint() {
  const h = $('truthHint');
  if (state.truth && state.nodule) {
    h.hidden = false;
    h.textContent = `Bu sentetik taramanın üretim etiketi: ${state.truth.label ? 'kötü huylu' : 'iyi huylu'} görünüm. Gerçek bir vakada bu bilgi sitoloji veya patolojiden gelir.`;
  } else h.hidden = true;
}

async function refreshDataset() {
  const cases = await store.allCases();
  const n = cases.length;
  const clin = cases.filter((c) => c.source === 'clinical').length;
  const mal = cases.filter((c) => c.label === 1).length;
  $('dataPill').textContent = `${n} vaka${clin ? ` · ${clin} klinik` : ''}`;
  const box = $('datasetStats');
  box.innerHTML = '';
  const stat = (k, v) => el('div', { class: 'stat' }, el('span', {}, k), el('b', {}, String(v)));
  box.append(stat('Toplam', n), stat('İyi huylu', n - mal), stat('Kötü huylu', mal), stat('Klinik', clin), stat('Sentetik', n - clin));
  return cases;
}

async function addCurrentCase() {
  const v = $('labelSelect').value;
  if (!v || !state.nodule) return;
  const [lab, detail] = v.split('|');
  const f = { ...state.nodule.features };
  delete f.contour;
  await store.addCases([{
    createdAt: new Date().toISOString(),
    source: state.src.kind === 'synthetic' ? 'synthetic' : 'clinical',
    label: +lab, labelDetail: detail, features: f, patch: state.nodule.patch.data,
    tirads: { sel: state.sel, category: state.tr.category, points: state.tr.points },
    aiSuggestion: state.suggestion.sel, editedFields: [...state.edited],
    predictionAtEntry: state.prediction?.p ?? null, sourceName: state.src.name,
  }]);
  $('labelSelect').value = '';
  $('addCaseBtn').disabled = true;
  $('trainStatus').textContent = `Vaka eklendi (${detail}). Modeli yeniden eğiterek bu bilgiyi öğrenmesini sağlayın.`;
  await refreshDataset();
}

async function generateSynthetic() {
  const n = +$('synthCount').value;
  const btn = $('synthBtn');
  btn.disabled = true;
  const base = 10000 + Math.floor(Math.random() * 1e6);
  const out = [];
  for (let i = 0; i < n; i++) {
    const c = syntheticTrainingCase(base + i, i % 2);
    if (c) out.push({ createdAt: new Date().toISOString(), source: 'synthetic', label: c.label, labelDetail: 'Sentetik üretim etiketi', features: c.features, patch: c.patch, segSource: c.segSource });
    if (i % 4 === 3) { busy(`Sentetik eğitim vakası üretiliyor (${i + 1}/${n})`, (i + 1) / n); $('trainStatus').textContent = `Sentetik vaka üretiliyor: ${i + 1}/${n}`; await tick(); }
  }
  await store.addCases(out);
  busy(null);
  btn.disabled = false;
  $('trainStatus').textContent = `${out.length} sentetik vaka eklendi.`;
  await refreshDataset();
}

function drawLoss(hist) {
  const c = $('lossChart'), ctx = c.getContext('2d');
  const W = c.width, H = c.height, pad = { l: 34, r: 8, t: 10, b: 22 };
  ctx.clearRect(0, 0, W, H);
  const ink = cssVar('--muted'), line = cssVar('--line');
  ctx.font = '10px "IBM Plex Mono", monospace';
  if (!hist?.length) { ctx.fillStyle = ink; ctx.fillText('Eğitim başlatılmadı', pad.l, H / 2); return; }
  const n = Math.max(hist.length, 2);
  const vals = hist.flatMap((h) => [h.loss, h.valLoss]).filter(Number.isFinite);
  const ymax = Math.max(0.8, ...vals) * 1.05;
  const X = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - v / ymax) * (H - pad.t - pad.b);
  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = ink;
  for (const v of [0, ymax / 2, ymax]) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W - pad.r, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), 2, Y(v) + 3); }
  ctx.fillText('epok 1', pad.l, H - 6);
  ctx.textAlign = 'right'; ctx.fillText(`epok ${hist.length}`, W - pad.r, H - 6); ctx.textAlign = 'left';
  const series = [['loss', cssVar('--accent'), []], ['valLoss', cssVar('--focus'), [4, 3]]];
  for (const [k, col, dash] of series) {
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash);
    ctx.beginPath();
    hist.forEach((h, i) => (i ? ctx.lineTo(X(i), Y(h[k])) : ctx.moveTo(X(i), Y(h[k]))));
    ctx.stroke();
    const last = hist[hist.length - 1];
    ctx.setLineDash([]); ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(X(hist.length - 1), Y(last[k]), 3, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.setLineDash([]);
}

function drawRoc(metrics) {
  const c = $('rocChart'), ctx = c.getContext('2d');
  const W = c.width, H = c.height, p = 22;
  ctx.clearRect(0, 0, W, H);
  const ink = cssVar('--muted'), line = cssVar('--line');
  ctx.strokeStyle = line; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(p, H - p); ctx.lineTo(W - 6, 6); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeRect(p, 6, W - p - 6, H - p - 6);
  ctx.fillStyle = ink; ctx.font = '10px "IBM Plex Mono", monospace';
  ctx.fillText('1−özgüllük', W / 2 - 24, H - 6);
  ctx.save(); ctx.translate(10, H / 2 + 24); ctx.rotate(-Math.PI / 2); ctx.fillText('duyarlılık', 0, 0); ctx.restore();
  if (!metrics?.roc?.length) return;
  const X = (v) => p + v * (W - p - 6), Y = (v) => H - p - v * (H - p - 6);
  ctx.fillStyle = cssVar('--accent-soft');
  ctx.beginPath(); ctx.moveTo(X(0), Y(0));
  for (const pt of metrics.roc) ctx.lineTo(X(pt.fpr), Y(pt.tpr));
  ctx.lineTo(X(1), Y(0)); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 2;
  ctx.beginPath();
  metrics.roc.forEach((pt, i) => (i ? ctx.lineTo(X(pt.fpr), Y(pt.tpr)) : ctx.moveTo(X(pt.fpr), Y(pt.tpr))));
  ctx.stroke();
  const y = metrics.youden;
  if (y) { ctx.fillStyle = cssVar('--accent'); ctx.beginPath(); ctx.arc(X(1 - y.spec), Y(y.sens), 3.5, 0, 2 * Math.PI); ctx.fill(); }
}

function renderModelInfo() {
  const { meta } = state;
  const pill = $('modelPill');
  $('exportModelBtn').hidden = !(meta && CAN_DOWNLOAD);
  if (!meta) {
    pill.textContent = hasTF() ? 'Model eğitilmedi' : 'TensorFlow.js yüklenemedi';
    pill.classList.remove('is-ready');
    $('metricTiles').innerHTML = ''; $('historyTable').innerHTML = '';
    drawLoss(null); drawRoc(null); $('rocCaption').textContent = 'ROC eğrisi';
    return;
  }
  const m = meta.metrics;
  pill.textContent = `${modelName(meta)} · AUC ${fmtNum(m.auc ?? 0, 2)} · ${meta.nTrain + meta.nVal} vaka`;
  pill.classList.add('is-ready');
  drawLoss(meta.trainHistory); drawRoc(m);
  $('rocCaption').textContent = `ROC · doğrulama seti (${m.nVal} vaka)`;
  const tiles = $('metricTiles');
  tiles.innerHTML = '';
  const t = (k, v) => el('div', { class: 'stat' }, el('span', {}, k), el('b', {}, v));
  tiles.append(
    t('AUC', fmtNum(m.auc ?? 0, 3)),
    t('Duyarlılık', fmtPct(m.youden.sens, 0)),
    t('Özgüllük', fmtPct(m.youden.spec, 0)),
    t('Skordaki ağırlık', fmtNum(modelReliability(meta), 2)),
  );
  const hist = meta.history.slice().reverse();
  $('historyTable').innerHTML = '';
  $('historyTable').append(el('table', {},
    el('thead', {}, el('tr', {}, ['Sürüm', 'Tarih', 'Vaka', 'Klinik', 'AUC', 'Duyarlılık', 'Özgüllük'].map((h) => el('th', {}, h)))),
    el('tbody', {}, hist.map((r) => el('tr', { class: r.version === meta.version ? 'current' : null },
      el('td', {}, `v${r.version}`), el('td', {}, new Date(r.date).toLocaleString('tr-TR', { dateStyle: 'short', timeStyle: 'short' })),
      el('td', {}, r.n), el('td', {}, r.nClinical), el('td', {}, fmtNum(r.auc ?? 0, 3)), el('td', {}, fmtPct(r.sens, 0)), el('td', {}, fmtPct(r.spec, 0)))))));
  $('trainStatus').textContent = `Son eğitim: ${new Date(meta.date).toLocaleString('tr-TR')} · eşik (Youden) ${fmtNum(m.youden.t, 2)} · doğrulama vakaları eğitimde kullanılmadı.`;
}

async function train() {
  if (!hasTF()) { $('trainStatus').textContent = 'TensorFlow.js yüklenemedi; model eğitimi bu tarayıcıda kullanılamıyor.'; return; }
  const btn = $('trainBtn');
  btn.disabled = true;
  try {
    const cases = await store.allCases();
    $('trainStatus').textContent = `${cases.length} vaka ile eğitim başlıyor…`;
    const epochs = +$('epochSelect').value;
    const { model, meta } = await trainModel(cases, {
      epochs, prevMeta: state.meta,
      onEpoch: (h) => { drawLoss(h); const l = h[h.length - 1]; $('trainStatus').textContent = `Epok ${l.epoch}/${epochs} · kayıp ${l.loss.toFixed(3)} · doğrulama kaybı ${l.valLoss.toFixed(3)}`; },
    });
    if (state.model) state.model.dispose();
    state.model = model; state.meta = meta;
    await store.setKV('modelMeta', meta);
    renderModelInfo();
    updateScore();
  } catch (e) {
    $('trainStatus').textContent = e.message;
  } finally { btn.disabled = false; }
}

// ---------- örnek video indir ----------
async function downloadSampleVideo() {
  const c = generateCase({ seed: 7, preset: 'malignant', frames: 36 });
  const canvas = el('canvas', { width: c.w, height: c.h });
  const ctx = canvas.getContext('2d');
  const stream = canvas.captureStream(12);
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 4e6 });
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  const done = new Promise((r) => { rec.onstop = r; });
  rec.start();
  const seq = [...c.frames.keys(), ...[...c.frames.keys()].reverse()];
  for (let i = 0; i < seq.length; i++) {
    const fr = c.frames[seq[i]];
    ctx.putImageData(new ImageData(fr.data, c.w, c.h), 0, 0);
    busy(`Örnek video kaydediliyor (${i + 1}/${seq.length})`, (i + 1) / seq.length);
    await new Promise((r) => setTimeout(r, 1000 / 12));
  }
  rec.stop();
  await done;
  busy(null);
  download('thirdeye-ornek-tarama.webm', new Blob(chunks, { type: 'video/webm' }));
}

// ---------- olaylar ----------
function bind() {
  for (const b of document.querySelectorAll('.step')) b.addEventListener('click', () => goStep(b.dataset.step));
  $('fileInput').addEventListener('change', (e) => { loadFiles(e.target.files); e.target.value = ''; });
  const drop = $('drop');
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('is-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('is-over'); loadFiles(e.dataTransfer.files); });
  $('viewer').addEventListener('dragover', (e) => e.preventDefault());
  $('viewer').addEventListener('drop', (e) => { e.preventDefault(); loadFiles(e.dataTransfer.files); });
  for (const b of document.querySelectorAll('[data-sample]')) b.addEventListener('click', () => loadSample(b.dataset.sample));
  $('downloadSample').hidden = !(CAN_DOWNLOAD && typeof MediaRecorder !== 'undefined');
  $('downloadSample').addEventListener('click', downloadSampleVideo);
  $('depthInput').addEventListener('change', () => {
    if (!state.prep) return;
    state.mmPerPx = (+$('depthInput').value * 10) / state.prep.h;
    renderSourceFacts();
    if (state.selTrack) selectTrack(state.selTrack);
  });
  $('optClean').addEventListener('change', () => { if (state.prep) hint('Temizleme ayarı bir sonraki yüklemede uygulanır.'); });
  $('frameSlider').addEventListener('input', (e) => { stopPlay(); setFrame(+e.target.value); });
  $('playBtn').addEventListener('click', togglePlay);
  const setView = (v) => {
    state.view = v;
    $('viewClean').classList.toggle('is-on', v === 'clean'); $('viewClean').setAttribute('aria-pressed', v === 'clean');
    $('viewOrig').classList.toggle('is-on', v === 'orig'); $('viewOrig').setAttribute('aria-pressed', v === 'orig');
    render();
  };
  $('viewClean').addEventListener('click', () => setView('clean'));
  $('viewOrig').addEventListener('click', () => setView('orig'));
  for (const id of ['optContours', 'optMeasure']) $(id).addEventListener('change', render);
  $('optSaliency').addEventListener('change', () => {
    if ($('optSaliency').checked && !state.prediction) hint('Dikkat haritası için önce Öğrenme sekmesinde modeli eğitin.', true);
    if (state.nodule) setFrame(state.nodule.bestFrame);
  });
  $('sensSlider').addEventListener('input', () => { $('sensOut').textContent = SENS[+$('sensSlider').value].label; });
  $('sensSlider').addEventListener('change', () => runDetection());
  document.addEventListener('keydown', (e) => {
    if (e.target.closest('input, select, textarea') || !state.prep) return;
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    if (e.key === 'ArrowRight') { stopPlay(); setFrame(state.cur + 1); }
    if (e.key === 'ArrowLeft') { stopPlay(); setFrame(state.cur - 1); }
  });
  // Telefonda adres çubuğu açılıp kapandıkça çok sayıda resize olayı gelir: kare başına bir kez çiz.
  let resizeQueued = false;
  window.addEventListener('resize', () => { if (resizeQueued) return; resizeQueued = true; requestAnimationFrame(() => { resizeQueued = false; render(); }); });

  $('labelSelect').addEventListener('change', () => { $('addCaseBtn').disabled = !($('labelSelect').value && state.nodule); });
  $('addCaseBtn').addEventListener('click', addCurrentCase);
  $('synthBtn').addEventListener('click', generateSynthetic);
  $('trainBtn').addEventListener('click', train);
  $('resetModel').addEventListener('click', async () => {
    await deleteModel(); await store.setKV('modelMeta', null);
    if (state.model) state.model.dispose();
    state.model = null; state.meta = null;
    const pre = await loadPretrained();
    if (pre) { state.model = pre.model; state.meta = pre.meta; }
    $('trainStatus').textContent = pre ? 'Yerel model silindi; Stanford ön-eğitimli modele dönüldü.' : 'Model sıfırlandı. Skor yeniden yalnızca TI-RADS’a dayanıyor.';
    renderModelInfo(); updateScore();
  });
  $('exportModelBtn').addEventListener('click', () => state.model && exportModel(state.model));
  $('clearSynth').addEventListener('click', async () => {
    const n = await store.deleteWhere((c) => c.source === 'synthetic');
    $('trainStatus').textContent = `${n} sentetik vaka silindi.`;
    refreshDataset();
  });
  $('exportData').hidden = !CAN_DOWNLOAD;
  $('exportData').addEventListener('click', async () => {
    const j = await store.exportDataset();
    download(`thirdeye-veriseti-${j.cases.length}.json`, new Blob([JSON.stringify(j)], { type: 'application/json' }));
  });
  $('importInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    try {
      const n = await store.importDataset(JSON.parse(await file.text()));
      $('trainStatus').textContent = `${n} vaka içe aktarıldı.`;
      refreshDataset();
    } catch (err) { $('trainStatus').textContent = err.message; }
  });
  $('exportReport').addEventListener('click', exportReport);
  $('exportJson').addEventListener('click', () => download(`thirdeye-sonuc-${Date.now()}.json`, new Blob([JSON.stringify(resultJson(), null, 2)], { type: 'application/json' })));
}

function renderAll() { renderCandidates(); renderFeatures(); renderTiradsForm(); renderTiradsResult(); renderScore(); render(); markSteps(); }

async function init() {
  if (MOBILE) $('framesSelect').value = '24';
  bind();
  renderAll();
  drawLoss(null); drawRoc(null);
  try {
    await refreshDataset();
    const meta = await store.getKV('modelMeta');
    if (meta && hasTF()) {
      const model = await loadModel();
      if (model) { state.model = model; state.meta = meta; }
    }
    if (!state.model) {
      const pre = await loadPretrained();
      if (pre) { state.model = pre.model; state.meta = pre.meta; }
    }
  } catch (e) { console.warn(e); }
  renderModelInfo();
  $('storageNote').textContent = store.storageMode() === 'memory' ? 'Tarayıcı depolaması kapalı: veriler sayfa kapanınca silinir.' : 'Veriler yalnızca bu tarayıcıda saklanır.';
  // İlk açılışta çalışır durumda görünmesi için şüpheli örnek tarama yüklenir.
  const HASH = { kayit: 'source', nodul: 'nodule', tirads: 'tirads', skor: 'score', ogrenme: 'learn' };
  const want = HASH[location.hash.slice(1)];
  const sample = ['benign', 'random'].includes(new URLSearchParams(location.search).get('ornek')) ? new URLSearchParams(location.search).get('ornek') : 'malignant';
  if (!location.search.includes('nosample')) await loadSample(sample);
  if (want) goStep(want);
  window.addEventListener('hashchange', () => { const s2 = HASH[location.hash.slice(1)]; if (s2) goStep(s2); });
}

init();
