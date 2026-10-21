// Uçtan uca analiz akışı: ön işleme → tespit/takip → nodül → özellikler → model girdisi.
// Arayüz (app.js) ve sentetik eğitim seti üretimi aynı fonksiyonları kullanır.

import { rgbaToGray, samplePatch, clamp } from './imgproc.js';
import { detectSector, cropFrame, cleanOverlays } from './preprocess.js';
import {
  detectCandidates, trackCandidates, expandMask, refineMask, segmentFromSeed,
  smoothFor, rowBackground, ratioMap, maskToCropped, propagate, DEFAULT_DETECT, cleanCandidate,
} from './detect.js';
import { computeFeatures, echogenicFoci } from './features.js';
import { generateCase } from './synth.js';

export const PATCH = 64;
const tick = () => new Promise((r) => setTimeout(r, 0));

export function preprocess(frames, { clean = true } = {}) {
  const sector = detectSector(frames, frames[0].width, frames[0].height);
  const orig = frames.map((f) => cropFrame(f, sector));
  const w = sector.w, h = sector.h;
  let cleaned = orig, overlayMasks = null, overlayRatio = 0;
  if (clean) {
    const r = cleanOverlays(orig);
    cleaned = r.cleaned; overlayMasks = r.masks; overlayRatio = r.ratio;
  }
  const grays = cleaned.map((f) => rgbaToGray(f.data, w, h));
  return { sector, w, h, orig, cleaned, overlayMasks, overlayRatio, grays };
}

// Arayüzü kilitlememek için kareler arasında nefes alan sürüm (telefonlarda önemli).
export async function preprocessAsync(frames, { clean = true, onProgress = null } = {}) {
  const sector = detectSector(frames, frames[0].width, frames[0].height);
  const orig = frames.map((f) => cropFrame(f, sector));
  const w = sector.w, h = sector.h;
  let cleaned = orig, overlayMasks = null, overlayRatio = 0;
  if (clean) {
    const r = await cleanOverlays(orig, undefined, async (k, n) => { onProgress?.((k + 1) / n); if (k % 3 === 2) await tick(); });
    cleaned = r.cleaned; overlayMasks = r.masks; overlayRatio = r.ratio;
  }
  const grays = cleaned.map((f) => rgbaToGray(f.data, w, h));
  return { sector, w, h, orig, cleaned, overlayMasks, overlayRatio, grays };
}

// Kare başına ön hesaplar (yumuşatma, satır referansı, oran haritası) önbelleğe alınır.
export function makePreCache(grays, w, h) {
  const cache = new Map();
  return (f) => {
    if (!cache.has(f)) {
      const s = smoothFor(grays[f], w, h);
      const bg = rowBackground(s, w, h);
      // Yalnızca oran haritası ve satır referansı saklanır (bellek tasarrufu).
      cache.set(f, { bg, ratio: ratioMap(s, bg, w, h) });
    }
    return cache.get(f);
  };
}

export async function detectAll(grays, w, h, cfg = DEFAULT_DETECT, onProgress = null, preCache = null) {
  const pc = preCache ?? makePreCache(grays, w, h);
  const perFrame = [];
  for (let f = 0; f < grays.length; f++) {
    perFrame.push(detectCandidates(grays[f], w, h, cfg, pc(f)).candidates);
    onProgress?.((f + 1) / grays.length);
    await tick();
  }
  return { perFrame, tracks: trackCandidates(perFrame), preCache: pc };
}

// Bir izden (otomatik) nodül nesnesi: kare maskeleri + en büyük kesit + özellikler.
export async function noduleFromTrack(track, grays, w, h, mmPerPx, preCache = null, maxFrames = 5) {
  const frameMasks = new Map(track.items.map((it) => [it.frame, it.cand]));
  // Her izlenen karede aday, uyarlanabilir segmentasyonla dış sınıra kadar genişletilir;
  // ölçüm karesi, gerçek segmentasyon alanı en büyük olan karedir.
  let bestFrame = track.bestFrame, mask = refineMask(cleanCandidate(expandMask(track.best, w, h), w, h), w, h), bestArea = 0;
  // Aday alanı en büyük birkaç karede segmentasyon yapılır (tüm karelerde yapmak yavaş ve gereksiz).
  const chosen = [...track.items].sort((a, b) => b.cand.area - a.cand.area).slice(0, maxFrames);
  for (const it of chosen) {
    await tick();
    const b = it.cand;
    const pre = preCache ? preCache(it.frame) : null;
    const seg = segmentFromSeed(grays[it.frame], w, h, b.cx, b.cy, 0.6, pre, b);
    if (!seg.ok) continue;
    const cm = maskToCropped(seg.mask, w, h);
    if (!cm) continue;
    let inter = 0;
    for (let y = 0; y < b.bh; y++) for (let x = 0; x < b.bw; x++) if (b.m[y * b.bw + x] && seg.mask[(y + b.y0) * w + x + b.x0]) inter++;
    if (inter < 0.6 * b.area) continue;
    frameMasks.set(it.frame, cm);
    if (cm.area > bestArea) { bestArea = cm.area; bestFrame = it.frame; mask = seg.mask; }
  }
  return finishNodule(frameMasks, bestFrame, mask, grays, w, h, mmPerPx, 'auto');
}

// Elle işaretlenen nodül: seçilen karedeki maske komşu karelere yayılır.
export function noduleFromManual(frame, mask, grays, w, h, mmPerPx, preCache) {
  const frameMasks = propagate(grays, w, h, frame, mask, preCache);
  let best = frame, bestA = 0;
  for (const [f, cm] of frameMasks) if (cm.area > bestA) { bestA = cm.area; best = f; }
  const bestMask = best === frame ? mask : refineMask(expandMask(frameMasks.get(best), w, h), w, h);
  return finishNodule(frameMasks, best, bestMask, grays, w, h, mmPerPx, 'manual');
}

function finishNodule(frameMasks, bestFrame, mask, grays, w, h, mmPerPx, source) {
  const features = computeFeatures(grays[bestFrame], w, h, mask, mmPerPx);
  if (!features) return null;
  // Odak taraması nodül çevresindeki küçük pencerede yapılır (tüm görüntü yerine).
  const fociInWindow = (gray, cm) => {
    const pad = 12;
    const x0 = Math.max(0, cm.x0 - pad), y0 = Math.max(0, cm.y0 - pad);
    const x1 = Math.min(w - 1, cm.x0 + cm.bw - 1 + pad), y1 = Math.min(h - 1, cm.y0 + cm.bh - 1 + pad);
    const ww = x1 - x0 + 1, wh = y1 - y0 + 1;
    const g = new Float32Array(ww * wh), m = new Uint8Array(ww * wh);
    for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
      g[y * ww + x] = gray[(y + y0) * w + x + x0];
      const mx = x + x0 - cm.x0, my = y + y0 - cm.y0;
      if (mx >= 0 && my >= 0 && mx < cm.bw && my < cm.bh) m[y * ww + x] = cm.m[my * cm.bw + mx];
    }
    return echogenicFoci(g, ww, wh, m, mmPerPx, features.parenchyma, Math.sqrt(cm.area / Math.PI), cm.cx - x0, cm.cy - y0, features.solidMean);
  };
  // Ekojenik odaklar tek karede kaçabilir: nodülün görüldüğü tüm karelerde aranır.
  if (frameMasks.size > 1) {
    const eqR = Math.sqrt(features.areaPx / Math.PI);
    let pMax = features.punctateCount, mMax = features.macroCount, rMax = features.rimCoverage, framesWithPunctate = 0;
    for (const [f, cm] of frameMasks) {
      if (f === bestFrame) { if (features.punctateCount) framesWithPunctate++; continue; }
      const r = fociInWindow(grays[f], cm);
      pMax = Math.max(pMax, r.punctate); mMax = Math.max(mMax, r.macro); rMax = Math.max(rMax, r.rimCoverage);
      if (r.punctate) framesWithPunctate++;
    }
    // Tek karede tek parlak nokta gürültü olabilir: en az iki karede görülmesi istenir.
    features.punctateCount = framesWithPunctate >= 2 || pMax >= 2 ? pMax : 0;
    features.macroCount = mMax;
    features.rimCoverage = rMax;
    features.framesWithPunctate = framesWithPunctate;
  }
  features.framesSeen = frameMasks.size;
  return { frameMasks, bestFrame, mask, features, source, patch: makePatch(grays[bestFrame], w, h, mask, features) };
}

// Model girdisi: nodül merkezli kare yama, 2 kanal (gri ton, maske), 64×64, Uint8.
export function makePatch(gray, w, h, mask, f) {
  const { x0, y0, x1, y1 } = f.bbox;
  const side = Math.max(x1 - x0 + 1, y1 - y0 + 1) * 1.3;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const px = cx - side / 2, py = cy - side / 2;
  const g = samplePatch(gray, w, h, px, py, side, side, PATCH);
  const mf = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) mf[i] = mask[i] ? 255 : 0;
  const m = samplePatch(mf, w, h, px, py, side, side, PATCH);
  const out = new Uint8Array(PATCH * PATCH * 2);
  for (let i = 0; i < PATCH * PATCH; i++) { out[2 * i] = clamp(Math.round(g[i]), 0, 255); out[2 * i + 1] = m[i] > 127 ? 255 : 0; }
  return { data: out, box: { x: px, y: py, side } };
}

function iou(a, b) {
  let inter = 0, uni = 0;
  for (let i = 0; i < a.length; i++) { if (a[i] && b[i]) inter++; if (a[i] || b[i]) uni++; }
  return uni ? inter / uni : 0;
}

// Sentetik eğitim vakası: tek kare üretilir, akıştaki segmentasyon denenir;
// başarısızsa (ör. izoekoik nodül) üreticinin gerçek maskesi kullanılır.
export function syntheticTrainingCase(seed, label, size = { w: 320, h: 240 }) {
  const r = Math.sin(seed * 12.9898) * 43758.5453;
  const zJ = ((r - Math.floor(r)) - 0.5) * 0.4;
  const c = generateCase({ seed, label, w: size.w, h: size.h, frames: 1, calipers: false, text: false, zJitter: zJ });
  const s = c.sector;
  const crop = cropFrame(c.frames[0], s);
  const w = s.w, h = s.h;
  const gray = rgbaToGray(crop.data, w, h);
  const truth = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) truth[y * w + x] = c.truth.mask[(y + s.y) * c.w + x + s.x];
  const seg = segmentFromSeed(gray, w, h, c.truth.center.x - s.x, c.truth.center.y - s.y);
  let mask = truth, segSource = 'truth';
  if (seg.ok && iou(seg.mask, truth) > 0.5) { mask = seg.mask; segSource = 'auto'; }
  const features = computeFeatures(gray, w, h, mask, c.mmPerPx);
  if (!features) return null;
  return { features, patch: makePatch(gray, w, h, mask, features).data, label: c.truth.label, segSource, params: c.truth.params };
}

export function cmFromFull(mask, w, h) { return maskToCropped(mask, w, h); }
