// Ön işleme: görüntü sektörünü bulma, ekran işaretlerini (kalibre "+", "x",
// noktalı ölçüm çizgileri, renkli yazılar) tespit edip inpainting ile temizleme.
// caliper-removal-lab reposundaki OpenCV yaklaşımının tarayıcı uyarlaması;
// videoda ayrıca "her karede sabit kalan parlak işaret" ipucu kullanılır.

import { open, dilate, largestComponent, labelComponents, inpaintRGBA } from './imgproc.js';

export function detectSector(frames, w, h) {
  const n = Math.min(frames.length, 8);
  const step = Math.max(1, Math.floor(frames.length / n));
  const acc = new Float32Array(w * h);
  let used = 0;
  for (let f = 0; f < frames.length && used < n; f += step, used++) {
    const d = frames[f].data;
    for (let i = 0, j = 0; i < w * h; i++, j += 4) acc[i] += (d[j] + d[j + 1] + d[j + 2]) / 3;
  }
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) bin[i] = acc[i] / used > 12 ? 1 : 0;
  const { comp } = largestComponent(open(bin, w, h, 3), w, h);
  if (!comp || comp.area < 0.2 * w * h) return { x: 0, y: 0, w, h };
  return { x: comp.x0, y: comp.y0, w: comp.x1 - comp.x0 + 1, h: comp.y1 - comp.y0 + 1 };
}

export function cropFrame(frame, s) {
  const out = new Uint8ClampedArray(s.w * s.h * 4);
  const src = frame.data, W = frame.width;
  for (let y = 0; y < s.h; y++) {
    const so = ((y + s.y) * W + s.x) * 4;
    out.set(src.subarray(so, so + s.w * 4), y * s.w * 4);
  }
  return { data: out, width: s.w, height: s.h };
}

export const DEFAULT_OVERLAY = { brightThreshold: 225, maxNeutralSat: 60, minColorSat: 100, dilation: 1 };

// Tek karede işaret adayları: parlak-nötr veya renkli pikseller; sonra şekil filtresi
// (ince/çizgisel/düşük doluluk) ile kompakt parlak dokular (ör. mikrokalsifikasyon) korunur.
export function frameOverlayCandidates(frame, cfg = DEFAULT_OVERLAY) {
  const { data, width: w, height: h } = frame;
  const cand = new Uint8Array(w * h);
  const colored = new Float32Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    const r = data[j], g = data[j + 1], b = data[j + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx ? ((mx - mn) / mx) * 255 : 0;
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    const isColor = sat >= cfg.minColorSat && mx >= 120;
    if ((gray >= cfg.brightThreshold && sat <= cfg.maxNeutralSat) || isColor) cand[i] = 1;
    colored[i] = isColor ? 1 : 0;
  }
  const { labels, comps } = labelComponents(cand, w, h, colored);
  const keep = new Uint8Array(comps.length + 1);
  for (const c of comps) {
    const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
    const fill = c.area / (bw * bh);
    const elong = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
    if ((c.mean >= 0.3 && c.area >= 2) || (c.area >= 3 && elong >= 3) || (c.area >= 6 && c.area <= 4000 && fill < 0.42)) keep[c.id] = 1;
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (labels[i] && keep[labels[i]]) out[i] = 1;
  return { mask: out, raw: cand };
}

// Tüm karelerde işaret maskesi + temizlenmiş kareler.
export function cleanOverlays(frames, cfg = DEFAULT_OVERLAY, onFrame = null) {
  if (onFrame) return cleanOverlaysAsync(frames, cfg ?? DEFAULT_OVERLAY, onFrame);
  const w = frames[0].width, h = frames[0].height, n = frames.length;
  const per = frames.map((f) => frameOverlayCandidates(f, cfg));
  // Videoda: karelerin %90'ında aday olan pikseller sabit ekran işaretidir.
  let staticMask = null;
  if (n >= 6) {
    const cnt = new Uint16Array(w * h);
    for (const p of per) for (let i = 0; i < w * h; i++) cnt[i] += p.raw[i];
    staticMask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) staticMask[i] = cnt[i] >= 0.9 * n ? 1 : 0;
  }
  let totalPx = 0;
  const masks = [];
  const cleaned = frames.map((f, k) => {
    let m = per[k].mask;
    if (staticMask) { m = Uint8Array.from(m); for (let i = 0; i < w * h; i++) if (staticMask[i]) m[i] = 1; }
    m = dilate(m, w, h, cfg.dilation);
    let c = 0;
    for (let i = 0; i < w * h; i++) c += m[i];
    totalPx += c;
    masks.push(m);
    const data = Uint8ClampedArray.from(f.data);
    if (c) inpaintRGBA(data, m, w, h);
    return { data, width: w, height: h };
  });
  return { cleaned, masks, ratio: totalPx / (w * h * n) };
}

// Kare kare ilerleyen sürüm: her kareden sonra arayüze nefes aldırır.
async function cleanOverlaysAsync(frames, cfg, onFrame) {
  const w = frames[0].width, h = frames[0].height, n = frames.length;
  const per = frames.map((f) => frameOverlayCandidates(f, cfg));
  // Videoda: karelerin %90'ında aday olan pikseller sabit ekran işaretidir.
  let staticMask = null;
  if (n >= 6) {
    const cnt = new Uint16Array(w * h);
    for (const p of per) for (let i = 0; i < w * h; i++) cnt[i] += p.raw[i];
    staticMask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) staticMask[i] = cnt[i] >= 0.9 * n ? 1 : 0;
  }
  let totalPx = 0;
  const masks = [];
  const cleaned = [];
  for (let k = 0; k < frames.length; k++) {
    const f = frames[k];
    let m = per[k].mask;
    if (staticMask) { m = Uint8Array.from(m); for (let i = 0; i < w * h; i++) if (staticMask[i]) m[i] = 1; }
    m = dilate(m, w, h, cfg.dilation);
    let c = 0;
    for (let i = 0; i < w * h; i++) c += m[i];
    totalPx += c;
    masks.push(m);
    const data = Uint8ClampedArray.from(f.data);
    if (c) inpaintRGBA(data, m, w, h);
    cleaned.push({ data, width: w, height: h });
    await onFrame(k, frames.length);
  }
  return { cleaned, masks, ratio: totalPx / (w * h * n) };
}
