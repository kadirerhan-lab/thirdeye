// Nodül tespiti, kareler arası takip ve tohum noktasından segmentasyon.
//
// Yaklaşım (klasik görüntü işleme, eğitim gerektirmez):
//  1. Benek gürültüsünü bastırmak için Gauss yumuşatma.
//  2. Derinliğe bağlı zayıflamayı telafi etmek için satır bazlı parankim referansı
//     (her satırın 60. yüzdeliği) ve ekojenite oranı haritası.
//  3. Referansa göre hipoekoik (veya belirgin hiperekoik) bölgeler + morfolojik temizlik.
//  4. Alan, kontrast, şekil ve konum ile aday puanlama.
//  5. Videoda adayların IoU ile kareden kareye bağlanması; kalıcı izler nodül adayıdır.
// Bu modül ileride ONNX ile çalışan bir U-Net / YOLO segmentasyon modeliyle
// değiştirilebilecek şekilde aynı çıktı formatını üretir.

import {
  gaussBlur, open, close, fillHoles, labelComponents, componentMask, dilate,
  percentileOf, radialContour, clamp, convexHull, polygonArea,
} from './imgproc.js';

export const DEFAULT_DETECT = { hypo: 0.78, hyper: 1.35, topSkip: 0.1, minArea: 0.0015, maxArea: 0.3 };

export function smoothFor(gray, w, h) {
  return gaussBlur(gray, w, h, Math.max(2, w / 160));
}

export function rowBackground(s, w, h) {
  const bg = new Float32Array(h);
  const row = new Float32Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = s[y * w + x];
    // Önce satırın parlak kısmını (80. yüzdelik) bul, sonra ona yakın piksellerin medyanını al.
    const p80 = percentileOf(row, 0.8);
    const tissue = [];
    for (let x = 0; x < w; x++) if (row[x] > 0.55 * p80) tissue.push(row[x]);
    bg[y] = tissue.length ? percentileOf(tissue, 0.5) : p80;
  }
  // satırlar arası yumuşatma
  const r = Math.max(2, Math.round(h / 24));
  const out = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let a = 0, n = 0;
    for (let k = -r; k <= r; k++) { const yy = clamp(y + k, 0, h - 1); a += bg[yy]; n++; }
    out[y] = Math.max(18, a / n);
  }
  return out;
}

export function ratioMap(s, bg, w, h) {
  const r = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) r[y * w + x] = s[y * w + x] / bg[y];
  return r;
}

function cropMask(full, w, c) {
  const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
  const m = new Uint8Array(bw * bh);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) m[y * bw + x] = full[(y + c.y0) * w + x + c.x0];
  return { x0: c.x0, y0: c.y0, bw, bh, m };
}

export function expandMask(cm, w, h) {
  const full = new Uint8Array(w * h);
  for (let y = 0; y < cm.bh; y++) for (let x = 0; x < cm.bw; x++) {
    if (cm.m[y * cm.bw + x]) full[(y + cm.y0) * w + x + cm.x0] = 1;
  }
  return full;
}

export function maskToCropped(full, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (full[y * w + x]) {
    area++; sx += x; sy += y;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (!area) return null;
  const cm = cropMask(full, w, { x0, y0, x1, y1 });
  return { ...cm, area, cx: sx / area, cy: sy / area };
}

// Bir karedeki nodül adayları.
export function detectCandidates(gray, w, h, cfg = DEFAULT_DETECT, pre = null) {
  const s = pre?.s ?? smoothFor(gray, w, h);
  const bg = pre?.bg ?? rowBackground(s, w, h);
  const ratio = pre?.ratio ?? ratioMap(s, bg, w, h);
  const top = Math.round(h * cfg.topSkip);
  const out = [];
  for (const polarity of ['hypo', 'hyper']) {
    let bin = new Uint8Array(w * h);
    for (let y = top; y < h; y++) for (let x = 0; x < w; x++) {
      const v = ratio[y * w + x];
      bin[y * w + x] = polarity === 'hypo' ? (v < cfg.hypo ? 1 : 0) : (v > cfg.hyper ? 1 : 0);
    }
    bin = fillHoles(close(open(bin, w, h, 2), w, h, 3), w, h);
    const { labels, comps } = labelComponents(bin, w, h, ratio);
    for (const c of comps) {
      const fa = c.area / (w * h);
      if (fa < cfg.minArea || fa > cfg.maxArea) continue;
      const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
      const fill = c.area / (bw * bh);
      const elong = Math.max(bw / bh, bh / bw);
      const contrast = Math.abs(1 - c.mean);
      const shapeF = clamp((fill - 0.35) / 0.4, 0.1, 1);
      const elongF = elong > 3.5 ? 0.3 : 1;
      const border = c.x0 <= 1 || c.x1 >= w - 2 || c.y1 >= h - 2;
      const borderF = border ? 0.35 : 1;
      const lateralF = c.cx < 0.1 * w || c.cx > 0.9 * w ? 0.7 : 1;
      const polF = polarity === 'hyper' ? 0.6 : 1;
      const score = contrast * Math.sqrt(fa) * 10 * shapeF * elongF * borderF * lateralF * polF;
      if (score < 0.05) continue;
      const full = componentMask(labels, c.id, w, h);
      out.push({ ...cropMask(full, w, c), area: c.area, cx: c.cx, cy: c.cy, meanRatio: c.mean, polarity, score });
    }
  }
  out.sort((a, b) => b.score - a.score);
  return { candidates: out.slice(0, 6), s, bg, ratio };
}

function bboxIoU(a, b) {
  const ix = Math.max(0, Math.min(a.x0 + a.bw, b.x0 + b.bw) - Math.max(a.x0, b.x0));
  const iy = Math.max(0, Math.min(a.y0 + a.bh, b.y0 + b.bh) - Math.max(a.y0, b.y0));
  const inter = ix * iy;
  return inter / (a.bw * a.bh + b.bw * b.bh - inter);
}

// Kareler arası takip: açgözlü IoU eşleştirme, en fazla 2 kare boşluk.
export function trackCandidates(perFrame) {
  const F = perFrame.length;
  const tracks = [];
  for (let f = 0; f < F; f++) {
    const used = new Set();
    for (const c of perFrame[f]) {
      let best = null, bestIoU = 0.25;
      for (const t of tracks) {
        if (used.has(t)) continue;
        const last = t.items[t.items.length - 1];
        if (f - last.frame > 2 || f === last.frame) continue;
        const iou = bboxIoU(last.cand, c);
        if (iou > bestIoU) { bestIoU = iou; best = t; }
      }
      if (best) { best.items.push({ frame: f, cand: c }); used.add(best); }
      else { const t = { items: [{ frame: f, cand: c }] }; tracks.push(t); used.add(t); }
    }
  }
  const minLen = F >= 6 ? Math.max(2, Math.round(0.08 * F)) : 1;
  const res = tracks.filter((t) => t.items.length >= minLen).map((t) => {
    const n = t.items.length;
    const mean = t.items.reduce((a, it) => a + it.cand.score, 0) / n;
    const best = t.items.reduce((a, it) => (it.cand.area > a.cand.area ? it : a));
    const persistence = F > 1 ? n / F : 1;
    const areas = t.items.map((it) => it.cand.area);
    const am = areas.reduce((a, b) => a + b, 0) / n;
    const cv = Math.sqrt(areas.reduce((a, b) => a + (b - am) ** 2, 0) / n) / am;
    const meanRatio = t.items.reduce((a, it) => a + it.cand.meanRatio, 0) / n;
    const vesselLike = F >= 12 && persistence >= 0.95 && cv < 0.15 && meanRatio < 0.35;
    let score = mean * (0.4 + 0.6 * Math.min(1, n / Math.max(1, 0.3 * F)));
    if (vesselLike) score *= 0.4;
    return { items: t.items, n, persistence, score, bestFrame: best.frame, best: best.cand, vesselLike };
  });
  res.sort((a, b) => b.score - a.score);
  res.forEach((t, i) => { t.id = i + 1; t.confidence = clamp(t.score / 1.4, 0.05, 0.99); });
  return res.slice(0, 5);
}

// Uyarlanabilir segmentasyon: tohum çevresindeki pencerede bir dizi eşik denenir;
// her eşikte tohumu içeren bölgenin sınırındaki ortalama kenar gücü ölçülür.
// Güçlü kenarlı (en iyi skorun %70'i üzeri) adaylar içinden en kapsayıcı olan seçilir:
// böylece kistik iç kısım yerine halo/dış sınır dahil tüm nodül alınır.
export function adaptiveSegment(ratio, w, h, sx, sy, polarity = 'hypo', init = null) {
  const half = init ? Math.max(init.bw, init.bh) * 1.3 + 20 : 90;
  const cx0 = init ? init.x0 + init.bw / 2 : sx, cy0 = init ? init.y0 + init.bh / 2 : sy;
  const X0 = Math.max(0, Math.floor(cx0 - half)), X1 = Math.min(w - 1, Math.ceil(cx0 + half));
  const Y0 = Math.max(0, Math.floor(cy0 - half)), Y1 = Math.min(h - 1, Math.ceil(cy0 + half));
  const ww = X1 - X0 + 1, wh = Y1 - Y0 + 1;
  const R = new Float32Array(ww * wh);
  for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) R[y * ww + x] = ratio[(y + Y0) * w + x + X0];
  const G = new Float32Array(ww * wh);
  for (let y = 1; y < wh - 1; y++) for (let x = 1; x < ww - 1; x++) {
    const i = y * ww + x;
    G[i] = Math.hypot(R[i + 1] - R[i - 1], R[i + ww] - R[i - ww]) / 2;
  }
  const lx = Math.round(sx - X0), ly = Math.round(sy - Y0);
  const thresholds = [];
  if (polarity === 'hypo') for (let t = 0.5; t <= 0.96; t += 0.02) thresholds.push(t);
  else for (let t = 1.06; t <= 1.8; t += 0.03) thresholds.push(t);
  const cands = [];
  for (const t of thresholds) {
    let bin = new Uint8Array(ww * wh);
    for (let i = 0; i < ww * wh; i++) bin[i] = polarity === 'hypo' ? (R[i] < t ? 1 : 0) : (R[i] > t ? 1 : 0);
    bin = fillHoles(close(open(bin, ww, wh, 1), ww, wh, 2), ww, wh);
    const { labels, comps } = labelComponents(bin, ww, wh);
    let id = lx >= 0 && ly >= 0 && lx < ww && ly < wh ? labels[ly * ww + lx] : 0;
    if (!id) {
      let bd = 1e9;
      for (const c of comps) { const d = Math.hypot(c.cx - lx, c.cy - ly); if (d < bd) { bd = d; id = c.id; } }
      if (bd > 8) id = 0;
    }
    if (!id) continue;
    const c = comps.find((q) => q.id === id);
    if (c.area < 40 || c.area > 0.6 * ww * wh) continue;
    const touches = (c.x0 === 0 && X0 > 0) || (c.y0 === 0 && Y0 > 0) || (c.x1 === ww - 1 && X1 < w - 1) || (c.y1 === wh - 1 && Y1 < h - 1);
    if (touches) continue;
    // Başlangıç adayından aşırı büyüyen veya komşu yapıya taşan (düşük doluluk) bölgeler elenir.
    if (init && c.area > 6 * (init.area ?? init.bw * init.bh * 0.7)) continue;
    if (c.area / ((c.x1 - c.x0 + 1) * (c.y1 - c.y0 + 1)) < 0.45) continue;
    let gs = 0, gn = 0;
    for (let y = c.y0; y <= c.y1; y++) for (let x = c.x0; x <= c.x1; x++) {
      const i = y * ww + x;
      if (labels[i] !== id) continue;
      if (x === 0 || y === 0 || x === ww - 1 || y === wh - 1 || labels[i - 1] !== id || labels[i + 1] !== id || labels[i - ww] !== id || labels[i + ww] !== id) { gs += G[i]; gn++; }
    }
    const bpts = [];
    for (let y = c.y0; y <= c.y1; y++) {
      let first = -1, last = -1;
      for (let x = c.x0; x <= c.x1; x++) if (labels[y * ww + x] === id) { if (first < 0) first = x; last = x; }
      if (first >= 0) { bpts.push({ x: first, y }); bpts.push({ x: last + 1, y }); bpts.push({ x: first, y: y + 1 }); bpts.push({ x: last + 1, y: y + 1 }); }
    }
    const hullA = polygonArea(convexHull(bpts));
    const solidity = hullA ? c.area / hullA : 1;
    if (solidity < 0.8) continue;
    cands.push({ t, area: c.area, edge: gn ? gs / gn : 0, labels, id, solidity });
  }
  if (!cands.length) return null;
  const maxE = Math.max(...cands.map((c) => c.edge));
  const strong = cands.filter((c) => c.edge >= 0.7 * maxE);
  const pick = strong.reduce((a, b) => (b.area > a.area ? b : a));
  let sub = new Uint8Array(ww * wh);
  for (let i = 0; i < ww * wh; i++) sub[i] = pick.labels[i] === pick.id ? 1 : 0;
  // İnce köprülerle bağlanmış komşu yapıları (damar, derin doku) ayırmak için açma.
  const rOpen = Math.max(2, Math.round(0.08 * Math.sqrt(pick.area)));
  const opened = open(sub, ww, wh, rOpen);
  const oc = labelComponents(opened, ww, wh);
  if (oc.comps.length) {
    let keep = lx >= 0 && ly >= 0 && lx < ww && ly < wh ? oc.labels[ly * ww + lx] : 0;
    if (!keep) keep = oc.comps.reduce((a, b) => (b.area > a.area ? b : a)).id;
    const kept = new Uint8Array(ww * wh);
    for (let i = 0; i < ww * wh; i++) if (oc.labels[i] === keep) kept[i] = 1;
    sub = dilate(kept, ww, wh, rOpen);
    for (let i = 0; i < ww * wh; i++) sub[i] = sub[i] && pick.labels[i] === pick.id ? 1 : 0;
  }
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) if (sub[y * ww + x]) mask[(y + Y0) * w + x + X0] = 1;
  return { mask: refineMask(mask, w, h), threshold: pick.t, edge: pick.edge };
}

// Tıklanan noktadan segmentasyon. Kontrast yoksa açıklayıcı bir neden döner.
export function segmentFromSeed(gray, w, h, sx, sy, sens = 0.6, pre = null, init = null) {
  const s = pre?.s ?? smoothFor(gray, w, h);
  const bg = pre?.bg ?? rowBackground(s, w, h);
  const ratio = pre?.ratio ?? ratioMap(s, bg, w, h);
  sx = Math.round(clamp(sx, 0, w - 1)); sy = Math.round(clamp(sy, 0, h - 1));
  let acc = 0, n = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    const x = sx + dx, y = sy + dy;
    if (x < 0 || y < 0 || x >= w || y >= h) continue;
    acc += ratio[y * w + x]; n++;
  }
  const rs = acc / n;
  let polarity;
  if (rs < 0.9) polarity = 'hypo';
  else if (rs > 1.12) polarity = 'hyper';
  else return { ok: false, reason: 'Bu noktada çevre dokuya göre yeterli kontrast yok. Nodülü sürükleyerek elips ile çizebilirsiniz.' };
  const r = adaptiveSegment(ratio, w, h, sx, sy, polarity, init);
  if (!r) return { ok: false, reason: 'Belirgin sınırlı bir bölge bulunamadı. Nodülün iç kısmına tıklayın veya elips çizin.' };
  return { ok: true, mask: r.mask, polarity };
}

export function ellipseMask(w, h, x0, y0, x1, y1) {
  const m = new Uint8Array(w * h);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, a = Math.abs(x1 - x0) / 2, b = Math.abs(y1 - y0) / 2;
  if (a < 3 || b < 3) return null;
  for (let y = Math.max(0, Math.floor(cy - b)); y <= Math.min(h - 1, cy + b); y++) {
    for (let x = Math.max(0, Math.floor(cx - a)); x <= Math.min(w - 1, cx + a); x++) {
      if (((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1) m[y * w + x] = 1;
    }
  }
  return m;
}

// Maske düzeltme: kapanış, delik doldurma, kontur yumuşatma.
export function refineMask(mask, w, h) {
  const m = fillHoles(close(mask, w, h, 2), w, h);
  const f = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) f[i] = m[i];
  const b = gaussBlur(f, w, h, 1.5);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = b[i] >= 0.5 ? 1 : 0;
  return out;
}

// Elle seçilen bir nodülü komşu karelere yayar (tohum = önceki karenin ağırlık merkezi).
export function propagate(grays, w, h, startFrame, startMask, preCache = null) {
  const res = new Map();
  const start = maskToCropped(startMask, w, h);
  if (!start) return res;
  res.set(startFrame, start);
  for (const dir of [-1, 1]) {
    let prev = start;
    for (let f = startFrame + dir; f >= 0 && f < grays.length; f += dir) {
      const pre = preCache ? preCache(f) : null;
      const r = segmentFromSeed(grays[f], w, h, prev.cx, prev.cy, 0.6, pre, prev);
      if (!r.ok) break;
      const cm = maskToCropped(r.mask, w, h);
      if (!cm || cm.area > prev.area * 2.2 || cm.area < prev.area * 0.3) break;
      if (Math.hypot(cm.cx - prev.cx, cm.cy - prev.cy) > Math.sqrt(prev.area) * 0.6) break;
      res.set(f, cm);
      prev = cm;
    }
  }
  return res;
}

export function contourOf(cm, n = 96) {
  // Kırpılmış maskeden çizim için radyal kontur (tam görüntü koordinatlarında).
  const pts = radialContour(cm.m, cm.bw, cm.bh, cm.cx - cm.x0, cm.cy - cm.y0, n);
  return pts.map((p) => ({ x: p.x + cm.x0, y: p.y + cm.y0 }));
}

export { dilate };
