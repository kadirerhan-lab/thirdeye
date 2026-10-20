// Küçük, bağımlılıksız görüntü işleme araç kutusu.
// Tüm fonksiyonlar düz diziler (Float32Array / Uint8Array) ve w, h ile çalışır,
// böylece hem tarayıcıda hem Node testlerinde kullanılabilir.

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export function rgbaToGray(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  }
  return g;
}

// Ayrılabilir kutu bulanıklaştırma (kenarlar sabitlenir). rx/ry yarıçap.
export function boxBlur(src, w, h, rx, ry = rx) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  if (rx > 0) {
    const n = 2 * rx + 1;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let acc = 0;
      for (let k = -rx; k <= rx; k++) acc += src[row + clamp(k, 0, w - 1)];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = acc / n;
        acc += src[row + Math.min(x + rx + 1, w - 1)] - src[row + Math.max(x - rx, 0)];
      }
    }
  } else tmp.set(src);
  if (ry > 0) {
    const n = 2 * ry + 1;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -ry; k <= ry; k++) acc += tmp[clamp(k, 0, h - 1) * w + x];
      for (let y = 0; y < h; y++) {
        out[y * w + x] = acc / n;
        acc += tmp[Math.min(y + ry + 1, h - 1) * w + x] - tmp[Math.max(y - ry, 0) * w + x];
      }
    }
  } else out.set(tmp);
  return out;
}

const boxRadius = (sigma) => Math.max(0, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2));

// Üç geçişli kutu bulanıklaştırma ile Gauss yaklaşımı.
export function gaussBlur(src, w, h, sigmaX, sigmaY = sigmaX) {
  const rx = boxRadius(sigmaX), ry = boxRadius(sigmaY);
  if (rx === 0 && ry === 0) return Float32Array.from(src);
  let a = boxBlur(src, w, h, rx, ry);
  a = boxBlur(a, w, h, rx, ry);
  return boxBlur(a, w, h, rx, ry);
}

// İkili morfoloji (0/1 Uint8Array), kare yapı elemanı, önek toplamlarıyla O(1).
function morph(bin, w, h, r, isDilate) {
  if (r <= 0) return Uint8Array.from(bin);
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  const pre = new Int32Array(Math.max(w, h) + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) pre[x + 1] = pre[x] + bin[row + x];
    for (let x = 0; x < w; x++) {
      const a = Math.max(0, x - r), b = Math.min(w - 1, x + r);
      const c = pre[b + 1] - pre[a];
      tmp[row + x] = isDilate ? (c > 0 ? 1 : 0) : (c === b - a + 1 ? 1 : 0);
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) pre[y + 1] = pre[y] + tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      const a = Math.max(0, y - r), b = Math.min(h - 1, y + r);
      const c = pre[b + 1] - pre[a];
      out[y * w + x] = isDilate ? (c > 0 ? 1 : 0) : (c === b - a + 1 ? 1 : 0);
    }
  }
  return out;
}
export const dilate = (bin, w, h, r) => morph(bin, w, h, r, true);
export const erode = (bin, w, h, r) => morph(bin, w, h, r, false);
export const open = (bin, w, h, r) => dilate(erode(bin, w, h, r), w, h, r);
export const close = (bin, w, h, r) => erode(dilate(bin, w, h, r), w, h, r);

// Bağlı bileşen etiketleme (8 komşuluk). Her bileşen için alan, sınır kutusu, ağırlık merkezi.
export function labelComponents(bin, w, h, values = null) {
  const labels = new Int32Array(w * h);
  const comps = [];
  const stack = new Int32Array(w * h);
  let next = 1;
  for (let i = 0; i < w * h; i++) {
    if (!bin[i] || labels[i]) continue;
    const id = next++;
    let sp = 0;
    stack[sp++] = i;
    labels[i] = id;
    let area = 0, sx = 0, sy = 0, sv = 0;
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    while (sp > 0) {
      const p = stack[--sp];
      const px = p % w, py = (p - px) / w;
      area++; sx += px; sy += py;
      if (values) sv += values[p];
      if (px < x0) x0 = px; if (px > x1) x1 = px;
      if (py < y0) y0 = py; if (py > y1) y1 = py;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = py + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx;
          if (nx < 0 || nx >= w) continue;
          const q = ny * w + nx;
          if (bin[q] && !labels[q]) { labels[q] = id; stack[sp++] = q; }
        }
      }
    }
    comps.push({ id, area, x0, y0, x1, y1, cx: sx / area, cy: sy / area, mean: values ? sv / area : 0 });
  }
  return { labels, comps };
}

export function componentMask(labels, id, w, h) {
  const m = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (labels[i] === id) m[i] = 1;
  return m;
}

export function largestComponent(bin, w, h) {
  const { labels, comps } = labelComponents(bin, w, h);
  if (!comps.length) return { mask: new Uint8Array(w * h), comp: null };
  const best = comps.reduce((a, b) => (b.area > a.area ? b : a));
  return { mask: componentMask(labels, best.id, w, h), comp: best };
}

// Kenardan ulaşılamayan arka plan piksellerini doldurur (kistik boşluklar, kalsifikasyonlar).
export function fillHoles(bin, w, h) {
  const reached = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const push = (i) => { if (!bin[i] && !reached[i]) { reached[i] = 1; stack[sp++] = i; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp > 0) {
    const p = stack[--sp];
    const px = p % w, py = (p - px) / w;
    if (px > 0) push(p - 1);
    if (px < w - 1) push(p + 1);
    if (py > 0) push(p - w);
    if (py < h - 1) push(p + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = bin[i] || !reached[i] ? 1 : 0;
  return out;
}

// 3-4 chamfer mesafe dönüşümü: maske içindeki her pikselin en yakın arka plana uzaklığı (piksel).
export function distanceInside(bin, w, h) {
  const INF = 1e9;
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = bin[i] ? INF : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x > 0) v = Math.min(v, d[i - 1] + 3); else v = Math.min(v, 3);
    if (y > 0) {
      v = Math.min(v, d[i - w] + 3);
      if (x > 0) v = Math.min(v, d[i - w - 1] + 4);
      if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4);
    } else v = Math.min(v, 3);
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x < w - 1) v = Math.min(v, d[i + 1] + 3); else v = Math.min(v, 3);
    if (y < h - 1) {
      v = Math.min(v, d[i + w] + 3);
      if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4);
      if (x > 0) v = Math.min(v, d[i + w - 1] + 4);
    } else v = Math.min(v, 3);
    d[i] = v;
  }
  for (let i = 0; i < w * h; i++) d[i] /= 3;
  return d;
}

// Ağırlık merkezinden ışın atarak radyal kontur (yıldız biçimli nesneler için).
export function radialContour(mask, w, h, cx, cy, n = 256) {
  const pts = [];
  const rmax = Math.hypot(w, h);
  for (let k = 0; k < n; k++) {
    const t = (2 * Math.PI * k) / n;
    const dx = Math.cos(t), dy = Math.sin(t);
    let r = 0;
    for (let s = 0.5; s < rmax; s += 0.5) {
      const x = Math.round(cx + dx * s), y = Math.round(cy + dy * s);
      if (x < 0 || y < 0 || x >= w || y >= h || !mask[y * w + x]) break;
      r = s;
    }
    pts.push({ x: cx + dx * r, y: cy + dy * r, r, t });
  }
  return pts;
}

export function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a) / 2;
}

export function polygonPerimeter(pts) {
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    s += Math.hypot(q.x - p.x, q.y - p.y);
  }
  return s;
}

// Monotone chain dışbükey zarf.
export function convexHull(points) {
  const p = points.map((q) => ({ x: q.x, y: q.y })).sort((a, b) => a.x - b.x || a.y - b.y);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

// Soğan kabuğu yöntemiyle basit inpainting: maskeli pikseller dıştan içe,
// bilinen komşularının ortalamasıyla doldurulur. rgba yerinde değiştirilir.
export function inpaintRGBA(rgba, mask, w, h) {
  const known = new Uint8Array(w * h);
  let todo = [];
  for (let i = 0; i < w * h; i++) { if (mask[i]) todo.push(i); else known[i] = 1; }
  let guard = 0;
  while (todo.length && guard++ < 400) {
    const filled = [];
    const rest = [];
    for (const p of todo) {
      const px = p % w, py = (p - px) / w;
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = px + dx, ny = py + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (!known[q]) continue;
        r += rgba[q * 4]; g += rgba[q * 4 + 1]; b += rgba[q * 4 + 2]; n++;
      }
      if (n) {
        rgba[p * 4] = r / n; rgba[p * 4 + 1] = g / n; rgba[p * 4 + 2] = b / n;
        filled.push(p);
      } else rest.push(p);
    }
    if (!filled.length) break;
    for (const p of filled) known[p] = 1;
    todo = rest;
  }
  return rgba;
}

// 0–255 değerler için histogram tabanlı yüzdelik.
export function percentileOf(values, p) {
  const hist = new Uint32Array(256);
  let n = 0;
  for (let i = 0; i < values.length; i++) { hist[clamp(values[i] | 0, 0, 255)]++; n++; }
  if (!n) return 0;
  const target = p * (n - 1);
  let acc = 0;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > target) return v; }
  return 255;
}

export function maskedStats(values, mask) {
  let n = 0, s = 0, s2 = 0;
  for (let i = 0; i < values.length; i++) if (mask[i]) { n++; s += values[i]; s2 += values[i] * values[i]; }
  const mean = n ? s / n : 0;
  return { n, mean, std: n ? Math.sqrt(Math.max(0, s2 / n - mean * mean)) : 0 };
}

export function maskedPercentile(values, mask, p) {
  const arr = [];
  for (let i = 0; i < values.length; i++) if (mask[i]) arr.push(values[i]);
  return percentileOf(arr, p);
}

// Maske içindeki gri düzey eş-oluşum matrisi (16 düzey, sağ ve alt komşu).
export function glcmFeatures(gray, mask, w, h, levels = 16) {
  const m = new Float64Array(levels * levels);
  let total = 0;
  const q = (v) => clamp(Math.floor((v / 256) * levels), 0, levels - 1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!mask[i]) continue;
    const a = q(gray[i]);
    if (x + 1 < w && mask[i + 1]) { const b = q(gray[i + 1]); m[a * levels + b]++; m[b * levels + a]++; total += 2; }
    if (y + 1 < h && mask[i + w]) { const b = q(gray[i + w]); m[a * levels + b]++; m[b * levels + a]++; total += 2; }
  }
  if (!total) return { contrast: 0, homogeneity: 1, energy: 1, entropy: 0, correlation: 0 };
  let contrast = 0, homogeneity = 0, energy = 0, entropy = 0, mu = 0;
  for (let i = 0; i < levels; i++) for (let j = 0; j < levels; j++) {
    const p = m[i * levels + j] / total;
    if (!p) continue;
    contrast += p * (i - j) * (i - j);
    homogeneity += p / (1 + Math.abs(i - j));
    energy += p * p;
    entropy -= p * Math.log2(p);
    mu += i * p;
  }
  let varI = 0, cov = 0;
  for (let i = 0; i < levels; i++) for (let j = 0; j < levels; j++) {
    const p = m[i * levels + j] / total;
    if (!p) continue;
    varI += p * (i - mu) * (i - mu);
    cov += p * (i - mu) * (j - mu);
  }
  return { contrast, homogeneity, energy, entropy, correlation: varI > 1e-9 ? cov / varI : 0 };
}

// Bölgeyi yeniden örnekleyerek sabit boyutlu yama çıkarır (bilineer).
export function samplePatch(gray, w, h, x0, y0, bw, bh, size) {
  const out = new Float32Array(size * size);
  for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) {
    const fx = clamp(x0 + ((i + 0.5) / size) * bw - 0.5, 0, w - 1);
    const fy = clamp(y0 + ((j + 0.5) / size) * bh - 0.5, 0, h - 1);
    const ix = Math.floor(fx), iy = Math.floor(fy);
    const ax = fx - ix, ay = fy - iy;
    const ix1 = Math.min(ix + 1, w - 1), iy1 = Math.min(iy + 1, h - 1);
    const v = gray[iy * w + ix] * (1 - ax) * (1 - ay) + gray[iy * w + ix1] * ax * (1 - ay)
      + gray[iy1 * w + ix] * (1 - ax) * ay + gray[iy1 * w + ix1] * ax * ay;
    out[j * size + i] = v;
  }
  return out;
}

// Seed'li, tekrarlanabilir rastgele sayı üreteci (mulberry32).
export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.normal = (mu = 0, sd = 1) => {
    const u = Math.max(1e-12, next()), v = next();
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  next.range = (a2, b2) => a2 + (b2 - a2) * next();
  next.int = (a2, b2) => Math.floor(a2 + (b2 - a2 + 1) * next());
  next.pick = (arr) => arr[Math.floor(next() * arr.length)];
  next.chance = (p) => next() < p;
  return next;
}
