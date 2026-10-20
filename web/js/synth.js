// Sentetik tiroid ultrason üreticisi.
// Amaç: gerçek hasta verisi olmadan uçtan uca demo yapabilmek ve modeli ilk aşamada
// eğitebilmek. Üretilen görüntüler GERÇEK DEĞİLDİR; doku katmanları, benek (speckle)
// gürültüsü, nodül morfolojisi ve kalibre işaretleri basitleştirilmiş fiziksel
// sezgilerle çizilir. Etiketler (iyi/kötü huylu) parametre dağılımlarından gelir.

import { rng, clamp, gaussBlur } from './imgproc.js';

const smooth = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

// Etikete göre nodül parametreleri. Dağılımlar bilinçli olarak örtüşür,
// böylece model "kolay" bir problem görmez.
export function sampleParams(r, label) {
  const mal = label === 1;
  const p = { label };
  p.sizeMm = mal ? r.range(7, 28) : r.range(8, 36);
  p.apRatio = mal ? clamp(r.normal(0.95, 0.22), 0.55, 1.5) : clamp(r.normal(0.76, 0.17), 0.45, 1.25);
  if (mal) p.echo = r.chance(0.3) ? r.range(0.32, 0.48) : r.range(0.5, 0.84);
  else {
    const t = r();
    p.echo = t < 0.5 ? r.range(0.88, 1.12) : t < 0.7 ? r.range(1.12, 1.35) : r.range(0.62, 0.9);
  }
  p.spongiform = false;
  p.cystic = 0;
  if (mal) {
    if (r.chance(0.12)) p.cystic = r.range(0.05, 0.25);
  } else {
    const t = r();
    if (t < 0.14) { p.spongiform = true; p.cystic = r.range(0.5, 0.7); p.echo = r.range(0.9, 1.15); }
    else if (t < 0.24) p.cystic = 0.93;
    else if (t < 0.55) p.cystic = r.range(0.15, 0.6);
  }
  p.irregularity = mal ? r.range(0.02, 0.13) : (r.chance(0.2) ? r.range(0.03, 0.07) : r.range(0, 0.03));
  p.edgeSoft = mal ? (r.chance(0.4) ? r.range(5, 9) : r.range(1.5, 3)) : (r.chance(0.2) ? r.range(4, 7) : r.range(1.2, 2.5));
  p.halo = mal ? r.chance(0.08) : r.chance(0.45);
  p.micro = mal ? (r.chance(0.5) ? r.int(3, 14) : 0) : (r.chance(0.06) ? r.int(1, 3) : 0);
  p.macro = r.chance(0.15) ? r.int(1, 2) : 0;
  p.rim = mal ? r.chance(0.08) : r.chance(0.06);
  p.heterogeneity = mal ? r.range(0.2, 0.42) : r.range(0.05, 0.22);
  return p;
}

// Belirli bir görünüm istendiğinde (demo düğmeleri) parametreleri zorlar.
export function presetParams(r, kind) {
  if (kind === 'benign') {
    const p = sampleParams(r, 0);
    Object.assign(p, { sizeMm: 21, apRatio: 0.68, echo: 0.68, cystic: 0.3, spongiform: false, irregularity: 0.01, edgeSoft: 1.6, halo: true, micro: 0, macro: 0, rim: false, heterogeneity: 0.12 });
    return p;
  }
  if (kind === 'malignant') {
    const p = sampleParams(r, 1);
    Object.assign(p, { sizeMm: 14, apRatio: 1.22, echo: 0.42, cystic: 0, spongiform: false, irregularity: 0.11, edgeSoft: 2.2, halo: false, micro: 9, macro: 0, rim: false, heterogeneity: 0.3 });
    return p;
  }
  return sampleParams(r, r.chance(0.5) ? 1 : 0);
}

function speckleField(r, w, h) {
  const re = new Float32Array(w * h), im = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { re[i] = r.normal(); im[i] = r.normal(); }
  // Lateral çözünürlük eksenel çözünürlükten kötü: benek yatay olarak uzar.
  const a = gaussBlur(re, w, h, 1.4, 0.8), b = gaussBlur(im, w, h, 1.4, 0.8);
  const s = new Float32Array(w * h);
  let mean = 0;
  for (let i = 0; i < w * h; i++) { s[i] = Math.pow(Math.hypot(a[i], b[i]), 0.6); mean += s[i]; }
  mean /= w * h;
  for (let i = 0; i < w * h; i++) s[i] /= mean;
  return s;
}

function lowFreqField(r, w, h, sigma) {
  const n = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) n[i] = r.normal();
  const f = gaussBlur(n, w, h, sigma);
  let s2 = 0;
  for (let i = 0; i < w * h; i++) s2 += f[i] * f[i];
  const sd = Math.sqrt(s2 / (w * h)) || 1;
  for (let i = 0; i < w * h; i++) f[i] /= sd;
  return f;
}

// Statik anatomi: cilt, cilt altı yağ, strap kaslar, tiroid parankimi,
// karotis arter (anekoik), trakea (parlak yay + akustik gölge), derin kaslar.
function anatomyMap(r, sw, sh) {
  const T = new Float32Array(sw * sh);
  const stri = lowFreqField(r, sw, sh, 2);
  const carotid = { x: sw * r.range(0.9, 0.95), y: sh * r.range(0.5, 0.58), r: sh * r.range(0.11, 0.14) };
  const trachea = { x: sw * r.range(0.0, 0.04), y: sh * 0.22, r: sh * 0.2 };
  const P = 132;
  for (let v = 0; v < sh; v++) {
    const att = 1 - 0.32 * (v / sh);
    for (let u = 0; u < sw; u++) {
      const un = u / sw;
      let t;
      const thyBottom = sh * (0.84 - 0.55 * (un - 0.45) * (un - 0.45));
      if (v < 4) t = 175;
      else if (v < 14) t = 72;
      else if (v < 40) t = 58 * (1 + 0.22 * Math.sin(v * 1.1 + stri[v * sw + u] * 0.8));
      else if (v < 43) t = 150; // kapsül
      else if (v < thyBottom) t = P;
      else if (v < thyBottom + 3) t = 140;
      else t = 52;
      // karotis
      const dc = Math.hypot(u - carotid.x, v - carotid.y);
      if (dc < carotid.r) t = 6;
      else if (dc < carotid.r + 2.5) t = 165;
      else if (v > carotid.y && Math.abs(u - carotid.x) < carotid.r) t *= 1.25; // posterior güçlenme
      // trakea
      const dt = Math.hypot(u - trachea.x, v - trachea.y);
      if (Math.abs(dt - trachea.r) < 2.5 && v < trachea.y + trachea.r * 0.2) t = 210;
      else if (u < trachea.x + trachea.r * 0.9 && v > trachea.y - trachea.r * 0.4) t *= 0.18;
      T[v * sw + u] = t * att;
    }
  }
  return T;
}

// Normalize nodül uzayında (yarıçap 1) küresel kistik boşluklar.
function cystBlobs(r, p) {
  const blobs = [];
  if (p.cystic <= 0) return blobs;
  if (p.cystic >= 0.9) {
    blobs.push({ x: r.range(-0.03, 0.03), y: r.range(-0.03, 0.03), z: 0, r: 0.93 });
    return blobs;
  }
  if (p.spongiform) {
    const n = r.int(14, 24);
    const rr = Math.sqrt(p.cystic / n) * 1.15;
    for (let i = 0; i < n; i++) {
      const a = r.range(0, 2 * Math.PI), d = Math.sqrt(r()) * (0.85 - rr);
      blobs.push({ x: Math.cos(a) * d, y: Math.sin(a) * d, z: r.range(-0.5, 0.5), r: rr * r.range(0.75, 1.25) });
    }
    return blobs;
  }
  const n = r.int(1, 3);
  for (let i = 0; i < n; i++) {
    const rr = Math.sqrt(p.cystic / n) * r.range(0.9, 1.1);
    const a = r.range(0, 2 * Math.PI), d = r.range(0, Math.max(0, 0.85 - rr));
    blobs.push({ x: Math.cos(a) * d, y: Math.sin(a) * d, z: r.range(-0.25, 0.25), r: Math.min(0.8, rr) });
  }
  return blobs;
}

export function generateCase(opts = {}) {
  const {
    seed = 1, label = null, preset = null, w = 480, h = 360, frames = 36,
    calipers = true, text = true, zJitter = 0,
  } = opts;
  const r = rng(seed);
  const p = preset ? presetParams(r, preset) : sampleParams(r, label == null ? (r.chance(0.5) ? 1 : 0) : label);

  // Lineer prob görüntü alanı (sektör) ve çevresindeki siyah kenar.
  const sx = Math.round(w * 0.05), sy = Math.round(h * 0.08);
  const sw = w - 2 * sx, sh = h - sy - Math.round(h * 0.02);
  const mmPerPx = 40 / sw; // 4 cm genişlikte lineer prob

  const T0 = anatomyMap(r, sw, sh);
  const tex = lowFreqField(r, sw, sh, 3.5);
  const sp1 = speckleField(r, sw, sh);
  const speckles = frames === 1 ? [sp1, sp1, sp1] : [speckleField(r, sw, sh), sp1, speckleField(r, sw, sh)];

  // Nodül geometrisi
  let a = p.sizeMm / 2 / mmPerPx;
  let b = a * p.apRatio;
  const maxB = sh * 0.27, maxA = sw * 0.3;
  const k = Math.min(1, maxB / b, maxA / a);
  a *= k; b *= k;
  const cx = Math.min(sw * r.range(0.36, 0.6), sw * 0.86 - a * 1.15 - 14);
  const cy = clamp(sh * r.range(0.46, 0.56), 48 + b + 6, sh * 0.8 - b);
  const harmonics = [];
  for (let n = 2; n <= 14; n++) {
    harmonics.push({ n, amp: p.irregularity * r.range(0.3, 1) * (n >= 5 ? 1 : 0.6) / (1 + 0.12 * n), ph: r.range(0, 2 * Math.PI) });
  }
  const radiusMod = (t) => { let s = 1; for (const hm of harmonics) s += hm.amp * Math.sin(hm.n * t + hm.ph); return s; };
  const blobs = cystBlobs(r, p);
  const micro = [];
  for (let i = 0; i < p.micro; i++) {
    const ang = r.range(0, 2 * Math.PI), d = Math.sqrt(r()) * 0.75;
    micro.push({ x: Math.cos(ang) * d, y: Math.sin(ang) * d, z: r.range(-0.55, 0.55), s: r.range(1.0, 1.7) });
  }
  const macro = [];
  for (let i = 0; i < p.macro; i++) {
    const ang = r.range(0, 2 * Math.PI), d = r.range(0, 0.5);
    macro.push({ x: Math.cos(ang) * d, y: Math.sin(ang) * d, z: r.range(-0.3, 0.3), rpx: r.range(7, 11) });
  }
  const rim = p.rim ? { t0: r.range(0, 2 * Math.PI), span: r.range(0.6, 1.4) * Math.PI } : null;

  const P = 132;
  const out = [];
  let bestFrame = 0, bestAbs = 9;
  const truthMask = new Uint8Array(w * h);

  for (let f = 0; f < frames; f++) {
    const z = frames === 1 ? zJitter : -1.25 + (2.5 * f) / (frames - 1);
    if (Math.abs(z) < bestAbs) { bestAbs = Math.abs(z); bestFrame = f; }
    const T = Float32Array.from(T0);
    const post = new Float32Array(sw * sh).fill(1);
    const B = new Float32Array(sw * sh); // kalsifikasyon katmanı
    const s = Math.sqrt(Math.max(0, 1 - z * z));
    const ccx = cx + z * 6, ccy = cy;
    if (s > 0.05) {
      const as = a * s, bs = b * s;
      const soft = p.edgeSoft / as;
      const haloW = 3.2 / as;
      const pad = Math.ceil(Math.max(as, bs) * (1 + p.irregularity * 3) + 12);
      const u0 = Math.max(0, Math.floor(ccx - pad)), u1 = Math.min(sw - 1, Math.ceil(ccx + pad));
      const v0 = Math.max(0, Math.floor(ccy - pad)), v1 = Math.min(sh - 1, Math.ceil(ccy + pad));
      // Kistik boşlukların bu kesitteki yarıçapları
      const vis = blobs.map((bl) => ({ ...bl, re: Math.sqrt(Math.max(0, bl.r * bl.r - (z - bl.z) * (z - bl.z))) })).filter((bl) => bl.re > 0.02);
      let cysticVisible = 0;
      for (const bl of vis) cysticVisible += bl.re * bl.re;
      cysticVisible = Math.min(1, cysticVisible / (s * s + 1e-6));
      for (let v = v0; v <= v1; v++) for (let u = u0; u <= u1; u++) {
        const du = (u - ccx) / a, dv = (v - ccy) / b; // normalize (yarıçap 1, tam kesit)
        const rho = Math.hypot(du, dv);
        const th = Math.atan2(dv, du);
        const R = s * radiusMod(th);
        const d = rho / R;
        const i = v * sw + u;
        if (d < 1 + soft + haloW) {
          // iç ekojenite + doku heterojenliği
          let inside = P * p.echo * (1 - 0.32 * (v / sh)) * (1 + p.heterogeneity * tex[i]);
          for (const bl of vis) {
            const dd = Math.hypot(du - bl.x, dv - bl.y) / bl.re;
            if (dd < 1.15) inside = inside * smooth(0.85, 1.15, dd) + 6 * (1 - smooth(0.85, 1.15, dd));
          }
          const alpha = 1 - smooth(1 - soft, 1 + soft, d);
          let val = inside * alpha + T[i] * (1 - alpha);
          if (p.halo && d > 0.97 && d < 1 + haloW) val *= 0.38;
          T[i] = val;
        }
      }
      // posterior güçlenme (kistik içerik)
      if (cysticVisible > 0.05) {
        const bottom = ccy + b * s;
        for (let u = Math.max(0, Math.floor(ccx - a * s * 0.8)); u <= Math.min(sw - 1, ccx + a * s * 0.8); u++) {
          for (let v = Math.ceil(bottom); v < sh; v++) {
            const fade = Math.exp(-(v - bottom) / (sh * 0.25));
            post[v * sw + u] *= 1 + 0.45 * cysticVisible * fade;
          }
        }
      }
      // mikrokalsifikasyonlar (punktat ekojenik odaklar)
      for (const m of micro) {
        if (Math.abs(z - m.z) > 0.2) continue;
        const mx = ccx + m.x * a * s, my = ccy + m.y * b * s;
        const rr = m.s;
        for (let v = Math.floor(my - 2); v <= Math.ceil(my + 2); v++) for (let u = Math.floor(mx - 2); u <= Math.ceil(mx + 2); u++) {
          if (u < 0 || v < 0 || u >= sw || v >= sh) continue;
          const dd = Math.hypot(u - mx, v - my);
          if (dd <= rr) B[v * sw + u] = 245;
        }
      }
      // makrokalsifikasyon + akustik gölge
      for (const m of macro) {
        if (Math.abs(z - m.z) > 0.3) continue;
        const mx = ccx + m.x * a * s, my = ccy + m.y * b * s;
        const rr = m.rpx * Math.sqrt(1 - Math.min(0.9, ((z - m.z) / 0.3) ** 2));
        for (let v = Math.floor(my - rr); v <= Math.ceil(my + rr); v++) for (let u = Math.floor(mx - rr); u <= Math.ceil(mx + rr); u++) {
          if (u < 0 || v < 0 || u >= sw || v >= sh) continue;
          const dd = Math.hypot((u - mx), (v - my) * 1.6);
          if (dd <= rr) B[v * sw + u] = 235;
        }
        for (let u = Math.floor(mx - rr); u <= Math.ceil(mx + rr); u++) for (let v = Math.ceil(my + rr * 0.6); v < sh; v++) {
          if (u < 0 || u >= sw) continue;
          post[v * sw + u] *= 0.3 + 0.7 * smooth(0, 1, Math.abs(u - mx) / rr);
        }
      }
      // periferik (rim) kalsifikasyon
      if (rim) {
        for (let k2 = 0; k2 < 160; k2++) {
          const t = rim.t0 + (rim.span * k2) / 159;
          const R = s * radiusMod(t);
          const px = ccx + Math.cos(t) * R * a, py = ccy + Math.sin(t) * R * b;
          for (let v = Math.floor(py - 2); v <= Math.ceil(py + 2); v++) for (let u = Math.floor(px - 2); u <= Math.ceil(px + 2); u++) {
            if (u < 0 || v < 0 || u >= sw || v >= sh) continue;
            if (Math.hypot(u - px, v - py) <= 1.8) B[v * sw + u] = 228;
          }
        }
        const bottom = ccy + b * s;
        for (let u = Math.floor(ccx - a * s * 0.6); u <= ccx + a * s * 0.6; u++) for (let v = Math.ceil(bottom); v < sh; v++) {
          if (u >= 0 && u < sw) post[v * sw + u] *= 0.55;
        }
      }
    }

    // benek gürültüsü (kesit boyunca alanlar arasında karışım)
    const zz = clamp((z + 1.25) / 2.5, 0, 1) * 2;
    const k0 = Math.min(1, Math.floor(zz)), fr = zz - k0;
    const S0 = speckles[k0], S1 = speckles[k0 + 1];
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    for (let v = 0; v < sh; v++) for (let u = 0; u < sw; u++) {
      const i = v * sw + u;
      const S = S0[i] * (1 - fr) + S1[i] * fr;
      // log sıkıştırmaya benzer yumuşak doygunluk: doku parlaklığı 255'e nadiren ulaşır
      const val = clamp(B[i] > 0 ? B[i] * (0.9 + 0.1 * S) : 255 * (1 - Math.exp(-(T[i] * post[i] * S) / 180)), 0, 255);
      const o = ((v + sy) * w + (u + sx)) * 4;
      rgba[o] = val; rgba[o + 1] = val; rgba[o + 2] = val * 0.98;
    }

    // derinlik işaretleri (sektör dışında)
    for (let cm = 0; cm <= 4; cm++) {
      const yy = sy + Math.round((cm * 10) / mmPerPx);
      if (yy >= h - 1) continue;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const o = ((yy + dy) * w + (w - Math.round(sx / 2) + dx)) * 4;
        if (o >= 0 && o < rgba.length) { rgba[o] = 230; rgba[o + 1] = 230; rgba[o + 2] = 230; }
      }
    }

    // Kalibre işaretleri: ölçüm için dondurulmuş karelerde ekranda kalan "+" ve "x"
    if (calipers && s > 0.05 && Math.abs(z) < 0.35) {
      const L = { x: sx + ccx - a * s * radiusMod(Math.PI), y: sy + ccy };
      const Rr = { x: sx + ccx + a * s * radiusMod(0), y: sy + ccy };
      const Tp = { x: sx + ccx, y: sy + ccy - b * s * radiusMod(-Math.PI / 2) };
      const Bt = { x: sx + ccx, y: sy + ccy + b * s * radiusMod(Math.PI / 2) };
      const put = (x, y, c) => {
        x = Math.round(x); y = Math.round(y);
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        const o = (y * w + x) * 4; rgba[o] = c[0]; rgba[o + 1] = c[1]; rgba[o + 2] = c[2];
      };
      const plus = (pt, c) => { for (let d = -5; d <= 5; d++) { put(pt.x + d, pt.y, c); put(pt.x, pt.y + d, c); } };
      const cross = (pt, c) => { for (let d = -4; d <= 4; d++) { put(pt.x + d, pt.y + d, c); put(pt.x + d, pt.y - d, c); } };
      const dotted = (p1, p2, c) => {
        const n = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        for (let t = 8; t < n - 8; t += 7) for (let q = 0; q < 3; q++) { const x = p1.x + ((p2.x - p1.x) * (t + q)) / n, y = p1.y + ((p2.y - p1.y) * (t + q)) / n; put(x, y, c); }
      };
      const Y = [255, 214, 10], W = [240, 240, 240];
      plus(L, Y); plus(Rr, Y); dotted(L, Rr, Y);
      cross(Tp, W); cross(Bt, W); dotted(Tp, Bt, W);
    }

    let frame = { data: rgba, width: w, height: h };
    if (text && typeof OffscreenCanvas !== 'undefined' && typeof ImageData !== 'undefined') {
      const oc = new OffscreenCanvas(w, h);
      const ctx = oc.getContext('2d');
      ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
      ctx.fillStyle = '#e8e8e8';
      ctx.font = '11px monospace';
      ctx.fillText('SIM L12-5  TIROID  D4.0cm  G54  FR18', sx, sy - 10);
      ctx.fillText(`${String(f + 1).padStart(2, '0')}/${frames}`, w - sx - 40, sy - 10);
      frame = ctx.getImageData(0, 0, w, h);
    }
    out.push(frame);
  }

  // En büyük kesitte gerçek nodül maskesi (eğitim verisi için).
  {
    const z = frames === 1 ? zJitter : -1.25 + (2.5 * bestFrame) / (frames - 1);
    const s = Math.sqrt(Math.max(0, 1 - z * z));
    const ccx = cx + z * 6, ccy = cy;
    for (let v = 0; v < sh; v++) for (let u = 0; u < sw; u++) {
      const du = (u - ccx) / a, dv = (v - ccy) / b;
      const rho = Math.hypot(du, dv);
      if (rho < s * radiusMod(Math.atan2(dv, du))) truthMask[(v + sy) * w + (u + sx)] = 1;
    }
    var truth = {
      label: p.label, params: p, bestFrame,
      center: { x: sx + ccx, y: sy + ccy }, mask: truthMask,
      sizeMm: 2 * a * s * mmPerPx,
    };
  }

  return { frames: out, w, h, mmPerPx, sector: { x: sx, y: sy, w: sw, h: sh }, truth };
}
