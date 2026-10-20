// Nodül maskesinden nicel özellikler: boyut, şekil, kenar, ekojenite, kompozisyon,
// ekojenik odaklar, posterior akustik bulgular ve doku (GLCM).
// Bu değerler hem TI-RADS önerisine hem de öğrenen modelin girdisine dönüşür.

import {
  gaussBlur, boxBlur, dilate, erode, labelComponents, radialContour, polygonArea,
  polygonPerimeter, convexHull, maskedStats, maskedPercentile, glcmFeatures,
  distanceInside, clamp,
} from './imgproc.js';

export function computeFeatures(gray, w, h, mask, mmPerPx) {
  const s1 = gaussBlur(gray, w, h, 1.0);
  const s2 = gaussBlur(gray, w, h, 2.0);

  let area = 0, sx = 0, sy = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    area++; sx += x; sy += y;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (area < 20) return null;
  const cx = sx / area, cy = sy / area;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const eqR = Math.sqrt(area / Math.PI);

  // --- Şekil ve kenar ---
  const pts = radialContour(mask, w, h, cx, cy, 256);
  const polyA = polygonArea(pts), polyP = polygonPerimeter(pts);
  const hull = convexHull(pts);
  const hullA = polygonArea(hull), hullP = polygonPerimeter(hull);
  const circularity = clamp((4 * Math.PI * polyA) / (polyP * polyP), 0, 1);
  const solidity = hullA > 0 ? clamp(polyA / hullA, 0, 1) : 1;
  const roughness = hullP > 0 ? polyP / hullP : 1;
  const rr = pts.map((p) => p.r);
  const meanR = rr.reduce((a, b) => a + b, 0) / rr.length || 1;
  let res2 = 0;
  const win = 10;
  for (let i = 0; i < rr.length; i++) {
    let a = 0;
    for (let k = -win; k <= win; k++) a += rr[(i + k + rr.length) % rr.length];
    const sm = a / (2 * win + 1);
    res2 += (rr[i] - sm) ** 2;
  }
  const radialIrreg = Math.sqrt(res2 / rr.length) / meanR;

  // --- Çevre parankim halkası ---
  const dIn = dilate(mask, w, h, 3);
  const dOut = dilate(mask, w, h, Math.max(8, Math.round(0.35 * eqR)));
  const ring = new Uint8Array(w * h);
  const halo = new Uint8Array(w * h);
  const dHalo = dilate(mask, w, h, 4);
  for (let i = 0; i < w * h; i++) {
    ring[i] = dOut[i] && !dIn[i] ? 1 : 0;
    halo[i] = dHalo[i] && !mask[i] ? 1 : 0;
  }
  const parenchyma = Math.max(20, maskedPercentile(s2, ring, 0.5));

  // --- Kompozisyon (kistik oran) ---
  const cysT = Math.max(14, 0.16 * parenchyma);
  let cyst = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (mask[i] && s2[i] < cysT) cyst[i] = 1;
  const cc = labelComponents(cyst, w, h);
  const keepC = new Set(cc.comps.filter((c) => c.area >= 8).map((c) => c.id));
  const solid = new Uint8Array(w * h);
  let cystN = 0;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) continue;
    if (cyst[i] && keepC.has(cc.labels[i])) cystN++; else { cyst[i] = 0; solid[i] = 1; }
  }
  const cysticFrac = cystN / area;
  const cystComps = keepC.size;

  // --- Ekojenite ---
  const solidStats = maskedStats(s2, solid);
  const echoRatio = solidStats.n > 10 ? solidStats.mean / parenchyma : 0;
  const s1Solid = maskedStats(s1, solid);
  const heterogeneity = s1Solid.mean > 0 ? s1Solid.std / s1Solid.mean : 0;
  const haloRatio = maskedStats(s1, halo).mean / parenchyma;

  // --- Ekojenik odaklar ---
  const solidMean = s1Solid.mean;
  const { punctate, macro, rimCoverage } = echogenicFoci(gray, w, h, mask, mmPerPx, parenchyma, eqR, cx, cy, solidMean);



  // --- Posterior akustik bulgu ---
  const band = Math.max(10, Math.round(0.5 * bh));
  const py0 = Math.min(h - 1, y1 + 3), py1 = Math.min(h - 1, y1 + 3 + band);
  const mean = (xa, xb) => {
    let a = 0, n = 0;
    for (let y = py0; y <= py1; y++) for (let x = Math.max(0, Math.round(xa)); x <= Math.min(w - 1, Math.round(xb)); x++) { a += s2[y * w + x]; n++; }
    return n ? a / n : NaN;
  };
  const center = mean(x0 + 0.2 * bw, x1 - 0.2 * bw);
  const sides = [mean(x0 - 0.6 * bw, x0 - 4), mean(x1 + 4, x1 + 0.6 * bw)].filter((v) => !Number.isNaN(v) && v > 0);
  const sideMean = sides.length ? sides.reduce((a, b) => a + b, 0) / sides.length : center;
  const posteriorRatio = py1 - py0 >= 4 && sideMean > 0 ? clamp(center / sideMean, 0.2, 3) : 1;

  // --- Kenar keskinliği ---
  const er = erode(mask, w, h, 1);
  let gsum = 0, gn = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    if (!mask[i] || er[i]) continue;
    const gx = s2[i + 1] - s2[i - 1], gy = s2[i + w] - s2[i - w];
    gsum += Math.hypot(gx, gy) / 2; gn++;
  }
  const delta = Math.max(12, Math.abs(maskedStats(s2, mask).mean - parenchyma));
  const edgeSharpness = clamp(((gn ? gsum / gn : 0) * 5) / delta, 0, 1.5);

  const glcm = glcmFeatures(s1, mask, w, h);

  const widthMm = bw * mmPerPx, heightMm = bh * mmPerPx;
  return {
    // boyut
    areaPx: area, areaMm2: area * mmPerPx * mmPerPx, widthMm, heightMm,
    maxDiamMm: Math.max(widthMm, heightMm), apRatio: bh / bw,
    bbox: { x0, y0, x1, y1 }, centroid: { x: cx, y: cy },
    // şekil / kenar
    circularity, solidity, roughness, radialIrreg, edgeSharpness,
    // ekojenite / kompozisyon
    parenchyma, solidMean, echoRatio, cysticFrac, cystComps, heterogeneity, haloRatio,
    // odaklar
    punctateCount: punctate, macroCount: macro, rimCoverage,
    // posterior
    posteriorRatio,
    // doku
    glcmContrast: glcm.contrast, glcmHomogeneity: glcm.homogeneity, glcmEnergy: glcm.energy,
    glcmEntropy: glcm.entropy, glcmCorrelation: glcm.correlation,
    contour: pts.filter((_, i) => i % 2 === 0).map((p) => ({ x: p.x, y: p.y })),
  };
}

// Ekojenik odaklar: ham görüntüde çok parlak, küçük ve yerel kontrastı yüksek bölgeler.
// Kenara yakın ve uzamış parlaklıklar periferik (rim) kalsifikasyon sayılır.
export function echogenicFoci(gray, w, h, mask, mmPerPx, parenchyma, eqR, cx, cy, nodMean = 0) {
  const local = boxBlur(gray, w, h, 4);
  const zone = dilate(mask, w, h, 2);
  const bright = new Uint8Array(w * h);
  const bt = Math.min(240, Math.max(200, 1.9 * parenchyma, 2.1 * nodMean));
  for (let i = 0; i < w * h; i++) if (zone[i] && gray[i] > bt && gray[i] - local[i] > 60) bright[i] = 1;
  const dist = distanceInside(mask, w, h);
  const bc = labelComponents(bright, w, h).comps.filter((c) => c.area >= 3);
  let punctate = 0, macro = 0;
  const rimBins = new Uint8Array(36);
  for (const c of bc) {
    const ix = Math.round(c.cx), iy = Math.round(c.cy);
    const dd = dist[iy * w + ix] || 0;
    const diamMm = 2 * Math.sqrt(c.area / Math.PI) * mmPerPx;
    const nearEdge = dd < Math.max(2.5, 0.12 * eqR);
    const elong = Math.max(c.x1 - c.x0 + 1, c.y1 - c.y0 + 1) / Math.max(1, Math.min(c.x1 - c.x0 + 1, c.y1 - c.y0 + 1));
    if (nearEdge && (elong > 2.2 || c.area > 25)) {
      const t = Math.atan2(c.cy - cy, c.cx - cx);
      const span = Math.max(1, Math.round(((Math.max(c.x1 - c.x0, c.y1 - c.y0) + 1) / (2 * Math.PI * eqR)) * 36));
      const b0 = Math.floor(((t + Math.PI) / (2 * Math.PI)) * 36);
      for (let k = 0; k < span; k++) rimBins[(b0 + k) % 36] = 1;
    } else if (diamMm <= 1.0) punctate++;
    else macro++;
  }
  const rimCoverage = rimBins.reduce((a, b) => a + b, 0) / 36;
  return { punctate, macro, rimCoverage };
}

// Arayüzde gösterim için gruplanmış özellik listesi.
export function featureRows(f) {
  const n = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  return [
    ['Boyut', [
      ['Transvers genişlik', `${n(f.widthMm, 1)} mm`],
      ['AP (derinlik) boyutu', `${n(f.heightMm, 1)} mm`],
      ['En büyük çap', `${n(f.maxDiamMm, 1)} mm`],
      ['Alan', `${n(f.areaMm2 / 100, 2)} cm²`],
    ]],
    ['Şekil ve kenar', [
      ['AP / transvers oranı', n(f.apRatio)],
      ['Dairesellik', n(f.circularity)],
      ['Solidite (dışbükeylik)', n(f.solidity)],
      ['Radyal düzensizlik', n(f.radialIrreg, 3)],
      ['Kenar keskinliği', n(f.edgeSharpness)],
    ]],
    ['Ekojenite ve içerik', [
      ['Ekojenite oranı (nodül/parankim)', n(f.echoRatio)],
      ['Kistik alan', `%${n(f.cysticFrac * 100, 0)}`],
      ['Kistik odak sayısı', String(f.cystComps)],
      ['Halo oranı', n(f.haloRatio)],
      ['İç heterojenite', n(f.heterogeneity)],
    ]],
    ['Ekojenik odaklar ve posterior', [
      ['Punktat odak (≤1 mm)', String(f.punctateCount)],
      ['Makrokalsifikasyon', String(f.macroCount)],
      ['Periferik kalsifikasyon kapsamı', `%${n(f.rimCoverage * 100, 0)}`],
      ['Posterior oran (gölge <1 < güçlenme)', n(f.posteriorRatio)],
    ]],
    ['Doku (GLCM)', [
      ['Kontrast', n(f.glcmContrast)],
      ['Homojenlik', n(f.glcmHomogeneity)],
      ['Entropi', n(f.glcmEntropy)],
    ]],
  ];
}
