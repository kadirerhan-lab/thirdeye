// ACR TI-RADS (2017) puanlama, kategori, İİAB / takip önerisi ve
// görüntü özelliklerinden otomatik tanımlayıcı önerisi.

export const TIRADS = {
  composition: {
    label: 'Kompozisyon',
    options: [
      { id: 'cystic', label: 'Kistik / neredeyse tamamen kistik', pts: 0 },
      { id: 'spongiform', label: 'Süngerimsi (spongiform)', pts: 0 },
      { id: 'mixed', label: 'Mikst kistik-solid', pts: 1 },
      { id: 'solid', label: 'Solid / neredeyse tamamen solid', pts: 2 },
    ],
  },
  echogenicity: {
    label: 'Ekojenite',
    options: [
      { id: 'anechoic', label: 'Anekoik', pts: 0 },
      { id: 'hyperiso', label: 'Hiperekoik veya izoekoik', pts: 1 },
      { id: 'hypo', label: 'Hipoekoik', pts: 2 },
      { id: 'veryhypo', label: 'Çok hipoekoik', pts: 3 },
    ],
  },
  shape: {
    label: 'Şekil (transvers planda)',
    options: [
      { id: 'wider', label: 'Genişliği yüksekliğinden fazla', pts: 0 },
      { id: 'taller', label: 'Yüksekliği genişliğinden fazla', pts: 3 },
    ],
  },
  margin: {
    label: 'Kenar',
    options: [
      { id: 'smooth', label: 'Düzgün', pts: 0 },
      { id: 'illdefined', label: 'Belirsiz', pts: 0 },
      { id: 'lobulated', label: 'Lobüle veya düzensiz', pts: 2 },
      { id: 'ete', label: 'Ekstratiroidal uzanım', pts: 3 },
    ],
  },
  foci: {
    label: 'Ekojenik odaklar',
    multi: true,
    options: [
      { id: 'none', label: 'Yok / büyük kuyruklu yıldız artefaktı', pts: 0 },
      { id: 'macro', label: 'Makrokalsifikasyon', pts: 1 },
      { id: 'rim', label: 'Periferik (rim) kalsifikasyon', pts: 2 },
      { id: 'punctate', label: 'Punktat ekojenik odaklar', pts: 3 },
    ],
  },
};

// Kategori başına malignite oranları: Middleton ve ark., AJR 2017;208:1331
// (ACR TI-RADS çok merkezli doğrulama, 3422 nodül).
export const TR_INFO = {
  TR1: { name: 'Benign', risk: 0.003, fna: null, follow: null },
  TR2: { name: 'Şüpheli değil', risk: 0.015, fna: null, follow: null },
  TR3: { name: 'Hafif şüpheli', risk: 0.048, fna: 25, follow: 15, schedule: '1., 3. ve 5. yıllarda' },
  TR4: { name: 'Orta derecede şüpheli', risk: 0.091, fna: 15, follow: 10, schedule: '1., 2., 3. ve 5. yıllarda' },
  TR5: { name: 'Yüksek derecede şüpheli', risk: 0.35, fna: 10, follow: 5, schedule: '5 yıl boyunca yılda bir' },
};

const pts = (field, id) => TIRADS[field].options.find((o) => o.id === id)?.pts ?? 0;

export function categoryFromPoints(p) {
  // ACR: 0 puan TR1, 2 puan TR2, 3 puan TR3, 4–6 puan TR4, ≥7 puan TR5.
  // 1 puan ACR tablosunda ayrıca tanımlı değildir; yaygın uygulamada TR1 kabul edilir.
  if (p <= 1) return 'TR1';
  if (p === 2) return 'TR2';
  if (p === 3) return 'TR3';
  if (p <= 6) return 'TR4';
  return 'TR5';
}

export function scoreTirads(sel) {
  const foci = (sel.foci || []).filter((f) => f !== 'none');
  const breakdown = {
    composition: pts('composition', sel.composition),
    echogenicity: pts('echogenicity', sel.echogenicity),
    shape: pts('shape', sel.shape),
    margin: pts('margin', sel.margin),
    foci: foci.reduce((a, f) => a + pts('foci', f), 0),
  };
  let note = null;
  if (sel.composition === 'spongiform') {
    // ACR: süngerimsi nodüllerde diğer kategorilerden puan eklenmez.
    for (const k of Object.keys(breakdown)) breakdown[k] = 0;
    note = 'Süngerimsi nodül: ACR kuralına göre diğer kategorilerden puan eklenmez.';
  }
  const points = Object.values(breakdown).reduce((a, b) => a + b, 0);
  const category = categoryFromPoints(points);
  return { points, category, breakdown, note, info: TR_INFO[category] };
}

export function recommendation(category, maxDiamMm) {
  const info = TR_INFO[category];
  const f1 = (v) => v.toFixed(1).replace('.', ',');
  const cm = f1(maxDiamMm / 10);
  if (!info.fna) return { action: 'none', text: `${category}: İİAB önerilmez (${cm} cm).` };
  if (maxDiamMm >= info.fna) return { action: 'fna', text: `${category}, ${cm} cm ≥ ${f1(info.fna / 10)} cm: İİAB (ince iğne aspirasyon biyopsisi) önerilir.` };
  if (maxDiamMm >= info.follow) return { action: 'follow', text: `${category}, ${cm} cm ≥ ${f1(info.follow / 10)} cm: ultrason takibi önerilir (${info.schedule}).` };
  return { action: 'none', text: `${category}, ${cm} cm: boyut eşiklerinin altında; İİAB veya takip gerekmez.` };
}

// Görüntü özelliklerinden TI-RADS tanımlayıcı önerisi. Her alan için gerekçe ve güven döner.
// Ekstratiroidal uzanım ve kuyruklu yıldız artefaktı otomatik önerilmez (anatomik bağlam gerekir).
const tr = (v, d = 2) => v.toFixed(d).replace('.', ',');

export function suggestTirads(f) {
  const conf = (d) => Math.max(0.5, Math.min(0.97, 0.5 + Math.abs(d)));
  const sel = {}, why = {}, c = {};

  if (f.cysticFrac >= 0.4 && f.cystComps >= 7) {
    sel.composition = 'spongiform'; c.composition = conf((f.cystComps - 7) / 10);
    why.composition = `${f.cystComps} küçük kistik odak, kistik alan %${Math.round(f.cysticFrac * 100)}`;
  } else if (f.cysticFrac >= 0.85) {
    sel.composition = 'cystic'; c.composition = conf((f.cysticFrac - 0.85) * 4);
    why.composition = `Kistik alan %${Math.round(f.cysticFrac * 100)}`;
  } else if (f.cysticFrac >= 0.05) {
    sel.composition = 'mixed'; c.composition = conf(Math.min(f.cysticFrac - 0.05, 0.85 - f.cysticFrac) * 3);
    why.composition = `Kistik alan %${Math.round(f.cysticFrac * 100)}`;
  } else {
    sel.composition = 'solid'; c.composition = conf((0.05 - f.cysticFrac) * 8);
    why.composition = `Kistik alan %${Math.round(f.cysticFrac * 100)} (<%5)`;
  }

  const er = f.echoRatio;
  if (f.cysticFrac >= 0.85) { sel.echogenicity = 'anechoic'; c.echogenicity = 0.8; why.echogenicity = 'Kistik içerik'; }
  else if (er < 0.5) { sel.echogenicity = 'veryhypo'; c.echogenicity = conf((0.5 - er) * 4); }
  else if (er < 0.85) { sel.echogenicity = 'hypo'; c.echogenicity = conf(Math.min(er - 0.5, 0.85 - er) * 4); }
  else { sel.echogenicity = 'hyperiso'; c.echogenicity = conf((er - 0.85) * 4); }
  if (!why.echogenicity) why.echogenicity = `Solid kısım / parankim = ${tr(er, 2)}`;

  sel.shape = f.apRatio > 1 ? 'taller' : 'wider';
  c.shape = conf((f.apRatio - 1) * 3);
  why.shape = `AP / transvers = ${tr(f.apRatio, 2)}`;

  const irregular = f.solidity < 0.9 || f.radialIrreg > 0.03;
  if (irregular) {
    sel.margin = 'lobulated';
    c.margin = conf(Math.max(0.9 - f.solidity, f.radialIrreg - 0.03) * 6);
    why.margin = `Solidite ${tr(f.solidity, 2)}, radyal düzensizlik ${tr(f.radialIrreg, 3)}`;
  } else if (f.edgeSharpness < 0.35) {
    sel.margin = 'illdefined'; c.margin = conf((0.35 - f.edgeSharpness) * 2);
    why.margin = `Kenar keskinliği düşük (${tr(f.edgeSharpness, 2)})`;
  } else {
    sel.margin = 'smooth'; c.margin = conf(Math.min(f.solidity - 0.9, 0.03 - f.radialIrreg) * 6);
    why.margin = `Solidite ${tr(f.solidity, 2)}, keskin sınır`;
  }

  const foci = [];
  if (f.punctateCount >= 1) foci.push('punctate');
  if (f.macroCount >= 1) foci.push('macro');
  if (f.rimCoverage >= 0.25) foci.push('rim');
  sel.foci = foci.length ? foci : ['none'];
  c.foci = foci.length ? conf(Math.min(0.45, 0.12 * (f.punctateCount + f.macroCount) + f.rimCoverage)) : 0.7;
  why.foci = `Punktat ${f.punctateCount}, makro ${f.macroCount}, periferik %${Math.round(f.rimCoverage * 100)}`;

  return { sel, conf: c, why };
}
