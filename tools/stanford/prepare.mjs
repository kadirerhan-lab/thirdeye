// Stanford verisinden ThirdEye özelliklerini ve model yamalarını çıkarır (eğitim öncesi hazırlık).
// Kullanım: node tools/stanford/prepare.mjs --data data/stanford
// Çıktı: <data>/prepared/cases.json (özellik vektörleri, gruplar, etiketler, radyolog/otomatik TI-RADS)
//        <data>/prepared/patches.bin (N×64×64×2 uint8 yama)
// Ardından: python tools/stanford/train_torch.py --data data/stanford  →  node tools/stanford/finalize.mjs --data data/stanford
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { rocAuc } from '../../web/js/model.js';
import { join } from 'node:path';
import { computeFeatures } from '../../web/js/features.js';
import { makePatch } from '../../web/js/pipeline.js';
import { segmentFromSeed } from '../../web/js/detect.js';
import { suggestTirads, scoreTirads, TIRADS } from '../../web/js/tirads.js';
import { featureVector, FEATURES } from '../../web/js/model.js';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const DATA = arg('data', 'data/stanford');


function readNpy(path) {
  const b = readFileSync(path);
  const major = b[6];
  const hl = major === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const off = major === 1 ? 10 : 12;
  const header = b.toString('latin1', off, off + hl);
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  if (!/'descr':\s*'\|u1'/.test(header)) throw new Error(`${path}: yalnızca uint8 desteklenir (${header})`);
  return { shape, data: new Uint8Array(b.buffer, b.byteOffset + off + hl, shape.reduce((a, c) => a * c, 1)) };
}

const index = JSON.parse(readFileSync(join(DATA, 'index.json'), 'utf8'));
const nodules = index.nodules.filter((n) => n.label === 0 || n.label === 1);
console.log(`${nodules.length} etiketli nodül (${nodules.filter((n) => n.label === 1).length} kötü huylu)`);

// Radyoloğun TI-RADS tanımlayıcıları (sütun değerleri ACR puanı olarak yorumlanır).
const radiologistPoints = (m) => {
  const cols = ['composition', 'echogenicity', 'shape', 'margin', 'echogenicfoci'];
  let s = 0, found = 0;
  for (const c of cols) {
    const key = Object.keys(m).find((k) => k.toLowerCase().replace(/[^a-z]/g, '').endsWith(c));
    if (key != null && m[key] != null && !Number.isNaN(+m[key])) { s += +m[key]; found++; }
  }
  return found ? s : null;
};
const sizeMm = (m) => {
  const v = ['size_x', 'size_y', 'size_z'].map((k) => +m[k]).filter((x) => x > 0);
  if (!v.length) return null;
  const mx = Math.max(...v);
  return mx < 10 ? mx * 10 : mx; // cm ise mm'ye çevir
};

const cases = [];
const perNodule = [];
const dice = [];
let t0 = Date.now();
for (const [k, n] of nodules.entries()) {
  const img = readNpy(join(DATA, `${n.file}_img.npy`));
  const msk = readNpy(join(DATA, `${n.file}_mask.npy`));
  const [F, h, w] = img.shape;
  // Ölçek: en büyük kesitteki maske uzunluğu ↔ kayıtlı en büyük nodül boyutu.
  let bestF = 0, bestA = 0;
  for (let f = 0; f < F; f++) { let a = 0; const o = f * h * w; for (let i = 0; i < h * w; i++) a += msk.data[o + i]; if (a > bestA) { bestA = a; bestF = f; } }
  const ext = (f) => { let x0 = w, x1 = -1, y0 = h, y1 = -1; const o = f * h * w; for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (msk.data[o + y * w + x]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } return Math.max(x1 - x0 + 1, y1 - y0 + 1); };
  const sMm = sizeMm(n.meta);
  const mmPerPx = sMm ? sMm / ext(bestF) : 0.1;
  let featBest = null;
  for (let f = 0; f < F; f++) {
    const gray = Float32Array.from(img.data.subarray(f * h * w, (f + 1) * h * w));
    const mask = Uint8Array.from(msk.data.subarray(f * h * w, (f + 1) * h * w));
    const feats = computeFeatures(gray, w, h, mask, mmPerPx);
    if (!feats) continue;
    const patch = makePatch(gray, w, h, mask, feats).data;
    delete feats.contour;
    cases.push({ group: n.annot_id, label: n.label, features: feats, patch, source: 'clinical' });
    if (f === bestF) {
      featBest = feats;
      // Otomatik segmentasyonun uzman maskesine uyumu (tohum: uzman maskesinin merkezi).
      const seg = segmentFromSeed(gray, w, h, feats.centroid.x, feats.centroid.y);
      let inter = 0, a = 0, b = 0;
      if (seg.ok) for (let i = 0; i < h * w; i++) { a += seg.mask[i]; b += mask[i]; if (seg.mask[i] && mask[i]) inter++; }
      dice.push(seg.ok ? (2 * inter) / (a + b) : 0);
    }
  }
  if (featBest) {
    const sug = suggestTirads(featBest);
    perNodule.push({ id: n.annot_id, label: n.label, radPts: radiologistPoints(n.meta), autoPts: scoreTirads(sug.sel).points, autoSel: sug.sel, meta: n.meta });
  }
  if (k % 10 === 9) console.log(`  özellik çıkarımı ${k + 1}/${nodules.length} (${((Date.now() - t0) / 1000).toFixed(0)} sn)`);
}
console.log(`${cases.length} kare örneği, ${perNodule.length} nodül hazır`);

// --- Referanslar: radyolog TI-RADS ve otomatik TI-RADS ---
const aucOf = (rows, key) => { const r = rows.filter((x) => x[key] != null); return rocAuc(r.map((x) => x[key]), r.map((x) => x.label)).auc; };
const radAUC = aucOf(perNodule, 'radPts');
const autoAUC = aucOf(perNodule, 'autoPts');
const descriptorAgreement = {};
for (const field of ['composition', 'echogenicity', 'shape', 'margin', 'echogenicfoci']) {
  const key = (m) => Object.keys(m).find((k) => k.toLowerCase().replace(/[^a-z]/g, '').endsWith(field));
  let ok = 0, n = 0;
  for (const p of perNodule) {
    const k = key(p.meta);
    if (k == null || p.meta[k] == null) continue;
    const tf2 = field === 'echogenicfoci' ? 'foci' : field;
    const sel = p.autoSel[tf2];
    const pts = Array.isArray(sel) ? sel.reduce((a, id) => a + (TIRADS.foci.options.find((o) => o.id === id)?.pts ?? 0), 0) : TIRADS[tf2].options.find((o) => o.id === sel)?.pts;
    n++; if (+p.meta[k] === pts) ok++;
  }
  descriptorAgreement[field] = n ? ok / n : null;
}

// --- Çıktılar ---
const outDir = join(DATA, 'prepared');
mkdirSync(outDir, { recursive: true });
const patches = Buffer.alloc(cases.length * 64 * 64 * 2);
cases.forEach((c, i) => patches.set(c.patch, i * 64 * 64 * 2));
writeFileSync(join(outDir, 'patches.bin'), patches);
writeFileSync(join(outDir, 'cases.json'), JSON.stringify({
  featureKeys: FEATURES.map((f) => f[0]),
  rows: cases.map((c) => ({ group: c.group, label: c.label, fv: featureVector(c.features) })),
  nodules: perNodule.map(({ meta, ...p }) => p),
  reference: { radiologistAUC: radAUC, autoTiradsAUC: autoAUC, descriptorAgreement, diceMean: dice.reduce((a, b) => a + b, 0) / dice.length, diceOver70: dice.filter((d) => d >= 0.7).length / dice.length },
}));
console.log(`Hazırlandı: ${outDir}  (radyolog TI-RADS AUC ${radAUC?.toFixed(3)}, otomatik TI-RADS AUC ${autoAUC?.toFixed(3)})`);
