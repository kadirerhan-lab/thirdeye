// Öğrenen model (TensorFlow.js, tarayıcıda eğitim ve çıkarım).
//
// Mimari: iki girişli küçük bir ağ
//   • Görüntü dalı: 64×64×2 yama (gri ton + nodül maskesi) → 3 evrişim bloğu → global ortalama havuzlama
//   • Özellik dalı: 20 nicel görüntü özelliği (standartlaştırılmış) → yoğun katman
//   • Birleştirme → dropout → yoğun → sigmoid (malignite olasılığı)
// Hekimin girdiği kesin tanılarla (sitoloji / patoloji) veri seti büyüdükçe yeniden eğitilir;
// her sürümün doğrulama metrikleri saklanır ve skordaki ağırlığı bu metriklere bağlıdır.

import { PATCH } from './pipeline.js';
import { rng } from './imgproc.js';

const MODEL_URL = 'indexeddb://thirdeye-model';

export const FEATURES = [
  ['maxDiamMm', 'En büyük çap', Math.log1p],
  ['apRatio', 'AP / transvers oranı'],
  ['circularity', 'Dairesellik'],
  ['solidity', 'Solidite'],
  ['roughness', 'Kontur pürüzlülüğü'],
  ['radialIrreg', 'Radyal düzensizlik'],
  ['edgeSharpness', 'Kenar keskinliği'],
  ['echoRatio', 'Ekojenite oranı'],
  ['cysticFrac', 'Kistik alan'],
  ['cystComps', 'Kistik odak sayısı', Math.log1p],
  ['heterogeneity', 'İç heterojenite'],
  ['haloRatio', 'Halo oranı'],
  ['punctateCount', 'Punktat odaklar', Math.log1p],
  ['macroCount', 'Makrokalsifikasyon', Math.log1p],
  ['rimCoverage', 'Periferik kalsifikasyon'],
  ['posteriorRatio', 'Posterior akustik bulgu'],
  ['glcmContrast', 'Doku kontrastı'],
  ['glcmHomogeneity', 'Doku homojenliği'],
  ['glcmEntropy', 'Doku entropisi'],
  ['glcmCorrelation', 'Doku korelasyonu'],
];

export function featureVector(f) {
  return FEATURES.map(([k, , tr]) => {
    const v = Number(f[k]) || 0;
    return tr ? tr(v) : v;
  });
}

// TensorFlow.js kaynağı: tarayıcıda T(), Node'da (eğitim betikleri) setTF ile verilir.
let TF = null;
export const setTF = (t) => { TF = t; };
const T = () => TF ?? (typeof window !== 'undefined' ? window.tf : null);
export const hasTF = () => !!T();

export function buildModel(nFeat) {
  const tf = T();
  const imgIn = tf.input({ shape: [PATCH, PATCH, 2], name: 'patch' });
  let x = imgIn;
  for (const filters of [8, 16, 32]) {
    x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
    x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);
  }
  x = tf.layers.globalAveragePooling2d({}).apply(x);
  const featIn = tf.input({ shape: [nFeat], name: 'features' });
  const f = tf.layers.dense({ units: 16, activation: 'relu' }).apply(featIn);
  let z = tf.layers.concatenate().apply([x, f]);
  z = tf.layers.dropout({ rate: 0.3 }).apply(z);
  z = tf.layers.dense({ units: 16, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 1e-3 }) }).apply(z);
  const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(z);
  const model = tf.model({ inputs: [imgIn, featIn], outputs: out });
  model.compile({ optimizer: tf.train.adam(2e-3), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return model;
}

function patchToFloat(u8, flip = false) {
  const out = new Float32Array(PATCH * PATCH * 2);
  for (let y = 0; y < PATCH; y++) for (let x = 0; x < PATCH; x++) {
    const sx = flip ? PATCH - 1 - x : x;
    const si = (y * PATCH + sx) * 2, di = (y * PATCH + x) * 2;
    out[di] = u8[si] / 255;
    out[di + 1] = u8[si + 1] / 255;
  }
  return out;
}

export function rocAuc(scores, labels) {
  const pairs = scores.map((s, i) => ({ s, y: labels[i] })).sort((a, b) => b.s - a.s);
  const P = labels.filter((y) => y === 1).length, N = labels.length - P;
  if (!P || !N) return { auc: null, roc: [] };
  let tp = 0, fp = 0, auc = 0, prevFpr = 0, prevTpr = 0;
  const roc = [{ fpr: 0, tpr: 0, t: 1 }];
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].y === 1) tp++; else fp++;
    if (i < pairs.length - 1 && pairs[i + 1].s === pairs[i].s) continue;
    const fpr = fp / N, tpr = tp / P;
    auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
    roc.push({ fpr, tpr, t: pairs[i].s });
    prevFpr = fpr; prevTpr = tpr;
  }
  return { auc, roc };
}

// Grup (nodül) bazında tabakalı ayrım: aynı nodülün kareleri aynı tarafta kalır.
const groupOf = (c, i) => c.group ?? c.id ?? `i${i}`;
export function groupFolds(cases, k, seed = 42) {
  const groups = new Map();
  cases.forEach((c, i) => { const g = groupOf(c, i); if (!groups.has(g)) groups.set(g, { label: c.label, idx: [] }); groups.get(g).idx.push(i); });
  const r = rng(seed);
  const folds = Array.from({ length: k }, () => []);
  for (const lab of [0, 1]) {
    const gs = [...groups.values()].filter((g) => g.label === lab);
    for (let i = gs.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [gs[i], gs[j]] = [gs[j], gs[i]]; }
    gs.forEach((g, n) => folds[n % k].push(...g.idx));
  }
  return folds;
}

export function normStats(cases) {
  const F = FEATURES.length;
  const fv = cases.map((c) => featureVector(c.features));
  const mean = new Array(F).fill(0), std = new Array(F).fill(0);
  for (const v of fv) v.forEach((x, i) => { mean[i] += x / fv.length; });
  for (const v of fv) v.forEach((x, i) => { std[i] += (x - mean[i]) ** 2 / fv.length; });
  for (let i = 0; i < F; i++) std[i] = Math.sqrt(std[i]) || 1;
  return { mean, std };
}

function tensors(cases, mean, std, augment) {
  const tf = T();
  const F = FEATURES.length;
  let items = cases.map((c) => ({ c, flip: false }));
  if (augment) {
    // Sınıf dengesi için azınlık sınıfı çoğaltılır; her vakanın yatay aynası da eklenir.
    const t1 = cases.filter((c) => c.label === 1), t0 = cases.filter((c) => c.label === 0);
    const big = Math.max(t1.length, t0.length);
    items = [];
    for (const g of [t0, t1]) for (let i = 0; i < big; i++) { items.push({ c: g[i % g.length], flip: false }); items.push({ c: g[i % g.length], flip: true }); }
  }
  const xi = new Float32Array(items.length * PATCH * PATCH * 2);
  const xf = new Float32Array(items.length * F);
  const y = new Float32Array(items.length);
  items.forEach((it, k) => {
    xi.set(patchToFloat(it.c.patch, it.flip), k * PATCH * PATCH * 2);
    xf.set(featureVector(it.c.features).map((x, i) => (x - mean[i]) / std[i]), k * F);
    y[k] = it.c.label;
  });
  return { xi: tf.tensor4d(xi, [items.length, PATCH, PATCH, 2]), xf: tf.tensor2d(xf, [items.length, F]), y: tf.tensor2d(y, [items.length, 1]) };
}

// Bir eğitim / doğrulama ayrımında modeli eğitir, doğrulama tahminlerini döndürür.
export async function fitModel(train, val, { epochs = 30, onEpoch = null, batchSize = 32 } = {}) {
  const tf = T();
  const { mean, std } = normStats(train);
  const tr = tensors(train, mean, std, true);
  const va = val.length ? tensors(val, mean, std, false) : null;
  const model = buildModel(FEATURES.length);
  const history = [];
  try {
    await model.fit([tr.xi, tr.xf], tr.y, {
      epochs, batchSize, shuffle: true, validationData: va ? [[va.xi, va.xf], va.y] : undefined,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          history.push({ epoch: epoch + 1, loss: logs.loss, valLoss: logs.val_loss ?? logs.loss, acc: logs.acc ?? logs.accuracy, valAcc: logs.val_acc ?? logs.val_accuracy });
          if (onEpoch) onEpoch(history);
          if (tf.nextFrame && typeof requestAnimationFrame !== 'undefined') await tf.nextFrame();
        },
      },
    });
    let preds = [];
    if (va) { const p = model.predict([va.xi, va.xf]); preds = Array.from(await p.data()); p.dispose(); }
    return { model, mean, std, preds, history };
  } finally {
    tf.dispose([tr.xi, tr.xf, tr.y, ...(va ? [va.xi, va.xf, va.y] : [])]);
  }
}

// Kare düzeyindeki tahminleri nodül düzeyinde ortalayıp metrikleri hesaplar.
export function evaluate(cases, preds) {
  const by = new Map();
  cases.forEach((c, i) => { const g = groupOf(c, i); if (!by.has(g)) by.set(g, { y: c.label, p: [] }); by.get(g).p.push(preds[i]); });
  const ps = [], ys = [];
  for (const v of by.values()) { ps.push(v.p.reduce((a, b) => a + b, 0) / v.p.length); ys.push(v.y); }
  const { auc, roc } = rocAuc(ps, ys);
  let best = { j: -1, t: 0.5, sens: 0, spec: 0 };
  for (const pt of roc) { const j = pt.tpr - pt.fpr; if (j > best.j) best = { j, t: pt.t, sens: pt.tpr, spec: 1 - pt.fpr }; }
  const P = ys.filter((v) => v === 1).length, N = ys.length - P;
  let tp = 0, tn = 0;
  ps.forEach((p, i) => { if (p >= 0.5 && ys[i] === 1) tp++; if (p < 0.5 && ys[i] === 0) tn++; });
  return { auc, roc, youden: best, sens50: P ? tp / P : null, spec50: N ? tn / N : null, nVal: ys.length, valPredictions: ps.map((p, i) => [p, ys[i]]) };
}

export async function trainModel(cases, { epochs = 30, onEpoch = null, prevMeta = null, saveTo = MODEL_URL } = {}) {
  const labeled = cases.filter((c) => c.label === 0 || c.label === 1);
  const n1 = labeled.filter((c) => c.label === 1).length, n0 = labeled.length - n1;
  if (n0 < 8 || n1 < 8) throw new Error(`Eğitim için her sınıftan en az 8 vaka gerekir (şu an: ${n0} iyi huylu, ${n1} kötü huylu).`);
  const folds = groupFolds(labeled, 5, 42);
  const valIdx = new Set(folds[0]);
  const train = labeled.filter((_, i) => !valIdx.has(i)), val = labeled.filter((_, i) => valIdx.has(i));
  const { model, mean, std, preds, history } = await fitModel(train, val, { epochs, onEpoch });
  const metrics = evaluate(val, preds);
  if (saveTo) await model.save(saveTo);
  const prevHist = prevMeta?.history ?? [];
  const version = (prevMeta?.version ?? 0) + 1;
  const nClinical = labeled.filter((c) => c.source === 'clinical').length;
  const meta = {
    version, date: new Date().toISOString(), mean, std, featureKeys: FEATURES.map((f) => f[0]),
    nTrain: train.length, nVal: val.length, nBenign: n0, nMalignant: n1, nClinical,
    trainPrevalence: 0.5, metrics, trainHistory: history, origin: 'local',
    history: [...prevHist, { version, date: new Date().toISOString(), n: labeled.length, nClinical, auc: metrics.auc, sens: metrics.youden.sens, spec: metrics.youden.spec }],
  };
  return { model, meta };
}

export async function loadModel() {
  if (!hasTF()) return null;
  try { return await T().loadLayersModel(MODEL_URL); } catch { return null; }
}

export async function deleteModel() {
  if (!hasTF()) return;
  try { await T().io.removeModel(MODEL_URL); } catch { /* yok */ }
}

export async function exportModel(model) {
  await model.save('downloads://thirdeye-model');
}

// Tahmin + açıklanabilirlik: özellik katkıları (özelliği eğitim ortalamasına çekince
// log-olasılıktaki değişim) ve yama üzerindeki gradyan tabanlı dikkat haritası.
export function predict(model, meta, features, patchU8) {
  const tf = T();
  const F = FEATURES.length;
  const z = featureVector(features).map((x, i) => (x - meta.mean[i]) / meta.std[i]);
  const rows = [z];
  for (let i = 0; i < F; i++) { const r = z.slice(); r[i] = 0; rows.push(r); }
  const img = patchToFloat(patchU8);
  return tf.tidy(() => {
    const ti = tf.tensor4d(img, [1, PATCH, PATCH, 2]);
    const tiRep = ti.tile([rows.length, 1, 1, 1]);
    const tf2 = tf.tensor2d(rows.flat(), [rows.length, F]);
    const out = model.predict([tiRep, tf2]).dataSync();
    const lg = (p) => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
    const p = out[0];
    const contributions = FEATURES.map(([key, label], i) => ({ key, label, delta: lg(p) - lg(out[i + 1]), z: z[i] }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    const featT = tf.tensor2d([z], [1, F]);
    const g = tf.grad((x) => model.apply([x, featT], { training: false }).sum())(ti);
    const gd = g.abs().sum(3).dataSync();
    let mx = 0;
    for (const v of gd) mx = Math.max(mx, v);
    const saliency = Float32Array.from(gd, (v) => (mx ? v / mx : 0));
    return { p, contributions, saliency };
  });
}
