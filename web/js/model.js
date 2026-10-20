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

export const hasTF = () => typeof window !== 'undefined' && !!window.tf;

function buildModel(nFeat) {
  const tf = window.tf;
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

function stratifiedSplit(cases, valFrac, seed) {
  const r = rng(seed);
  const by = [[], []];
  for (const c of cases) by[c.label].push(c);
  const train = [], val = [];
  for (const g of by) {
    const a = g.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    const nv = Math.round(a.length * valFrac);
    val.push(...a.slice(0, nv)); train.push(...a.slice(nv));
  }
  return { train, val };
}

export async function trainModel(cases, { epochs = 30, onEpoch = null, prevMeta = null } = {}) {
  const tf = window.tf;
  const labeled = cases.filter((c) => c.label === 0 || c.label === 1);
  const n1 = labeled.filter((c) => c.label === 1).length, n0 = labeled.length - n1;
  if (n0 < 8 || n1 < 8) throw new Error(`Eğitim için her sınıftan en az 8 vaka gerekir (şu an: ${n0} iyi huylu, ${n1} kötü huylu).`);

  const { train, val } = stratifiedSplit(labeled, 0.2, 42);
  const fv = train.map((c) => featureVector(c.features));
  const F = FEATURES.length;
  const mean = new Array(F).fill(0), std = new Array(F).fill(0);
  for (const v of fv) v.forEach((x, i) => { mean[i] += x / fv.length; });
  for (const v of fv) v.forEach((x, i) => { std[i] += (x - mean[i]) ** 2 / fv.length; });
  for (let i = 0; i < F; i++) std[i] = Math.sqrt(std[i]) || 1;
  const norm = (f) => featureVector(f).map((x, i) => (x - mean[i]) / std[i]);

  // Sınıf dengesi için azınlık sınıfı çoğaltılır; her vakanın yatay aynası da eklenir.
  const t1 = train.filter((c) => c.label === 1), t0 = train.filter((c) => c.label === 0);
  const big = Math.max(t1.length, t0.length);
  const balanced = [];
  for (const g of [t0, t1]) for (let i = 0; i < big; i++) balanced.push(g[i % g.length]);
  const items = [];
  for (const c of balanced) { items.push({ c, flip: false }); items.push({ c, flip: true }); }

  const xi = new Float32Array(items.length * PATCH * PATCH * 2);
  const xf = new Float32Array(items.length * F);
  const y = new Float32Array(items.length);
  items.forEach((it, k) => {
    xi.set(patchToFloat(it.c.patch, it.flip), k * PATCH * PATCH * 2);
    xf.set(norm(it.c.features), k * F);
    y[k] = it.c.label;
  });
  const vxi = new Float32Array(val.length * PATCH * PATCH * 2);
  const vxf = new Float32Array(val.length * F);
  const vy = new Float32Array(val.length);
  val.forEach((c, k) => { vxi.set(patchToFloat(c.patch), k * PATCH * PATCH * 2); vxf.set(norm(c.features), k * F); vy[k] = c.label; });

  const model = buildModel(F);
  const tXi = tf.tensor4d(xi, [items.length, PATCH, PATCH, 2]);
  const tXf = tf.tensor2d(xf, [items.length, F]);
  const tY = tf.tensor2d(y, [items.length, 1]);
  const tVi = tf.tensor4d(vxi, [val.length, PATCH, PATCH, 2]);
  const tVf = tf.tensor2d(vxf, [val.length, F]);
  const tVy = tf.tensor2d(vy, [val.length, 1]);
  const history = [];
  try {
    await model.fit([tXi, tXf], tY, {
      epochs, batchSize: 32, shuffle: true, validationData: [[tVi, tVf], tVy],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          history.push({ epoch: epoch + 1, loss: logs.loss, valLoss: logs.val_loss, acc: logs.acc ?? logs.accuracy, valAcc: logs.val_acc ?? logs.val_accuracy });
          if (onEpoch) onEpoch(history);
          await tf.nextFrame();
        },
      },
    });
    const pv = Array.from(await model.predict([tVi, tVf]).data());
    const labels = Array.from(vy);
    const { auc, roc } = rocAuc(pv, labels);
    // Youden J ile en iyi eşik ve o eşikteki duyarlılık / özgüllük
    let best = { j: -1, t: 0.5, sens: 0, spec: 0 };
    const P = labels.filter((v) => v === 1).length, N = labels.length - P;
    for (const pt of roc) {
      const j = pt.tpr - pt.fpr;
      if (j > best.j) best = { j, t: pt.t, sens: pt.tpr, spec: 1 - pt.fpr };
    }
    let tp = 0, tn = 0;
    pv.forEach((p, i) => { if (p >= 0.5 && labels[i] === 1) tp++; if (p < 0.5 && labels[i] === 0) tn++; });
    const metrics = {
      auc, roc, youden: best, sens50: P ? tp / P : null, spec50: N ? tn / N : null,
      nVal: labels.length, valPredictions: pv.map((p, i) => [p, labels[i]]),
    };
    await model.save(MODEL_URL);
    const prevHist = prevMeta?.history ?? [];
    const version = (prevMeta?.version ?? 0) + 1;
    const nClinical = labeled.filter((c) => c.source === 'clinical').length;
    const meta = {
      version, date: new Date().toISOString(), mean, std, featureKeys: FEATURES.map((f) => f[0]),
      nTrain: train.length, nVal: val.length, nBenign: n0, nMalignant: n1, nClinical,
      trainPrevalence: 0.5, metrics, trainHistory: history,
      history: [...prevHist, { version, date: new Date().toISOString(), n: labeled.length, nClinical, auc, sens: best.sens, spec: best.spec }],
    };
    return { model, meta };
  } finally {
    tf.dispose([tXi, tXf, tY, tVi, tVf, tVy]);
  }
}

export async function loadModel() {
  if (!hasTF()) return null;
  try { return await window.tf.loadLayersModel(MODEL_URL); } catch { return null; }
}

export async function deleteModel() {
  if (!hasTF()) return;
  try { await window.tf.io.removeModel(MODEL_URL); } catch { /* yok */ }
}

export async function exportModel(model) {
  await model.save('downloads://thirdeye-model');
}

// Tahmin + açıklanabilirlik: özellik katkıları (özelliği eğitim ortalamasına çekince
// log-olasılıktaki değişim) ve yama üzerindeki gradyan tabanlı dikkat haritası.
export function predict(model, meta, features, patchU8) {
  const tf = window.tf;
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
