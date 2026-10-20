// PyTorch ile eğitilen ağırlıkları TensorFlow.js modeline aktarır, iki ortamın aynı sonucu
// verdiğini doğrular, web/models/stanford altına kaydeder ve sonuç raporunu yazar.
// Kullanım: node tools/stanford/finalize.mjs --data data/stanford [--out web/models/stanford] [--report docs/stanford-sonuclari.md]
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { setTF, buildModel, evaluate, rocAuc, FEATURES } from '../../web/js/model.js';
import { fuseRisk } from '../../web/js/risk.js';
import { categoryFromPoints, TR_INFO } from '../../web/js/tirads.js';
import { rng } from '../../web/js/imgproc.js';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const DATA = arg('data', 'data/stanford');
const OUT = arg('out', 'web/models/stanford');
const REPORT = arg('report', 'docs/stanford-sonuclari.md');
const SOURCE = arg('source', 'Stanford AIMI Thyroid Ultrasound Cine-clip v1.0');
await tf.setBackend('cpu');
setTF(tf);

const dir = join(DATA, 'prepared');
const cj = JSON.parse(readFileSync(join(dir, 'cases.json'), 'utf8'));
const tr = JSON.parse(readFileSync(join(dir, 'torch_result.json'), 'utf8'));
const rows = cj.rows;
const N = rows.length;
const patches = readFileSync(join(dir, 'patches.bin'));

// 1. Ağırlık aktarımı (katman eşleştirmesi şekle göre yapılır)
const model = buildModel(FEATURES.length);
const W = tr.final.weights;
const byShape = new Map();
for (const k of Object.keys(W)) byShape.set(W[k].kshape.join('x'), W[k]);
for (const layer of model.layers) {
  const ws = layer.getWeights();
  if (!ws.length) continue;
  const src = byShape.get(ws[0].shape.join('x'));
  if (!src) throw new Error(`${layer.name} için eşleşen ağırlık yok (${ws[0].shape})`);
  layer.setWeights([tf.tensor(src.kernel, ws[0].shape), tf.tensor(src.bias, ws[1].shape)]);
}

// 2. Eşdeğerlik kontrolü
const mean = tr.final.mean, std = tr.final.std;
const pidx = tr.parity.idx;
const xi = new Float32Array(pidx.length * 64 * 64 * 2);
pidx.forEach((i, k) => { for (let j = 0; j < 64 * 64 * 2; j++) xi[k * 64 * 64 * 2 + j] = patches[i * 64 * 64 * 2 + j] / 255; });
const xf = pidx.flatMap((i) => rows[i].fv.map((v, j) => (v - mean[j]) / std[j]));
const pj = Array.from(model.predict([tf.tensor4d(xi, [pidx.length, 64, 64, 2]), tf.tensor2d(xf, [pidx.length, FEATURES.length])]).dataSync());
const maxDiff = Math.max(...pj.map((p, k) => Math.abs(p - tr.parity.p[k])));
console.log(`PyTorch ↔ TensorFlow.js en büyük fark: ${maxDiff.toExponential(2)}`);
if (maxDiff > 1e-3) throw new Error('Aktarım doğrulanamadı: iki ortamın tahminleri farklı.');

// 3. Metrikler
const cases = rows.map((r) => ({ group: r.group, label: r.label }));
const cv = evaluate(cases, tr.oof);
const r = rng(11), boots = [];
for (let b = 0; b < 1000; b++) {
  const s = Array.from({ length: cv.valPredictions.length }, () => cv.valPredictions[Math.floor(r() * cv.valPredictions.length)]);
  const a = rocAuc(s.map((p) => p[0]), s.map((p) => p[1])).auc;
  if (a != null) boots.push(a);
}
boots.sort((a, b) => a - b);
const ci = [boots[Math.floor(0.025 * boots.length)], boots[Math.floor(0.975 * boots.length)]];
const nod = cj.nodules;
const nMal = nod.filter((n) => n.label === 1).length;
const predBy = new Map();
rows.forEach((row, i) => { if (!predBy.has(row.group)) predBy.set(row.group, []); predBy.get(row.group).push(tr.oof[i]); });
const fusionMeta = { metrics: { auc: cv.auc }, nTrain: nod.length, nClinical: nod.length, trainPrevalence: 0.5 };
const fr = nod.filter((n) => n.radPts != null).map((n) => {
  const p = predBy.get(n.id); const pm = p.reduce((a, b) => a + b, 0) / p.length;
  const cat = categoryFromPoints(n.radPts);
  return { y: n.label, fused: fuseRisk(cat, pm, fusionMeta).final, prior: TR_INFO[cat].risk };
});
const fusedAUC = rocAuc(fr.map((x) => x.fused), fr.map((x) => x.y)).auc;
const catAUC = rocAuc(fr.map((x) => x.prior), fr.map((x) => x.y)).auc;

// 4. Kaydet
mkdirSync(OUT, { recursive: true });
await model.save(tf.io.withSaveHandler(async (art) => {
  writeFileSync(join(OUT, 'weights.bin'), Buffer.from(art.weightData));
  writeFileSync(join(OUT, 'model.json'), JSON.stringify({ format: 'layers-model', generatedBy: 'ThirdEye', modelTopology: art.modelTopology, weightsManifest: [{ paths: ['weights.bin'], weights: art.weightSpecs }] }));
  return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
}));
const date = new Date().toISOString();
const meta = {
  version: 1, origin: 'pretrained', source: SOURCE, date, mean, std, featureKeys: cj.featureKeys,
  nTrain: nod.length, nVal: 0, nBenign: nod.length - nMal, nMalignant: nMal, nClinical: nod.length, trainPrevalence: 0.5,
  metrics: { ...cv, aucCI: ci, foldAUC: tr.foldAUC, validation: `${tr.folds} katlı nodül bazında çapraz doğrulama` },
  trainHistory: tr.final.history.map((h) => ({ ...h, valLoss: h.loss })),
  history: [{ version: 1, date, n: nod.length, nClinical: nod.length, auc: cv.auc, sens: cv.youden.sens, spec: cv.youden.spec }],
};
writeFileSync(join(OUT, 'meta.json'), JSON.stringify(meta));

const f3 = (v) => (v == null ? '–' : v.toFixed(3));
const pct = (v) => (v == null ? '–' : `%${Math.round(v * 100)}`);
const ref = cj.reference;
const report = `# Stanford veri seti sonuçları

Veri: ${SOURCE} · ${nod.length} biyopsi doğrulamalı nodül (${nMal} kötü huylu) · ${N} kare örneği.
Doğrulama: ${tr.folds} katlı, nodül bazında çapraz doğrulama (aynı nodülün kareleri aynı katta). Metrikler nodül düzeyindedir. Tarih: ${date.slice(0, 10)}.

| Yöntem | AUC |
|---|---|
| Radyoloğun TI-RADS puanı | ${f3(ref.radiologistAUC)} |
| Radyoloğun TI-RADS kategorisi (literatür riski) | ${f3(catAUC)} |
| ThirdEye otomatik TI-RADS önerisi (uzman maskesiyle) | ${f3(ref.autoTiradsAUC)} |
| ThirdEye görüntü modeli | ${f3(cv.auc)} (%95 GA ${f3(ci[0])}–${f3(ci[1])}) |
| Birleşik skor: radyolog TI-RADS + model | ${f3(fusedAUC)} |

Kat bazında model AUC: ${tr.foldAUC.map(f3).join(' · ')}.
Youden eşiğinde (${f3(cv.youden.t)}) model duyarlılığı ${pct(cv.youden.sens)}, özgüllüğü ${pct(cv.youden.spec)}.

Otomatik TI-RADS tanımlayıcılarının radyologla birebir uyumu: ${Object.entries(ref.descriptorAgreement).map(([k, v]) => `${k} ${pct(v)}`).join(' · ')}.
Otomatik segmentasyonun uzman maskesine uyumu (en büyük kesit, merkezden tohumlama): ortalama Dice ${f3(ref.diceMean)}; Dice ≥ 0,7 olan nodül oranı ${pct(ref.diceOver70)}.

Notlar:
- Tek merkezli, küçük bir veri setidir; güven aralıkları geniştir. Dış doğrulama olmadan klinik performans olarak yorumlanmamalıdır.
- Model PyTorch ile eğitilip TensorFlow.js'e aktarılmıştır (tahmin farkı ${maxDiff.toExponential(1)}).
- Ölçek (mm/piksel), kayıtlı en büyük nodül boyutundan kalibre edilmiştir.
- Ham görüntüler repoya eklenmemiştir; yalnızca model ağırlıkları ve bu özet paylaşılır.
`;
mkdirSync('docs', { recursive: true });
writeFileSync(REPORT, report);
console.log(report);
