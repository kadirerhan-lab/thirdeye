import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreTirads, categoryFromPoints, recommendation, suggestTirads } from '../web/js/tirads.js';
import { fuseRisk, modelReliability } from '../web/js/risk.js';
import { boxBlur, fillHoles, labelComponents, convexHull, polygonArea } from '../web/js/imgproc.js';
import { generateCase } from '../web/js/synth.js';
import { preprocess, detectAll, noduleFromTrack } from '../web/js/pipeline.js';
import { rocAuc } from '../web/js/model.js';

test('ACR TI-RADS puan → kategori', () => {
  assert.equal(categoryFromPoints(0), 'TR1');
  assert.equal(categoryFromPoints(1), 'TR1'); // eski sürümde yanlışlıkla TR5 dönüyordu
  assert.equal(categoryFromPoints(2), 'TR2');
  assert.equal(categoryFromPoints(3), 'TR3');
  assert.equal(categoryFromPoints(6), 'TR4');
  assert.equal(categoryFromPoints(7), 'TR5');
});

test('süngerimsi nodülde diğer puanlar eklenmez', () => {
  const r = scoreTirads({ composition: 'spongiform', echogenicity: 'hypo', shape: 'taller', margin: 'smooth', foci: ['punctate'] });
  assert.equal(r.points, 0);
  assert.equal(r.category, 'TR1');
});

test('klasik şüpheli nodül TR5 ve İİAB eşiği', () => {
  const r = scoreTirads({ composition: 'solid', echogenicity: 'veryhypo', shape: 'taller', margin: 'lobulated', foci: ['punctate'] });
  assert.equal(r.points, 13);
  assert.equal(r.category, 'TR5');
  assert.equal(recommendation('TR5', 12).action, 'fna');
  assert.equal(recommendation('TR5', 7).action, 'follow');
  assert.equal(recommendation('TR3', 20).action, 'follow');
  assert.equal(recommendation('TR3', 26).action, 'fna');
  assert.equal(recommendation('TR2', 40).action, 'none');
});

test('model kanıtı TI-RADS riskini sınırlı ölçüde değiştirir', () => {
  assert.equal(fuseRisk('TR4', null, null).final.toFixed(3), '0.091');
  const meta = { metrics: { auc: 0.95 }, nTrain: 300, nClinical: 0, trainPrevalence: 0.5 };
  assert.ok(modelReliability(meta) <= 0.35 + 1e-9);
  const up = fuseRisk('TR4', 0.999, meta).final;
  assert.ok(up > 0.091 && up < 0.3, `beklenen sınırlı artış, gelen ${up}`);
});

test('görüntü işleme yardımcıları', () => {
  const w = 20, h = 20;
  const flat = new Float32Array(w * h).fill(50);
  assert.ok(boxBlur(flat, w, h, 3).every((v) => Math.abs(v - 50) < 1e-4));
  const ring = new Uint8Array(w * h);
  for (let y = 4; y < 16; y++) for (let x = 4; x < 16; x++) if (y === 4 || y === 15 || x === 4 || x === 15) ring[y * w + x] = 1;
  const filled = fillHoles(ring, w, h);
  assert.equal(filled.reduce((a, b) => a + b, 0), 144);
  assert.equal(labelComponents(ring, w, h).comps.length, 1);
  assert.equal(polygonArea(convexHull([{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 4 }, { x: 0, y: 4 }, { x: 2, y: 2 }])), 16);
});

test('ROC AUC', () => {
  assert.equal(rocAuc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]).auc, 1);
  assert.equal(rocAuc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]).auc, 0);
});

test('uçtan uca: sentetik taramalar doğru kategoriye düşer', async () => {
  for (const [preset, expect] of [['malignant', 'TR5'], ['benign', 'TR3']]) {
    const c = generateCase({ seed: preset === 'malignant' ? 7 : 11, preset, frames: 24 });
    const p = preprocess(c.frames);
    assert.ok(p.overlayRatio > 0, 'kaliper işaretleri bulunmalı');
    const d = await detectAll(p.grays, p.w, p.h);
    const t = d.tracks[0];
    const gx = Math.round(t.best.cx + p.sector.x), gy = Math.round(t.best.cy + p.sector.y);
    assert.equal(c.truth.mask[gy * c.w + gx], 1, 'ilk aday gerçek nodülün içinde olmalı');
    const n = noduleFromTrack(t, p.grays, p.w, p.h, c.mmPerPx, d.preCache);
    assert.equal(scoreTirads(suggestTirads(n.features).sel).category, expect);
  }
});

test('model modülü tarayıcı dışında güvenle yüklenir ve grup ayrımı sızıntısızdır', async () => {
  const { hasTF, groupFolds } = await import('../web/js/model.js');
  assert.equal(hasTF(), false);
  const cases = [];
  for (let g = 0; g < 20; g++) for (let f = 0; f < 3; f++) cases.push({ group: `n${g}`, label: g % 4 === 0 ? 1 : 0 });
  const folds = groupFolds(cases, 5, 1);
  const seen = new Map();
  folds.forEach((idx, k) => idx.forEach((i) => {
    const g = cases[i].group;
    assert.ok(!seen.has(g) || seen.get(g) === k, 'aynı nodül iki farklı katta olmamalı');
    seen.set(g, k);
  }));
  assert.equal(folds.flat().length, cases.length);
});
