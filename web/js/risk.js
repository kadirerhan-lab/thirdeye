// Nihai skor: TI-RADS kategorisinin literatür malignite oranı ön bilgi (prior) olarak
// alınır; öğrenen görüntü modeli bunu, kendi doğrulanmış güvenilirliği oranında
// log-olasılık uzayında günceller.
//
//   logit(nihai) = logit(TI-RADS riski) + w · [logit(p_model) − logit(eğitim prevalansı)]
//
// w (0–0.7) modelin doğrulama AUC'si ve eğitim vakası sayısından hesaplanır:
// model kanıtlanmadıkça skor TI-RADS'tan uzaklaşmaz.

import { TR_INFO } from './tirads.js';

const logit = (p) => Math.log(p / (1 - p));
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const clampP = (p) => Math.min(0.995, Math.max(0.002, p));

export function modelReliability(meta) {
  if (!meta || !meta.metrics) return 0;
  const auc = meta.metrics.auc ?? 0.5;
  const n = meta.nTrain ?? 0;
  const a = Math.max(0, Math.min(1, (auc - 0.6) / 0.3));
  const b = Math.min(1, n / 150);
  // Yalnızca sentetik / az sayıda klinik vakayla eğitilmiş modelin ağırlığı yarıya indirilir.
  const clinical = (meta.nClinical ?? 0) >= 30 ? 1 : 0.5;
  return +(0.7 * a * b * clinical).toFixed(3);
}

export function fuseRisk(category, pModel, meta) {
  const prior = TR_INFO[category].risk;
  const w = pModel == null ? 0 : modelReliability(meta);
  const trainPrev = meta?.trainPrevalence ?? 0.5;
  // Model kanıtı ±3 logit ile sınırlanır (olabilirlik oranı en fazla ≈20): tek bir model TI-RADS'ı ezemez.
  const lr = pModel == null ? 0 : Math.max(-3, Math.min(3, logit(clampP(pModel)) - logit(clampP(trainPrev))));
  const final = sigmoid(logit(prior) + w * lr);
  return { prior, pModel, w, lr, final, band: riskBand(final) };
}

export function riskBand(p) {
  if (p < 0.05) return { id: 'low', label: 'Düşük şüphe' };
  if (p < 0.2) return { id: 'mid', label: 'Orta şüphe' };
  return { id: 'high', label: 'Yüksek şüphe' };
}
