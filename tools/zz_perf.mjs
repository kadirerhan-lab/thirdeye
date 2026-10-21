import puppeteer from 'puppeteer-core';
import { resolve } from 'node:path';
const b = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, defaultViewport: { width: 1400, height: 950 } });
for (const [file, depth] of [['web/media/ornek-iyi-huylu-nodul.mp4', '3.0'], ['data/ornek-videolar/stanford-70-kotu-huylu-TR5.mp4', '4.8']]) {
  const p = await b.newPage();
  await p.evaluateOnNewDocument(() => {
    window.__long = [];
    new PerformanceObserver((l) => l.getEntries().forEach((e) => window.__long.push(Math.round(e.duration)))).observe({ type: 'longtask', buffered: true });
  });
  await p.goto('https://kadirerhan-lab.github.io/thirdeye/?nosample&v=' + Date.now(), { waitUntil: 'networkidle0' });
  await p.evaluate((d) => { const el = document.getElementById('depthInput'); el.value = d; el.dispatchEvent(new Event('change')); window.__long.length = 0; }, depth);
  const t0 = Date.now();
  await (await p.$('#fileInput')).uploadFile(resolve(file));
  await p.waitForFunction(() => document.getElementById('busy').hidden && document.querySelector('.tr-badge'), { timeout: 180000, polling: 200 });
  const tLoad = Date.now() - t0;
  const longLoad = await p.evaluate(() => { const l = window.__long.slice(); window.__long.length = 0; return l; });
  const sw = [];
  for (const step of ['tirads', 'score', 'learn', 'nodule', 'source', 'tirads']) {
    const t = await p.evaluate(async (s) => { const a = performance.now(); document.querySelector(`[data-step="${s}"]`).click(); await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); return Math.round(performance.now() - a); }, step);
    sw.push(`${step}:${t}ms`);
  }
  const longSw = await p.evaluate(() => window.__long.slice());
  const m = await p.metrics();
  console.log(file.split('/').pop(), `| yükleme ${(tLoad / 1000).toFixed(1)} sn | en uzun kilitlenmeler ${longLoad.sort((a, b) => b - a).slice(0, 6).join(', ')} ms (toplam ${longLoad.length}) | geçişler ${sw.join(' ')} | geçişte uzun görev ${longSw.join(',') || 'yok'} | JS bellek ${(m.JSHeapUsedSize / 1e6).toFixed(0)} MB`);
  await p.close();
}
await b.close();
