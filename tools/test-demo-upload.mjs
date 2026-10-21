// Sentetik örnek videoları demoya gerçek Chrome'da yükleyip sonucu yazdırır.
// Kullanım: node tools/test-demo-upload.mjs <ekran-görüntüsü-klasörü> [demo-adresi]
import puppeteer from 'puppeteer-core';
import { resolve } from 'node:path';
const out = process.argv[2] || '.';
const base = process.argv[3] || 'https://kadirerhan-lab.github.io/thirdeye/';
const b = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, defaultViewport: { width: 1400, height: 950 } });
for (const name of ['ornek-supheli-nodul', 'ornek-iyi-huylu-nodul']) {
  const p = await b.newPage();
  p.on('pageerror', (e) => console.log('pageerror', e.message));
  await p.goto(base + '?nosample&v=' + Date.now(), { waitUntil: 'networkidle0' });
  await p.evaluate(() => { const d = document.getElementById('depthInput'); d.value = '3.0'; d.dispatchEvent(new Event('change')); });
  await (await p.$('#fileInput')).uploadFile(resolve(`web/media/${name}.mp4`));
  await p.waitForFunction(() => document.getElementById('busy').hidden && document.querySelector('[data-step="nodule"]').classList.contains('is-active') && document.querySelector('.tr-badge'), { timeout: 120000, polling: 500 });
  const info = await p.evaluate(() => ({
    aday: [...document.querySelectorAll('.cand')].map((c) => c.innerText.replace(/\s+/g, ' ')).slice(0, 2),
    tirads: [...document.querySelectorAll('#tiradsResult .tr-line span')].map((s) => s.innerText).join(' · '),
    oneri: document.querySelector('#tiradsResult .rec')?.innerText,
  }));
  console.log(name, JSON.stringify(info, null, 1));
  await p.screenshot({ path: `${out}/demo_${name}.png` });
  await p.close();
}
await b.close();
