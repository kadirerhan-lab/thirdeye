// ThirdEye kullanım videosunu otomatik kaydeder.
// Kullanım:  npm start  (ayrı pencerede)  →  node tools/record-demo.mjs [url] [çıktı]
// Uygulama görünmez Chrome'da adım adım kullanılır; ekran akışı (CDP screencast) toplanır,
// uzun süren adımlar hızlandırılır ve kareler ffmpeg ile H.264 MP4'e kodlanır.
import puppeteer from 'puppeteer-core';
import { writeFileSync, mkdirSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import ffmpeg from 'ffmpeg-static';

const URL0 = process.argv[2] || 'http://localhost:5173/';
const OUT = process.argv[3] || 'web/media/thirdeye-kullanim-demosu.mp4';
const CHROME = process.env.CHROME_PATH || [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
].find((p) => existsSync(p));
const W = 1366, H = 820, FPS = 15;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({ protocolTimeout: 900000,
  executablePath: CHROME, headless: true, defaultViewport: { width: W, height: H },
  args: [`--window-size=${W},${H}`, '--hide-scrollbars', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage();
const frames = [];
const segments = []; // { t0, t1, target } hızlandırılmış aralıklar (saniye)
const cdp = await page.createCDPSession();
cdp.on('Page.screencastFrame', async (f) => {
  frames.push({ data: f.data, t: f.metadata.timestamp });
  try { await cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }); } catch { /* kapanıyor */ }
});
const now = () => Date.now() / 1000;

// Sayfaya altyazı ve tıklama göstergesi katmanı ekler.
async function overlay() {
  await page.evaluate(() => {
    if (document.getElementById('demoCap')) return;
    const st = document.createElement('style');
    st.textContent = `#demoCap{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);max-width:min(92vw,980px);background:rgba(10,15,20,.92);color:#f3f6f8;font:500 19px/1.4 "IBM Plex Sans",system-ui,sans-serif;padding:12px 20px;border-radius:10px;z-index:9999;box-shadow:0 8px 30px rgba(0,0,0,.35);border-left:4px solid #f0ae2e;transition:opacity .25s}
#demoCap b{color:#f0ae2e;font-weight:600;margin-right:8px}
#demoTag{position:fixed;right:16px;top:12px;background:rgba(10,15,20,.85);color:#f0ae2e;font:500 12px/1 "IBM Plex Mono",monospace;padding:6px 10px;border-radius:6px;z-index:9999}
.demoDot{position:fixed;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;border:3px solid #f0ae2e;z-index:9998;pointer-events:none;animation:demoP .7s ease-out forwards}
@keyframes demoP{from{transform:scale(.4);opacity:1}to{transform:scale(1.6);opacity:0}}`;
    document.head.append(st);
    const c = document.createElement('div'); c.id = 'demoCap'; c.style.opacity = '0'; document.body.append(c);
    const t = document.createElement('div'); t.id = 'demoTag'; t.textContent = 'ThirdEye · kullanım demosu'; document.body.append(t);
  });
}
async function caption(step, text) {
  await page.evaluate((s, x) => {
    const c = document.getElementById('demoCap');
    c.innerHTML = (s ? `<b>${s}</b>` : '') + x; c.style.opacity = '1';
  }, step, text);
}
async function click(sel) {
  const el = await page.waitForSelector(sel, { visible: true, timeout: 20000 });
  await el.scrollIntoView();
  const box = await el.boundingBox();
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.mouse.move(x, y, { steps: 12 });
  await page.evaluate((x2, y2) => { const d = document.createElement('div'); d.className = 'demoDot'; d.style.left = x2 + 'px'; d.style.top = y2 + 'px'; document.body.append(d); setTimeout(() => d.remove(), 800); }, x, y);
  await el.click();
}
const waitEnabled = (sel, timeout = 600000) => page.waitForFunction((s) => !document.querySelector(s).disabled, { timeout, polling: 500 }, sel);
const scrollTo = (sel, block = 'start') => page.evaluate((s, b) => document.querySelector(s)?.scrollIntoView({ behavior: 'smooth', block: b }), sel, block);
async function fast(target, fn) { const t0 = now(); await fn(); segments.push({ t0, t1: now(), target }); }

// ---------- senaryo ----------
await page.goto(URL0 + '?nosample', { waitUntil: 'networkidle0' });
await overlay();
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 82, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
await caption('', 'ThirdEye — tiroid ultrason videosundan nodül analizi, ACR TI-RADS önerisi ve öğrenen risk skoru');
await sleep(4000);

await caption('1 · Kayıt', 'Ultrason videosu, görüntü dizisi veya örnek tarama yüklenir. Derinlik bilgisi mm ölçeğini belirler.');
await page.hover('#drop'); await sleep(3000);
await click('[data-sample="malignant"]');
await page.waitForSelector('.cand', { timeout: 60000 });
await sleep(1500);

await caption('2 · Ön işleme', 'Görüntü alanı kırpılır; kaliper “+ / x” ve ölçüm çizgileri bulunup temizlenir (kırmızı: temizlenen işaretler).');
await click('#viewOrig'); await sleep(3500);
await click('#viewClean'); await sleep(1500);

await caption('3 · Tespit ve takip', 'Nodül adayları tüm karelerde izlenir; damar ve akustik gölge gibi sabit yapılar geri planda kalır.');
await click('#playBtn'); await sleep(5000); await click('#playBtn'); await sleep(500);
await click('.cand'); await sleep(2500);

await caption('4 · Özellikler', 'En büyük kesitte boyut, AP/transvers oranı, kenar, ekojenite, kistik oran ve kalsifikasyonlar ölçülür.');
await scrollTo('#featureTables', 'center'); await sleep(4000);
await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' })); await sleep(800);

await caption('5 · ACR TI-RADS', 'AI tanımlayıcıları önerir (AI etiketi); hekim onaylar veya düzeltir.');
await click('[data-step="tirads"]'); await sleep(4000);
await caption('5 · ACR TI-RADS', 'Hekim düzeltmesi puanı, kategoriyi ve İİAB önerisini anında günceller.');
await click('label[for="tr-margin-smooth"]'); await sleep(3500);
await click('label[for="tr-margin-lobulated"]'); await sleep(2000);
await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));

await caption('6 · Risk skoru', 'Model henüz eğitilmediği için skor yalnızca TI-RADS kategorisinin literatür riskine dayanır.');
await click('[data-step="score"]'); await sleep(4500);

await caption('7 · Öğrenme döngüsü', 'Sitoloji veya patoloji ile kesinleşen tanı girilir ve vaka eğitim veri setine eklenir.');
await click('[data-step="learn"]'); await sleep(1500);
await page.select('#labelSelect', '1|Bethesda VI (malign sitoloji)');
await page.evaluate(() => document.getElementById('labelSelect').dispatchEvent(new Event('change')));
await sleep(1500);
await click('#addCaseBtn'); await sleep(2500);

await caption('7 · Öğrenme döngüsü', 'Demo için 200 sentetik eğitim vakası üretiliyor (video hızlandırıldı).');
await fast(7, async () => { await click('#synthBtn'); await sleep(500); await waitEnabled('#synthBtn'); });
await sleep(1000);

await caption('8 · Model eğitimi', 'Derin öğrenme modeli tarayıcıda eğitiliyor; kayıp eğrisi canlı izlenir (video hızlandırıldı).');
await page.select('#epochSelect', '15');
await scrollTo('#trainBtn', 'center');
await fast(10, async () => { await click('#trainBtn'); await sleep(500); await waitEnabled('#trainBtn'); });
await caption('8 · Model eğitimi', 'Doğrulama sonuçları: AUC, duyarlılık, özgüllük, ROC eğrisi ve model sürüm geçmişi.');
await scrollTo('#metricTiles', 'center'); await sleep(5000);

await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
await caption('9 · Güncel skor', 'Skor artık TI-RADS riskini, modelin doğrulanmış güvenilirliği oranında görüntü kanıtıyla günceller.');
await click('[data-step="score"]'); await sleep(5000);
await caption('9 · Açıklanabilirlik', 'Modelin kararını etkileyen özellikler ve görüntüdeki AI dikkat haritası gösterilir.');
await click('#optSaliency'); await sleep(4500);

await caption('10 · Karşılaştırma', 'İyi huylu görünümlü örnek tarama: aynı akış, düşük risk.');
await click('[data-step="source"]'); await sleep(800);
await click('[data-sample="benign"]');
await page.waitForFunction(() => document.querySelector('[data-step="nodule"]').classList.contains('is-active'), { timeout: 60000 });
await sleep(1200);
await click('[data-step="score"]'); await sleep(5000);

await caption('', 'Öğrenen pilot: kesin tanılı, kaliteli vaka sayısı arttıkça modelin katkısı ve doğruluğu artar. Klinik kullanım için değildir.');
await sleep(5500);
await cdp.send('Page.stopScreencast');
await sleep(300);
console.log(`Toplanan kare: ${frames.length}`);

// ---------- zaman çizelgesi (hızlandırma) ----------
function mapT(t) {
  let out = t - frames[0].t;
  for (const s of segments) {
    if (t <= s.t0) continue;
    const inside = Math.min(t, s.t1) - s.t0;
    const scale = s.target / Math.max(0.001, s.t1 - s.t0);
    out -= inside * (1 - Math.min(1, scale));
  }
  return out;
}
const timeline = [];
let lastOut = -1;
for (const f of frames) {
  const tt = mapT(f.t);
  if (tt - lastOut < 1 / FPS && timeline.length) { timeline[timeline.length - 1].data = f.data; continue; }
  timeline.push({ t: tt, data: f.data }); lastOut = tt;
}
const total = timeline[timeline.length - 1].t + 2;
console.log(`Video süresi ≈ ${total.toFixed(1)} sn, ${timeline.length} kare`);

// ---------- kodlama: kareler + süreler → ffmpeg ile standart H.264 MP4 ----------
await browser.close();
const tmp = mkdtempSync(join(tmpdir(), 'thirdeye-demo-'));
const list = [];
timeline.forEach((f, k) => {
  const name = `f${String(k).padStart(5, '0')}.jpg`;
  writeFileSync(join(tmp, name), Buffer.from(f.data, 'base64'));
  const dur = (k + 1 < timeline.length ? timeline[k + 1].t : total) - f.t;
  list.push(`file '${name}'`, `duration ${Math.max(0.001, dur).toFixed(3)}`);
});
list.push(`file 'f${String(timeline.length - 1).padStart(5, '0')}.jpg'`);
writeFileSync(join(tmp, 'list.txt'), list.join(String.fromCharCode(10)));
mkdirSync(dirname(OUT), { recursive: true });
const r = spawnSync(ffmpeg, ['-y', '-f', 'concat', '-safe', '0', '-i', join(tmp, 'list.txt'),
  '-vf', 'fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '24',
  '-tune', 'stillimage', '-movflags', '+faststart', OUT], { stdio: 'inherit' });
rmSync(tmp, { recursive: true, force: true });
if (r.status !== 0) throw new Error('ffmpeg başarısız oldu');
console.log(`Kaydedildi: ${OUT}`);
