// Sentetik örnek ultrason videoları üretir (paylaşımı serbest; gerçek hasta verisi değildir).
// Kullanım: node tools/make-sample-videos.mjs  →  web/media/ornek-*.mp4
import { spawnSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import ffmpeg from 'ffmpeg-static';
import { generateCase } from '../web/js/synth.js';

const W = 640, H = 480, FRAMES = 48, FPS = 12;
const items = [
  { name: 'ornek-supheli-nodul', preset: 'malignant', seed: 7 },
  { name: 'ornek-iyi-huylu-nodul', preset: 'benign', seed: 11 },
];
mkdirSync('web/media', { recursive: true });
for (const it of items) {
  const c = generateCase({ seed: it.seed, preset: it.preset, w: W, h: H, frames: FRAMES, text: false });
  // İleri-geri tarama: prob nodülün üzerinden iki kez geçer.
  const order = [...c.frames.keys(), ...[...c.frames.keys()].reverse()];
  const raw = Buffer.alloc(order.length * W * H * 3);
  order.forEach((f, k) => {
    const d = c.frames[f].data;
    for (let i = 0; i < W * H; i++) { const o = (k * W * H + i) * 3; raw[o] = d[i * 4]; raw[o + 1] = d[i * 4 + 1]; raw[o + 2] = d[i * 4 + 2]; }
  });
  const out = `web/media/${it.name}.mp4`;
  const r = spawnSync(ffmpeg, ['-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${H}`, '-r', String(FPS), '-i', 'pipe:0',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', out], { input: raw, maxBuffer: 1 << 30 });
  if (r.status !== 0) throw new Error(r.stderr.toString());
  console.log(`${out}: ${order.length} kare, ${(order.length / FPS).toFixed(1)} sn, derinlik ≈ ${((c.mmPerPx * c.sector.h) / 10).toFixed(1)} cm`);
}
