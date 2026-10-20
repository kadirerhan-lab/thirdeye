// Bağımlılıksız yerel sunucu: web/ klasörünü http://localhost:5173 adresinde yayınlar.
// Kullanım: node server.js [--port 5173] [--no-open]
import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exec } from 'node:child_process';

const root = join(dirname(fileURLToPath(import.meta.url)), 'web');
const args = process.argv.slice(2);
let port = Number(args[args.indexOf('--port') + 1]) || Number(process.env.PORT) || 5173;
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml', '.webm': 'video/webm', '.mp4': 'video/mp4', '.ico': 'image/x-icon',
};

// İskeleti olmayan HTML parçaları (gerekirse) tam belgeye sarılır.
const wrap = (body) => `<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n${body}\n</body></html>`;

const server = http.createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (path === '/') path = '/index.html';
    const file = normalize(join(root, path));
    if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
    const s = await stat(file);
    if (!s.isFile()) throw new Error('not file');
    let data = await readFile(file);
    if (path.endsWith('.html') && !/^\s*<!doctype/i.test(data.toString('utf8', 0, 64))) data = Buffer.from(wrap(data.toString('utf8')));
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Bulunamadı');
  }
});

server.on('error', (e) => {
  if (e.code === 'EADDRINUSE' && port < 5190) { port++; server.listen(port); } else throw e;
});
server.on('listening', () => {
  const url = `http://localhost:${port}`;
  console.log(`ThirdEye çalışıyor: ${url}  (durdurmak için Ctrl+C)`);
  if (!args.includes('--no-open')) {
    const cmd = process.platform === 'win32' ? `start "" "${url}"` : process.platform === 'darwin' ? `open "${url}"` : `xdg-open "${url}"`;
    exec(cmd, () => {});
  }
});
server.listen(port);
