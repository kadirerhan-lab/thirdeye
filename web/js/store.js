// Yerel vaka deposu (IndexedDB). Veriler yalnızca bu tarayıcıda tutulur;
// dışa / içe aktarma ile merkezler arası veri birleştirme yapılabilir.
// IndexedDB kullanılamazsa (gizli pencere vb.) bellek içi depoya düşer.

const DB = 'thirdeye';
const VER = 1;
let dbp = null;
const memory = { cases: new Map(), kv: new Map(), next: 1 };
let useMemory = false;

function openDB() {
  if (dbp) return dbp;
  dbp = new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB, VER);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('cases')) db.createObjectStore('cases', { keyPath: 'id', autoIncrement: true });
        if (!db.objectStoreNames.contains('kv')) db.createObjectStore('kv');
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => { useMemory = true; resolve(null); };
    } catch { useMemory = true; resolve(null); }
  });
  return dbp;
}

async function tx(store, mode, fn) {
  const db = await openDB();
  if (!db || useMemory) return fn(null);
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const s = t.objectStore(store);
    let result;
    Promise.resolve(fn(s)).then((r) => { result = r; });
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
  });
}

const req2p = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

export async function addCases(list) {
  return tx('cases', 'readwrite', async (s) => {
    if (!s) { for (const c of list) { const id = memory.next++; memory.cases.set(id, { ...c, id }); } return list.length; }
    for (const c of list) { const { id, ...rest } = c; s.add(rest); }
    return list.length;
  });
}

export async function allCases() {
  return tx('cases', 'readonly', async (s) => (s ? req2p(s.getAll()) : [...memory.cases.values()]));
}

export async function deleteWhere(pred) {
  const all = await allCases();
  const ids = all.filter(pred).map((c) => c.id);
  await tx('cases', 'readwrite', async (s) => {
    for (const id of ids) { if (s) s.delete(id); else memory.cases.delete(id); }
  });
  return ids.length;
}

export async function getKV(key) {
  return tx('kv', 'readonly', async (s) => (s ? req2p(s.get(key)) : memory.kv.get(key)));
}

export async function setKV(key, value) {
  return tx('kv', 'readwrite', async (s) => { if (s) s.put(value, key); else memory.kv.set(key, value); });
}

const toB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
const fromB64 = (b) => { const s = atob(b); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; };

export async function exportDataset() {
  const cases = await allCases();
  return {
    format: 'thirdeye-dataset', version: 1, exportedAt: new Date().toISOString(),
    cases: cases.map(({ id, patch, ...c }) => ({ ...c, patch: toB64(patch) })),
  };
}

export async function importDataset(json) {
  if (json?.format !== 'thirdeye-dataset' || !Array.isArray(json.cases)) throw new Error('Bu dosya bir ThirdEye veri seti değil.');
  const list = json.cases.map((c) => ({ ...c, patch: fromB64(c.patch), importedAt: new Date().toISOString() }));
  await addCases(list);
  return list.length;
}

export const storageMode = () => (useMemory ? 'memory' : 'indexeddb');
