"""Stanford AIMI "Thyroid Ultrasound Cine-clip" veri setini ThirdEye eğitimi için hazırlar.

Kullanım:
  python tools/stanford/extract.py --hdf5 D:/stanford/dataset.hdf5 --metadata D:/stanford/metadata.csv --inspect
  python tools/stanford/extract.py --hdf5 D:/stanford/dataset.hdf5 --metadata D:/stanford/metadata.csv --out data/stanford

Çıktı (data/ altında kalır, repoya girmez):
  index.json            nodül listesi: annot_id, etiket, TI-RADS tanımlayıcıları, boyutlar, dosya adları
  <annot_id>_img.npy    [kare, yükseklik, genişlik] uint8 görüntüler (2x küçültülmüş)
  <annot_id>_mask.npy   aynı boyutta 0/1 uzman maskeleri

Veri seti Stanford Research Agreement kapsamında yalnızca ticari olmayan araştırma içindir;
ham görüntüler bu klasör dışına kopyalanmamalı ve paylaşılmamalıdır.
"""
import argparse
import gc
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd


def find(f, *names):
    keys = list(f.keys())
    for n in names:
        for k in keys:
            if k.lower() == n:
                return k
    for n in names:
        for k in keys:
            if n in k.lower():
                return k
    return None


def decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", "replace").rstrip("\x00")
    return str(v)


def inspect(f, meta):
    print("HDF5 içeriği:")
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks}")
    f.visititems(show)
    for k, v in f.attrs.items():
        print(f"  attr {k} = {v}")
    if meta is not None:
        print(f"\nMetadata: {meta.shape[0]} satır, sütunlar: {list(meta.columns)}")
        for c in meta.columns:
            if meta[c].dtype == object or meta[c].nunique() <= 12:
                print(f"  {c}: {meta[c].value_counts(dropna=False).to_dict()}")
            else:
                print(f"  {c}: min={meta[c].min()} max={meta[c].max()} ort={meta[c].mean():.2f}")


def label_of(row):
    for c in row.index:
        cl = c.lower()
        if "histo" in cl or "patho" in cl or "diagnos" in cl or cl in ("label", "malignant"):
            v = str(row[c]).strip().lower()
            if v in ("1", "malignant", "malign", "m", "true"):
                return 1
            if v in ("0", "benign", "b", "false"):
                return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--out", default="data/stanford")
    ap.add_argument("--frames", type=int, default=12, help="nodül başına alınacak kare sayısı")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata) if a.metadata else None
    f = h5py.File(a.hdf5, "r")
    if a.inspect:
        inspect(f, meta)
        return

    k_img = find(f, "image", "images", "img")
    k_mask = find(f, "mask", "masks", "seg")
    k_id = find(f, "annot_id", "annotation_id", "id")
    k_frame = find(f, "frame_num", "frame")
    if not (k_img and k_mask and k_id):
        sys.exit(f"Beklenen alanlar bulunamadı. Mevcut: {list(f.keys())}. Önce --inspect ile yapıya bakın.")
    print(f"Alanlar: görüntü={k_img} maske={k_mask} nodül={k_id} kare={k_frame}")
    ids = np.array([decode(x).strip() for x in f[k_id][:]])
    # Kare numarası metin olarak saklanmış olabilir ("10" < "2" sıralamasını önlemek için sayıya çevrilir).
    frames = np.array([int(decode(x).strip() or 0) for x in f[k_frame][:]]) if k_frame else np.arange(len(ids))
    os.makedirs(a.out, exist_ok=True)

    meta_by = {}
    if meta is not None:
        id_col = next((c for c in meta.columns if c.lower() in ("annot_id", "annotation_id", "id")), meta.columns[0])
        for _, row in meta.iterrows():
            meta_by[str(row[id_col]).strip()] = row

    # Kaldığı yerden devam: daha önce yazılmış nodüller atlanır.
    index_path = os.path.join(a.out, "index.json")
    partial_path = os.path.join(a.out, "index.partial.json")
    index = []
    for p in (index_path, partial_path):
        if os.path.exists(p):
            index = json.load(open(p, encoding="utf-8"))["nodules"]
            break
    done = {r["annot_id"] for r in index if os.path.exists(os.path.join(a.out, r["file"] + "_mask.npy"))}
    index = [r for r in index if r["annot_id"] in done]
    uniq = list(dict.fromkeys(ids))
    safe_of = lambda x: "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in x)
    # Özet kaydı olmadan diske yazılmış nodüller (yarıda kalan eski çalıştırma) de tanınır.
    for aid in uniq:
        sf = safe_of(aid)
        if aid not in done and os.path.exists(os.path.join(a.out, sf + "_mask.npy")) and os.path.exists(os.path.join(a.out, sf + "_img.npy")):
            try:
                for suffix in ("_mask.npy", "_img.npy"):
                    np.load(os.path.join(a.out, sf + suffix), mmap_mode="r")
            except Exception:
                continue
            row = meta_by.get(aid)
            index.append({"annot_id": aid, "file": sf, "frames": [], "n_total_frames": int((ids == aid).sum()),
                          "label": label_of(row) if row is not None else None,
                          "meta": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()} if row is not None else {}})
            done.add(aid)
    if done:
        print(f"{len(done)} nodül önceden çıkarılmış, atlanıyor.", flush=True)
    ds = a.downscale
    for n, aid in enumerate(uniq):
        if aid in done:
            continue
        idx = np.where(ids == aid)[0]
        idx = idx[np.argsort(frames[idx])]
        # Maske alanına göre en büyük kesitler + taramaya yayılmış kareler seçilir.
        # Maske alanları: bitişik kare aralıkları blok halinde okunur (36 GB dosyada kare kare okumaktan çok daha hızlı).
        areas = np.zeros(len(idx), np.int64)
        order = np.argsort(idx)
        sidx = idx[order]
        start = 0
        while start < len(sidx):
            end = start
            while end + 1 < len(sidx) and sidx[end + 1] == sidx[end] + 1 and end - start < 15:
                end += 1
            block = f[k_mask][sidx[start]:sidx[end] + 1]
            areas[order[start:end + 1]] = (block.reshape(block.shape[0], -1) > 0).sum(1)
            del block
            start = end + 1
        valid = idx[areas > 0]
        if len(valid) == 0:
            continue
        va = areas[areas > 0]
        top = valid[np.argsort(-va)[: max(1, a.frames // 2)]]
        spread = valid[np.linspace(0, len(valid) - 1, num=min(len(valid), a.frames), dtype=int)]
        pick = np.array(sorted(set(top.tolist()) | set(spread.tolist())))[: a.frames * 2]
        pick = np.sort(pick)
        imgs, msks = [], []
        for i in pick:
            im = f[k_img][i]
            if im.ndim == 3:
                im = im.mean(axis=-1)
            m = f[k_mask][i]
            if m.ndim == 3:
                m = m.max(axis=-1)
            h, w = (im.shape[0] // ds) * ds, (im.shape[1] // ds) * ds
            im = im[:h, :w].reshape(h // ds, ds, w // ds, ds).mean(axis=(1, 3))
            m = m[:h, :w].reshape(h // ds, ds, w // ds, ds).max(axis=(1, 3))
            imgs.append(im.astype(np.uint8))
            msks.append((m > 0).astype(np.uint8))
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in aid)
        np.save(os.path.join(a.out, f"{safe}_img.npy"), np.stack(imgs))
        np.save(os.path.join(a.out, f"{safe}_mask.npy"), np.stack(msks))
        row = meta_by.get(aid)
        rec = {"annot_id": aid, "file": safe, "frames": [int(frames[i]) for i in pick], "n_total_frames": int(len(idx)),
               "label": label_of(row) if row is not None else None,
               "meta": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()} if row is not None else {}}
        index.append(rec)
        json.dump({"source": os.path.basename(a.hdf5), "downscale": ds, "nodules": index}, open(partial_path, "w", encoding="utf-8"), ensure_ascii=False)
        del imgs, msks
        gc.collect()
        print(f"[{n + 1}/{len(uniq)}] {aid}: {len(pick)} kare, etiket={rec['label']}", flush=True)

    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump({"source": os.path.basename(a.hdf5), "downscale": ds, "nodules": index}, fp, ensure_ascii=False, indent=1)
    labs = [r["label"] for r in index]
    print(f"\nTamamlandı: {len(index)} nodül · iyi huylu {labs.count(0)} · kötü huylu {labs.count(1)} · etiketsiz {labs.count(None)}")


if __name__ == "__main__":
    main()
