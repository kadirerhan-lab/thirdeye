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
    frames = f[k_frame][:] if k_frame else np.arange(len(ids))
    os.makedirs(a.out, exist_ok=True)

    meta_by = {}
    if meta is not None:
        id_col = next((c for c in meta.columns if c.lower() in ("annot_id", "annotation_id", "id")), meta.columns[0])
        for _, row in meta.iterrows():
            meta_by[str(row[id_col]).strip()] = row

    index = []
    uniq = list(dict.fromkeys(ids))
    ds = a.downscale
    for n, aid in enumerate(uniq):
        idx = np.where(ids == aid)[0]
        idx = idx[np.argsort(frames[idx])]
        # Maske alanına göre en büyük kesitler + taramaya yayılmış kareler seçilir.
        masks_small = []
        areas = []
        for i in idx:
            m = f[k_mask][i]
            areas.append(int((m > 0).sum()))
        areas = np.array(areas)
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
        print(f"[{n + 1}/{len(uniq)}] {aid}: {len(pick)} kare, etiket={rec['label']}", flush=True)

    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fp:
        json.dump({"source": os.path.basename(a.hdf5), "downscale": ds, "nodules": index}, fp, ensure_ascii=False, indent=1)
    labs = [r["label"] for r in index]
    print(f"\nTamamlandı: {len(index)} nodül · iyi huylu {labs.count(0)} · kötü huylu {labs.count(1)} · etiketsiz {labs.count(None)}")


if __name__ == "__main__":
    main()
