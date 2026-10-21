"""Stanford cine-clip'lerinden ThirdEye'da test edilecek MP4 videolar üretir (YALNIZCA YEREL KULLANIM).

Kullanım: python tools/stanford/make_clips.py --hdf5 data/raw/dataset.hdf5 --metadata data/raw/metadata.csv --ids 70_ 190_ 28_
Çıktı: data/ornek-videolar/*.mp4 ve BENI_OKU.txt (gerçek tanı, radyolog TI-RADS, uygulamaya girilecek derinlik)

Stanford Research Use Agreement gereği bu videolar paylaşılamaz, yayınlanamaz ve klinik amaçla kullanılamaz.
"""
import argparse
import os
import subprocess

import h5py
import numpy as np
import pandas as pd

FFMPEG = os.path.join("node_modules", "ffmpeg-static", "ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="data/raw/dataset.hdf5")
    ap.add_argument("--metadata", default="data/raw/metadata.csv")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", default="data/ornek-videolar")
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(a.metadata).set_index("annot_id")
    f = h5py.File(a.hdf5, "r")
    ids = np.array([x.decode().strip() for x in f["annot_id"][:]])
    fnum = np.array([int(x.decode().strip() or 0) for x in f["frame_num"][:]])
    notes = ["ThirdEye test videoları — Stanford AIMI Thyroid Ultrasound Cine-clip",
             "UYARI: Yalnızca kişisel, ticari olmayan araştırma içindir. Paylaşmayın, yayınlamayın, klinik kararda kullanmayın.", ""]
    for aid in a.ids:
        idx = np.where(ids == aid)[0]
        idx = idx[np.argsort(fnum[idx])]
        row = meta.loc[aid]
        label = "kotu-huylu" if int(row["histopath_diagnosis"]) == 1 else "iyi-huylu"
        name = f"stanford-{aid.strip('_')}-{label}-TR{int(row['ti_rads_level'])}.mp4"
        # Ölçek: en büyük maske uzunluğu ↔ kayıtlı en büyük boyut (cm); görüntü alanı yüksekliği → derinlik
        best, best_a = idx[0], -1
        for i in idx[:: max(1, len(idx) // 30)]:
            s = int((f["mask"][i] > 0).sum())
            if s > best_a:
                best_a, best = s, i
        m = f["mask"][best] > 0
        ys, xs = np.where(m)
        ext_px = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        size_mm = max(row["size_x"], row["size_y"], row["size_z"]) * 10
        mm_per_px = size_mm / ext_px
        img0 = f["image"][best].astype(np.float32)
        rows = np.where((img0 > 12).mean(1) > 0.25)[0]
        depth_cm = (rows.max() - rows.min() + 1) * mm_per_px / 10 if len(rows) else 4.0
        h, w = img0.shape
        out = os.path.join(a.out, name)
        p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(a.fps),
                              "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        for s in range(0, len(idx), 16):
            for frame in f["image"][idx[s:s + 16]]:
                p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        p.stdin.close()
        p.wait()
        notes.append(f"{name}\n  Gerçek tanı (histopatoloji): {'KÖTÜ HUYLU' if label == 'kotu-huylu' else 'iyi huylu'}"
                     f"\n  Radyolog TI-RADS: TR{int(row['ti_rads_level'])} · boyut {row['size_x']}×{row['size_y']}×{row['size_z']} cm · {row['location']}"
                     f"\n  {len(idx)} kare, {len(idx) / a.fps:.1f} sn\n  ThirdEye'da 'Görüntü derinliği' alanına yaklaşık {depth_cm:.1f} cm girin.\n")
        print(f"{out}: {len(idx)} kare, derinlik ≈ {depth_cm:.1f} cm, {label}", flush=True)
    with open(os.path.join(a.out, "BENI_OKU.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(notes))


if __name__ == "__main__":
    main()
