"""ThirdEye modelini PyTorch ile eğitir (uygulamadaki TensorFlow.js mimarisinin birebir aynısı).

Kullanım: python tools/stanford/train_torch.py --data data/stanford [--epochs 30] [--folds 5]
Girdi:  <data>/prepared/cases.json, patches.bin   (prepare.mjs çıktısı)
Çıktı:  <data>/prepared/torch_result.json  (katlar dışı tahminler, kat AUC'leri, son model ağırlıkları)

Doğrulama nodül bazındadır: aynı nodülün kareleri aynı kata düşer; metrikler nodül düzeyinde hesaplanır.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 64


class Net(nn.Module):
    # TF.js karşılığı: conv(8)-pool, conv(16)-pool, conv(32)-pool, GAP | dense(16) ; concat → dropout → dense(16) → dense(1)
    def __init__(self, nfeat):
        super().__init__()
        self.c1 = nn.Conv2d(2, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.c3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fd = nn.Linear(nfeat, 16)
        self.d1 = nn.Linear(48, 16)
        self.out = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, f):
        for c in (self.c1, self.c2, self.c3):
            x = F.max_pool2d(F.relu(c(x)), 2)
        x = x.mean(dim=(2, 3))
        z = torch.cat([x, F.relu(self.fd(f))], 1)
        z = F.relu(self.d1(self.drop(z)))
        return self.out(z).squeeze(1)


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def group_folds(groups, labels, k, seed):
    rnd = random.Random(seed)
    g2l = {}
    for g, l in zip(groups, labels):
        g2l[g] = l
    folds = [[] for _ in range(k)]
    for lab in (0, 1):
        gs = [g for g, l in g2l.items() if l == lab]
        rnd.shuffle(gs)
        for n, g in enumerate(gs):
            folds[n % k].append(g)
    return [set(f) for f in folds]


def nodule_level(groups, labels, preds, idx):
    by = {}
    for i in idx:
        by.setdefault(groups[i], [labels[i], []])[1].append(preds[i])
    ys = [v[0] for v in by.values()]
    ps = [float(np.mean(v[1])) for v in by.values()]
    return ps, ys


def fit(X, Fv, y, idx, epochs, seed, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mean = Fv[idx].mean(0); std = Fv[idx].std(0); std[std == 0] = 1
    Fn = ((Fv - mean) / std).astype(np.float32)
    net = Net(Fv.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = [i for i in idx if y[i] == 1]; neg = [i for i in idx if y[i] == 0]
    big = max(len(pos), len(neg))
    Xt = torch.from_numpy(X); Ft = torch.from_numpy(Fn); yt = torch.from_numpy(y.astype(np.float32))
    for ep in range(epochs):
        # Sınıf dengesi: azınlık sınıfı yeniden örneklenir; her örnek rastgele yatay aynalanır.
        batch_idx = np.array([pos[i % len(pos)] for i in range(big)] + [neg[i % len(neg)] for i in range(big)])
        np.random.shuffle(batch_idx)
        net.train(); tot = 0.0
        for b in range(0, len(batch_idx), 32):
            bi = torch.from_numpy(batch_idx[b:b + 32])
            xb = Xt[bi]
            flip = torch.rand(len(bi)) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            loss = F.binary_cross_entropy_with_logits(net(xb, Ft[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(bi)
        if log is not None:
            log.append({"epoch": ep + 1, "loss": tot / len(batch_idx)})
    net.eval()
    return net, mean, std


def predict(net, X, Fv, mean, std, idx):
    with torch.no_grad():
        Fn = torch.from_numpy(((Fv[idx] - mean) / std).astype(np.float32))
        return torch.sigmoid(net(torch.from_numpy(X[idx]), Fn)).numpy()


def export(net):
    # TF.js düzenine çevir: conv [out,in,kh,kw] → [kh,kw,in,out]; dense [out,in] → [in,out]
    w = {}
    for name in ("c1", "c2", "c3"):
        m = getattr(net, name)
        w[name] = {"kernel": m.weight.detach().permute(2, 3, 1, 0).numpy().ravel().tolist(), "kshape": list(m.weight.permute(2, 3, 1, 0).shape),
                   "bias": m.bias.detach().numpy().tolist()}
    for name in ("fd", "d1", "out"):
        m = getattr(net, name)
        w[name] = {"kernel": m.weight.detach().t().numpy().ravel().tolist(), "kshape": list(m.weight.t().shape), "bias": m.bias.detach().numpy().tolist()}
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stanford")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds-only", action="store_true", help="yalnızca çapraz doğrulama katlarını çalıştır")
    a = ap.parse_args()
    d = os.path.join(a.data, "prepared")
    meta = json.load(open(os.path.join(d, "cases.json"), encoding="utf-8"))
    rows = meta["rows"]
    N = len(rows)
    X = np.fromfile(os.path.join(d, "patches.bin"), np.uint8).reshape(N, P, P, 2).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    Fv = np.array([r["fv"] for r in rows], np.float32)
    y = np.array([r["label"] for r in rows], np.int64)
    groups = [r["group"] for r in rows]
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    folds = group_folds(groups, y.tolist(), a.folds, a.seed)
    oof = np.zeros(N, np.float32)
    fold_auc = []
    # Her kat ayrı kaydedilir; yarıda kalan eğitim biten katları atlayarak sürdürülür.
    ckdir = os.path.join(d, "folds")
    os.makedirs(ckdir, exist_ok=True)
    sig = f"e{a.epochs}_f{a.folds}_s{a.seed}_n{N}"
    for k, fg in enumerate(folds):
        va = [i for i in range(N) if groups[i] in fg]
        ck = os.path.join(ckdir, f"{sig}_k{k}.json")
        if os.path.exists(ck):
            r = json.load(open(ck))
            oof[va] = np.array(r["pred"], np.float32)
            fold_auc.append(r["auc"])
            print(f"kat {k + 1}/{a.folds}: kayıttan (AUC {r['auc']})", flush=True)
            continue
        tr = [i for i in range(N) if groups[i] not in fg]
        net, mean, std = fit(X, Fv, y, tr, a.epochs, a.seed + k)
        oof[va] = predict(net, X, Fv, mean, std, va)
        ps, ys = nodule_level(groups, y.tolist(), oof.tolist(), va)
        fold_auc.append(auc(ps, ys))
        json.dump({"pred": oof[va].tolist(), "auc": fold_auc[-1]}, open(ck, "w"))
        print(f"kat {k + 1}/{a.folds}: AUC {fold_auc[-1]} ({len(ys)} nodül)", flush=True)

    if a.folds_only:
        print("Katlar tamamlandı; son model için --folds-only olmadan tekrar çalıştırın.")
        return
    hist = []
    net, mean, std = fit(X, Fv, y, list(range(N)), a.epochs, a.seed + 100, hist)
    par_idx = list(range(min(16, N)))
    parity = predict(net, X, Fv, mean, std, par_idx).tolist()
    json.dump({"oof": oof.tolist(), "foldAUC": fold_auc, "epochs": a.epochs, "folds": a.folds,
               "final": {"weights": export(net), "mean": mean.tolist(), "std": std.tolist(), "history": hist},
               "parity": {"idx": par_idx, "p": parity}},
              open(os.path.join(d, "torch_result.json"), "w"))
    ps, ys = nodule_level(groups, y.tolist(), oof.tolist(), range(N))
    print(f"Çapraz doğrulama (nodül düzeyi) AUC: {auc(ps, ys):.3f}")


if __name__ == "__main__":
    main()
