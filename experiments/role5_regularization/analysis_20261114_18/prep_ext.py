import numpy as np, csv, hashlib, os, glob
os.chdir("/data/eunjin/DL")
P = "common_data/v2_3458/fashion-common-expanded-partial-20261106-v2"
rows = list(csv.DictReader(open(f"{P}/training_manifest.csv")))
z = np.load(f"{P}/training_targets.npz", allow_pickle=False)
ids = [str(i) for i in z["image_ids"]]; cats = [str(c) for c in z["categories"]]
ref = [str(c) for c in np.load("runs_hr/R4_bce_kd_s42/tta/preds_holdout_s800_flip.npz")["categories"]]
assert cats == ref
row = {r["image_id"]: r for r in rows}
ext = [i for i in ids if row[i]["source"] == "external"]
pos = {i: k for k, i in enumerate(ids)}
t = z["targets"][[pos[i] for i in ext]].astype(np.float32)
paths = [os.path.abspath(os.path.join(P, "selected_external/images", row[i]["relative_path"])) for i in ext]
assert all(os.path.exists(p) for p in paths)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
bad = [p for p, i in zip(paths, ext) if sha(p) != row[i]["image_sha256"]]
print("ext", len(ext), "sha mismatch", len(bad), "label share 1/0/-1", [round(float((t == v).mean()), 4) for v in (1, 0, -1)])
print("pos per class", dict(zip([c[:10] for c in cats], (t == 1).sum(0).astype(int).tolist())))
ho = [f"fassion_tagging/data/train/images/{l.strip()}" for l in open("fassion_tagging/data/splits/holdout.txt") if l.strip()]
va = glob.glob("fassion_tagging/data/val/images/*")
te = [f for f in glob.glob("test_data/**/*", recursive=True) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
es = {row[i]["image_sha256"] for i in ext}
for n, f in (("holdout", ho), ("val", va), ("test (hash only, leak check)", te)):
    print(n, len(f), "overlap", len(es & {sha(x) for x in f}))
open("neg_data/ext3458_train.txt", "w").write("\n".join(paths) + "\n")
np.savez("neg_data/ext3458_labels.npz", filenames=np.array(paths), targets=t, categories=np.array(cats))
print("written")
