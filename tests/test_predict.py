"""Tests for the submission package. Run:  python tests/test_predict.py

* submission/predict.py gives the same model outputs and the same preprocessing as scripts/train_hires.py
  (random weights, small inputs; no pretrained download, CPU only)
* make_submission -> predict.py (as a script, 2 loader workers) -> check_submission, end to end on real val images
* edge cases: unreadable image, duplicate ids, empty / missing folder, parameter limit, grayscale / RGBA input
* the zip check (top-level files, folder name with spaces, network blocked)
Weights are random, so predictions are meaningless; only the plumbing is tested.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_hires as th  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pm = load("submission_predict", "submission/predict.py")
ms = load("make_submission", "scripts/make_submission.py")
cs = load("check_submission", "scripts/check_submission.py")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NAME = "convnext_tiny.fb_in22k_ft_in1k_384"
SHORT, MAXL = 96, 160

# a few real val images (portrait and landscape)
tmp = tempfile.TemporaryDirectory()
T = tmp.name
img_dir = os.path.join(T, "val_images")
os.makedirs(os.path.join(img_dir, "images"))
VAL_ZIP = os.path.join(ROOT, "MAT4091_fashion_val.zip")
VAL_DIR = os.environ.get("FASHION_VAL_DIR")                       # folder that contains images/ (when there is no zip)
if os.path.exists(VAL_ZIP):
    with zipfile.ZipFile(VAL_ZIP) as z:
        names = [n for n in z.namelist() if n.startswith("val/images/") and n.endswith(".jpg")][:10]
        for n in names:
            open(os.path.join(img_dir, "images", os.path.basename(n)), "wb").write(z.read(n))
elif VAL_DIR and os.path.isdir(os.path.join(VAL_DIR, "images")):
    import shutil
    for f in sorted(os.listdir(os.path.join(VAL_DIR, "images")))[:10]:
        shutil.copy(os.path.join(VAL_DIR, "images", f), os.path.join(img_dir, "images", f))
else:
    sys.exit("need MAT4091_fashion_val.zip in the repo root, or FASHION_VAL_DIR=<val folder that contains images/>")
files = sorted(os.listdir(os.path.join(img_dir, "images")))
paths = [os.path.join(img_dir, "images", f) for f in files]
pp = {"short": SHORT, "max_long": MAXL, "mean": list(MEAN), "std": list(STD)}

# ---- 1. model outputs identical to train_hires ------------------------------------------------------------------
CONFIGS = [
    {"head": "linear", "pool": "gap"},
    {"head": "linear_ms", "pool": "gapgmp"},
    {"head": "linear_ms", "pool": "csra", "csra_lambda": 0.3, "csra_t": 1.0},
    {"head": "query_ms", "query_dim": 384, "query_heads": 8, "head_dropout": 0.1},
]
x = torch.randn(2, 3, 96, 160)
sizes = torch.tensor([[96, 160], [64, 128]])                      # second image only fills part of the canvas
for c in CONFIGS:
    cfg = dict(model=NAME, drop_path=0.0, **c)
    torch.manual_seed(0)
    ref = th.build_hires_model(cfg, 28, False).eval()             # 27 classes + gate
    mine = pm.build_model(cfg, 28).eval()
    mine.load_state_dict(ref.state_dict())                        # strict: same keys and shapes
    assert sum(p.numel() for p in mine.parameters()) == sum(p.numel() for p in ref.parameters())
    with torch.no_grad():
        a, b = ref(x, sizes), mine(x, sizes)
    assert a.shape == (2, 28) and torch.allclose(a, b, atol=1e-5, rtol=1e-4), (c, (a - b).abs().max())
print("models match train_hires for", [(c["head"], c.get("pool")) for c in CONFIGS])

# ---- 2. preprocessing identical to HiResDataset (eval) ---------------------------------------------------------
for mirror in (False, True):
    ds = th.HiResDataset(img_dir, files, np.zeros((len(files), 27)), MEAN, STD, False, MAXL, 0.0, mirror=mirror)
    for i, p in enumerate(paths):
        ref_x, _, _, ref_canvas = ds[(i, SHORT)]
        mine_x, mine_m, portrait = pm.load_tensors(p, pp, True)
        got = mine_m if mirror else mine_x
        assert torch.equal(ref_x, got), (files[i], mirror)
        assert tuple(ref_canvas) == pm.canvas(SHORT, portrait, MAXL)
print("preprocessing identical for", len(paths), "images, plain and mirrored")

# ---- 3. end to end: fake run -> make_submission -> predict.py -> logits equal the reference flow ----------------------
cfg = dict(model=NAME, head="linear_ms", pool="gapgmp", drop_path=0.0, csra_lambda=0.3, csra_t=1.0, gate=True,
           gate_threshold=0.5, test_short=SHORT, max_long=MAXL)
torch.manual_seed(1)
ref = th.build_hires_model(cfg, 28, False).eval()
run_dir = os.path.join(T, "run")
os.makedirs(run_dir)
torch.save({"model": ref.state_dict(), "epoch": 7, "config": cfg, "categories": list(pm.CATEGORIES)}, os.path.join(run_dir, "best.pt"))
pkg = os.path.join(T, "pkg")
info = ms.convert(run_dir, pkg, threshold=0.5, class_threshold=["sweater=0.4", "leg warmer=0.45"], note="test")
assert info["params"] == sum(p.numel() for p in ref.parameters()) <= 50_000_000
assert sorted(os.listdir(pkg)) == ["README.md", "model.pt", "predict.py", "requirements.txt"]
readme = open(os.path.join(pkg, "README.md"), encoding="utf-8").read()
assert f"{info['params']:,}" in readme and "@@" not in readme and "python predict.py --input_dir TEST_IMAGES --output predictions.jsonl" in readme

out = os.path.join(T, "predictions.jsonl")
dump = os.path.join(T, "logits.npz")
r = subprocess.run([sys.executable, os.path.join(pkg, "predict.py"), "--input_dir", os.path.join(img_dir, "images"), "--output", out,
                    "--workers", "2", "--batch_size", "3", "--dump_logits", dump], capture_output=True, text=True, encoding="utf-8")
assert r.returncode == 0, r.stderr[-800:]
assert cs.validate_predictions(out, os.path.join(img_dir, "images")) == []
with np.load(dump) as z:                                          # close the file so Windows can delete the temp dir
    got = {k: z[k] for k in z.files}
assert list(got["filenames"]) == [os.path.splitext(f)[0] for f in files]
ds = th.HiResDataset(img_dir, files, np.zeros((len(files), 27)), MEAN, STD, False, MAXL, 0.0)
dsm = th.HiResDataset(img_dir, files, np.zeros((len(files), 27)), MEAN, STD, False, MAXL, 0.0, mirror=True)
exp = []
with torch.no_grad():
    for i in range(len(files)):
        ls = []
        for d in (ds, dsm):
            xb, _, sz, _ = th.pad_collate([d[(i, SHORT)]])
            ls.append(ref(xb, sz)[0].numpy())
        exp.append((ls[0] + ls[1]) / 2)
exp = np.stack(exp)
assert np.allclose(got["logits"], exp, atol=1e-4), np.abs(got["logits"] - exp).max()
# labels: thresholds per class (0.5, sweater 0.4, leg warmer 0.45), then the gate
thr = np.full(27, 0.5); thr[pm.CATEGORIES.index("sweater")] = 0.4; thr[pm.CATEGORIES.index("leg warmer")] = 0.45
sig = lambda v: 1 / (1 + np.exp(-v))
want = (sig(exp[:, :27]) >= thr) & (sig(exp[:, 27]) >= 0.5)[:, None]
recs = [json.loads(l) for l in open(out, encoding="utf-8")]
for rec, w in zip(recs, want):
    assert rec["labels"] == [pm.CATEGORIES[j] for j in np.flatnonzero(w)], rec
raw = open(out, encoding="utf-8").read().splitlines()[0]
assert raw.startswith('{"test_id":"') and '", "' not in raw and ": " not in raw      # compact like the assignment example
print("end to end ok:", len(recs), "records; sample:", raw[:90])

# ---- 4. edge cases ------------------------------------------------------------------------------------------------
def run_main(args):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = pm.main(args)
    return code, err.getvalue()


mp = os.path.join(pkg, "model.pt")
edge = os.path.join(T, "edge")
os.makedirs(edge)
Image.fromarray(np.random.randint(0, 255, (120, 80), dtype=np.uint8), "L").save(os.path.join(edge, "gray.png"))
Image.fromarray(np.random.randint(0, 255, (80, 120, 4), dtype=np.uint8), "RGBA").save(os.path.join(edge, "rgba.png"))
Image.new("RGB", (30, 400), (200, 10, 10)).save(os.path.join(edge, "tall_thin.jpg"))      # extreme aspect ratio
open(os.path.join(edge, "broken.jpg"), "wb").write(b"this is not an image")
eout = os.path.join(T, "edge.jsonl")
code, err = run_main(["--model", mp, "--input_dir", edge, "--output", eout, "--workers", "0"])
assert code == 0 and "broken.jpg could not be read" in err, (code, err[-300:])
recs = {json.loads(l)["test_id"]: json.loads(l)["labels"] for l in open(eout, encoding="utf-8")}
assert set(recs) == {"gray", "rgba", "tall_thin", "broken"} and recs["broken"] == []     # one record each, none dropped
assert cs.validate_predictions(eout, edge) == []
code, _ = run_main(["--model", mp, "--input_dir", edge, "--output", eout, "--workers", "0", "--strict"])
assert code == 4
dup = os.path.join(T, "dup")
os.makedirs(dup)
Image.new("RGB", (64, 64)).save(os.path.join(dup, "a.jpg"))
Image.new("RGB", (64, 64)).save(os.path.join(dup, "a.png"))
assert run_main(["--model", mp, "--input_dir", dup, "--output", eout])[0] == 2
empty = os.path.join(T, "empty")
os.makedirs(empty)
assert run_main(["--model", mp, "--input_dir", empty, "--output", eout])[0] == 2
assert run_main(["--model", mp, "--input_dir", os.path.join(T, "nope"), "--output", eout])[0] == 2
assert run_main(["--model", os.path.join(T, "missing.pt"), "--input_dir", edge, "--output", eout])[0] == 3
old = pm.PARAM_LIMIT
pm.PARAM_LIMIT = 1000
assert run_main(["--model", mp, "--input_dir", edge, "--output", eout])[0] == 3          # parameter limit enforced
pm.PARAM_LIMIT = old
print("edge cases ok (grayscale, RGBA, 30x400, unreadable, duplicates, empty, missing, limit)")

# ---- 5. the zip check -------------------------------------------------------------------------------------------
zp = ms.make_zip(pkg, os.path.join(T, "team_test.zip"))
probs = cs.check_zip(zp, os.path.join(img_dir, "images"), limit=4)
assert probs == [], probs
bad_zip = os.path.join(T, "bad.zip")
with zipfile.ZipFile(bad_zip, "w") as z:
    for n in ("predict.py", "model.pt", "requirements.txt"):
        z.write(os.path.join(pkg, n), arcname="folder/" + n)                              # nested folder, README missing
bp = cs.check_zip(bad_zip, os.path.join(img_dir, "images"), limit=2)
assert any("README.md" in p for p in bp) and any("top level" in p for p in bp), bp
print("zip check ok (passes the good zip, flags the bad one)")

# ---- 6. the prediction-file validator catches format errors -----------------------------------------------------------
badf = os.path.join(T, "bad.jsonl")
lines = ['{"test_id":"a","labels":["jacket","jacket"]}', '{"test_id":"a","labels":["not a class"]}', '{"test_id":"b","labels":"jacket"}',
         '{"id":"c","labels":[]}', "not json", '{"test_id":"d","labels":[],"extra":1}']
open(badf, "w", encoding="utf-8").write("\n".join(lines) + "\n")
p = " | ".join(cs.validate_predictions(badf))
for token in ("repeated", "unknown category", "list of strings", "keys must be exactly", "not valid JSON"):
    assert token in p, (token, p)
print("all predict tests passed")
tmp.cleanup()
