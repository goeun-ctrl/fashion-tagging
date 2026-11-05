"""FashionFormer -> image-level 27-class scores on our val. Run from the FashionFormer repo root:
    python ff_infer.py <config> <checkpoint> <val_dir> <out.npz>
Image score for class c = max sigmoid class score over the model's top-k (mask, class) predictions, i.e. the
same scores mmdet would put in bbox_result[c][:, 4]. Masks are not decoded (not needed for image labels)."""
import csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmdet.apis import init_detector, inference_detector

cfg, ckpt, val_dir, out = sys.argv[1:5]
val_dir = Path(val_dir)
cfg = Config.fromfile(cfg)
import_modules_from_strings(**cfg["custom_imports"])  # registers FashionFormer modules, as tools/test.py does
print("test pipeline:", cfg.data.test.pipeline, flush=True)
model = init_detector(cfg, ckpt, device="cuda:0")
head = model.roi_head.mask_head[-1]
head.get_seg_masks = lambda masks, labels, scores, attr, test_cfg, meta: (labels.cpu().numpy(), scores.cpu().numpy())
n_params = sum(p.numel() for p in model.parameters())
cats = [l.strip() for l in open(val_dir / "categories.txt", encoding="utf-8-sig") if l.strip()]
classes = list(model.CLASSES)
assert classes[:27] == cats, (classes[:27], cats)
rows = list(csv.DictReader(open(val_dir / "labels.csv", encoding="utf-8-sig")))
names = [r["file_name"] for r in rows]
z = np.load(val_dir / "labels.npz", allow_pickle=False)
key = {k: i for i, k in enumerate(z["filenames"].tolist())}
targets = z["targets"][[key[n] for n in names]]
scores = np.zeros((len(names), 46), np.float32)
t0 = time.perf_counter()
with torch.no_grad():
    for i, n in enumerate(names):
        labels, s = inference_detector(model, str(val_dir / "images" / n))
        np.maximum.at(scores[i], labels, s)
        if i % 200 == 0:
            print(i, f"{time.perf_counter() - t0:.0f}s", flush=True)
sec = (time.perf_counter() - t0) / len(names)
np.savez(out, filenames=np.array(names), targets=targets, scores=scores[:, :27], scores46=scores,
         categories=np.array(cats))
json.dump({"params_total": int(n_params), "sec_per_image": sec, "num_classes_model": len(classes)},
          open(str(out) + ".json", "w"), indent=1)
print("params", n_params, "sec/img", round(sec, 3))
