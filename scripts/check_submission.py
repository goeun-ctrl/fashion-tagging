"""Check a submission before it is handed in.

    # 1) a predictions file against the image folder it came from
    python scripts/check_submission.py --jsonl predictions.jsonl --input_dir TEST_IMAGES

    # 2) the ZIP itself: required files at the top level, then predict.py is run from a copy of the ZIP in a
    #    folder whose name has spaces, with network access blocked, on the first --limit images, and the output checked
    python scripts/check_submission.py --zip team_name.zip --input_dir TEST_IMAGES --limit 20

Checks of the predictions file (assignment section 6): one JSON object per image; keys test_id and labels only;
test_id = file name without extension and unique; labels is a list of exact category names without repeats; [] is fine;
no missing or extra images. Exit code 0 = PASS, 1 = problems found.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ["predict.py", "model.pt", "requirements.txt", "README.md"]


def _predict_module():
    spec = importlib.util.spec_from_file_location("submission_predict", os.path.join(ROOT, "submission", "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_predictions(jsonl_path, input_dir=None):
    """Return a list of problems (empty = valid)."""
    pm = _predict_module()
    valid, problems, seen = set(pm.CATEGORIES), [], {}
    if not os.path.isfile(jsonl_path):
        return [f"{jsonl_path}: file not found"]
    with open(jsonl_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                problems.append(f"line {n}: empty line")
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {n}: not valid JSON ({e})")
                continue
            if not isinstance(rec, dict) or set(rec) != {"test_id", "labels"}:
                problems.append(f"line {n}: keys must be exactly test_id and labels, got {sorted(rec) if isinstance(rec, dict) else type(rec).__name__}")
                continue
            tid, labels = rec["test_id"], rec["labels"]
            if not isinstance(tid, str) or not tid:
                problems.append(f"line {n}: test_id must be a non-empty string")
                continue
            if tid in seen:
                problems.append(f"line {n}: test_id {tid!r} repeated (first on line {seen[tid]})")
            seen[tid] = n
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                problems.append(f"line {n} ({tid}): labels must be a list of strings")
                continue
            bad = [x for x in labels if x not in valid]
            if bad:
                problems.append(f"line {n} ({tid}): unknown category names {bad}")
            if len(set(labels)) != len(labels):
                problems.append(f"line {n} ({tid}): a category is repeated")
    if input_dir is not None:
        _, ids, _ = pm.list_images(input_dir)
        missing, extra = sorted(set(ids) - set(seen)), sorted(set(seen) - set(ids))
        if missing:
            problems.append(f"{len(missing)} images without a record, e.g. {missing[:3]}")
        if extra:
            problems.append(f"{len(extra)} records without an image, e.g. {extra[:3]}")
    return problems


GUARD = r"""
import runpy, socket, sys
def _blocked(*a, **k):
    raise RuntimeError("network access attempted during inference")
socket.socket.connect = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def check_zip(zip_path, input_dir, limit=20, extra_args=()):
    problems = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    top = {n for n in names if "/" not in n}
    for r in REQUIRED:
        if r not in top:
            problems.append(f"{r} is not at the top level of the zip")
    if problems:                                                   # nothing to run if the package is incomplete
        return problems
    pm = _predict_module()
    files, _, _ = pm.list_images(input_dir)
    if not files:
        return problems + [f"no images in {input_dir}"]
    with tempfile.TemporaryDirectory() as tmp:
        pkg, imgs, out_dir = (os.path.join(tmp, d) for d in ("unzipped copy", "some images", "output dir"))
        os.makedirs(imgs), os.makedirs(out_dir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(pkg)
        for f in files[:limit]:
            shutil.copy(os.path.join(input_dir, f), imgs)
        out = os.path.join(out_dir, "predictions.jsonl")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-c", GUARD, os.path.join(pkg, "predict.py"), "--input_dir", imgs, "--output", out,
               "--workers", "0", *extra_args]
        r = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stderr.strip()[-1500:])
        if r.returncode != 0:
            return problems + [f"predict.py exited with code {r.returncode}"]
        problems += validate_predictions(out, imgs)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl")
    ap.add_argument("--zip")
    ap.add_argument("--input_dir")
    ap.add_argument("--limit", type=int, default=20)
    a, rest = ap.parse_known_args()
    if bool(a.jsonl) == bool(a.zip):
        ap.error("give exactly one of --jsonl or --zip")
    if a.zip and not a.input_dir:
        ap.error("--zip needs --input_dir (a folder with some test images)")
    problems = validate_predictions(a.jsonl, a.input_dir) if a.jsonl else check_zip(a.zip, a.input_dir, a.limit, rest)
    print("PASS" if not problems else "FAIL")
    for p in problems:
        print(" -", p)
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
