"""Compatibility shim: the training script moved to scripts/train.py. `python common/train_eval.py ...` still works."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "train.py"),
               run_name="__main__")
