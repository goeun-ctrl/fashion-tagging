"""Compatibility shim: moved to src/fashion_tagging/data/preprocess.py. Old code doing `from preprocess import ...`
with common/ on sys.path keeps working; new code should import fashion_tagging.data.preprocess."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.modules[__name__] = importlib.import_module("fashion_tagging.data.preprocess")
