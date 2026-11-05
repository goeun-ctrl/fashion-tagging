"""Entry points use -m so Windows DataLoader workers can import the same modules."""
import sys
from pathlib import Path

ROLE = Path(__file__).resolve().parent.parent
# The baseline and common modules use script-local absolute imports.
sys.path.insert(0, str(ROLE))
from data_check import SPLITS, require
