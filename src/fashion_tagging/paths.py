"""Repository paths, so code never guesses relative locations."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_TEMPLATE = CONFIG_DIR / "results_template.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train.py"
