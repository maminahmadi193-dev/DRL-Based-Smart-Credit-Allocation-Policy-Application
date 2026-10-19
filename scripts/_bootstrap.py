"""Allow scripts to import the existing env modules."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = PROJECT_ROOT / "env"

if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))
