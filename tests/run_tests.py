"""Run automated tests from the repository root.

Usage:
    python tests/run_tests.py
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = PROJECT_ROOT / "env"
TESTS_DIR = PROJECT_ROOT / "tests"

# Preserve existing imports from modules inside env/.
sys.path.insert(0, str(ENV_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TESTS_DIR),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
