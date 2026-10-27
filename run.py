#!/usr/bin/env python
"""Entry point that works without installation.

    python run.py <command> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from discourse_reliability.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
