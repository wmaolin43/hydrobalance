"""Test configuration.

English:
    Allow running `pytest` without installing the package.

日本語:
    パッケージをインストールしなくても `pytest` が動くように
    import path を調整します。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root contains the `hydrobalance/` package directory.
ROOT = Path(__file__).resolve().parents[1]

# EN: Add the parent dir so `import hydrobalance` works.
# JP: `import hydrobalance` が通るように親ディレクトリを追加。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
