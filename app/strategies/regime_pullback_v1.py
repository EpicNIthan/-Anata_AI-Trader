from __future__ import annotations

# Source is stored as a deterministic gzip sibling because the connected GitHub
# transport has a strict single-text-payload limit. Import behavior is identical.
import gzip as _gzip
from pathlib import Path as _Path

_source_path = _Path(__file__).with_suffix(_Path(__file__).suffix + ".gz")
_source = _gzip.decompress(_source_path.read_bytes())
exec(compile(_source, str(_source_path), "exec"), globals(), globals())
