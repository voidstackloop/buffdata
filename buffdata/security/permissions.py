"""Restrictive file permissions for locally-written files that can carry sensitive
content -- pipeline checkpoints (which can briefly hold pre-PII-redaction text, since
the "pii" stage runs after "validate"'s own checkpoint is saved -- see
buffdata/engine/pipeline.py's STAGES), report.json (record counts, provider/model,
dataset-revealing file paths), and the audit database. Owner-only (0600) on POSIX,
where "everyone else on this machine can read your files by default" is the real
default; a best-effort no-op everywhere else, since os.chmod doesn't express meaningful
ACLs on Windows and NTFS's own per-user-profile isolation already covers the common
case there.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Union


def restrict_to_owner(path: Union[str, Path]) -> None:
    """Set owner-read-write-only (0600) permissions on `path`. Silently does nothing if
    the path doesn't exist, or if the platform/filesystem doesn't support chmod (e.g. a
    FAT-formatted mount) -- this is defense in depth, not a guarantee the rest of the
    pipeline should ever depend on for correctness, so a failure here must never fail
    the write it's protecting.
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
