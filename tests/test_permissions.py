import os
import stat
import sys

import pytest

from buffdata.security.permissions import restrict_to_owner

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod doesn't express meaningful owner-only ACLs on Windows; this module is a best-effort no-op there by design.",
)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_restrict_to_owner_sets_0600(tmp_path):
    target = tmp_path / "secret.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o644)  # start world-readable, the common default umask result

    restrict_to_owner(target)

    assert _mode(target) == 0o600


def test_restrict_to_owner_is_a_silent_noop_for_a_missing_path(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    restrict_to_owner(missing)  # must not raise
