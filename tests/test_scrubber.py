import subprocess
import sys
import venv

import pytest

from buffdata.models.schemas import DatasetItem
from buffdata.optimizers.scrubber import PIIScrubber
from buffdata.security.policy import (
    ExecutionContext, SecurityError, SecurityPolicy, execution_context, presidio_anonymizer_python,
)


def _fail_import(monkeypatch, module_name):
    monkeypatch.setitem(sys.modules, module_name, None)


@pytest.fixture(scope="module")
def isolated_presidio_venv(tmp_path_factory):
    """A real, freshly-built second venv holding only presidio-anonymizer + a pinned
    cryptography<49 -- proving the isolated-venv path for real, not mocked. Builds fresh into
    a pytest tmp dir (cleaned up automatically) rather than depending on any path outside the
    test run; skips (doesn't fail the suite) if package installation isn't possible in this
    environment (e.g. no network access), since this is a real-infra verification, not a pure
    unit test."""
    root = tmp_path_factory.mktemp("presidio-venv")
    venv.create(root, with_pip=True)
    python = root / "bin" / "python"
    try:
        subprocess.run([str(python), "-m", "pip", "install", "--quiet",
            "presidio-anonymizer", "cryptography<49.0.0"], check=True, capture_output=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Could not build an isolated presidio-anonymizer venv: {exc}")
    return str(python)


def test_missing_anonymizer_disables_detection_too_without_a_configured_venv(monkeypatch):
    """Detection without a way to act on it must not run at all -- see the coupling comment
    in PIIScrubber.__init__. Without presidio_anonymizer_python configured, this is today's
    exact regex-only fallback."""
    monkeypatch.delenv("BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON", raising=False)
    _fail_import(monkeypatch, "presidio_anonymizer")
    scrubber = PIIScrubber()
    assert scrubber.analyzer is None
    assert scrubber._anonymize is None


def test_missing_analyzer_disables_everything_regardless_of_anonymizer(monkeypatch):
    _fail_import(monkeypatch, "presidio_analyzer")
    scrubber = PIIScrubber()
    assert scrubber.analyzer is None


def test_isolated_venv_recovers_full_detection_and_redaction(monkeypatch, isolated_presidio_venv):
    """The real end-to-end path: presidio_anonymizer unimportable in-process, but a real
    second venv is configured -- full presidio detection + redaction still works, through a
    genuine subprocess, not a mock."""
    monkeypatch.setenv("BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON", isolated_presidio_venv)
    _fail_import(monkeypatch, "presidio_anonymizer")
    scrubber = PIIScrubber()
    assert scrubber.analyzer is not None
    assert scrubber._anonymize == scrubber._sandboxed_anonymize  # bound methods compare by ==, not is

    text, counts = scrubber.scrub_text_with_audit("my name is Bob and I live in Paris")
    assert "Bob" not in text
    assert counts  # presidio actually detected and redacted something via the subprocess


def test_counts_and_result_never_desync_when_anonymization_fails(monkeypatch):
    scrubber = PIIScrubber()
    if scrubber.analyzer is None:
        pytest.skip("presidio not available in this environment")

    def always_fails(text, findings):
        raise RuntimeError("simulated anonymizer failure")
    monkeypatch.setattr(scrubber, "_anonymize", always_fails)

    text = "contact bob@example.com right away"
    result, counts = scrubber.scrub_text_with_audit(text)
    # The presidio path failed entirely -- counts must reflect nothing redacted by it, not
    # the findings analyze() produced before the failure. The deterministic regex fallback
    # (which runs regardless, after the presidio branch) still catches the email.
    assert counts.get("EMAIL_ADDRESS", 0) <= 1  # only the regex fallback's own count, if any
    assert "<EMAIL_ADDRESS>" in result or "bob@example.com" not in result


def test_presidio_anonymizer_python_precedence_matches_approved_plugins(monkeypatch):
    monkeypatch.delenv("BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON", raising=False)
    assert presidio_anonymizer_python() is None

    monkeypatch.setenv("BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON", "/env/python")
    assert presidio_anonymizer_python() == "/env/python"

    with execution_context(ExecutionContext(policy=SecurityPolicy(presidio_anonymizer_python="/ctx/python"))):
        assert presidio_anonymizer_python() == "/ctx/python"
