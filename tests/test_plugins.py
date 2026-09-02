import pytest

from buffdata import plugins
from buffdata.engine.validator import DatasetValidator
from buffdata.models.schemas import DatasetItem


class _FakeEntryPoint:
    def __init__(self, name, target):
        self.name = name
        self._target = target

    def load(self):
        return self._target


class _FakeBrokenEntryPoint:
    """Simulates a real import/resolution failure -- .load() itself raises, matching what
    importlib.metadata.EntryPoint.load() does when the target module can't be imported or
    the attribute doesn't exist, as opposed to a callable that loads fine but raises when
    later invoked (that failure mode is run_validator_plugins's job to guard, tested
    separately below)."""

    def __init__(self, name):
        self.name = name

    def load(self):
        raise ImportError("simulated broken plugin package")


@pytest.fixture(autouse=True)
def _reset_plugin_caches():
    # load_validator_plugins/load_pii_recognizer_plugins cache into module-level globals the
    # first time they're called -- exactly the same hazard as get_default_secret_resolver's
    # singleton, so every test here must reset it before and after.
    plugins._validator_cache = None
    plugins._pii_recognizer_cache = None
    yield
    plugins._validator_cache = None
    plugins._pii_recognizer_cache = None


def test_load_validator_plugins_discovers_and_loads(monkeypatch):
    def no_op_validator(item):
        return []

    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeEntryPoint("custom", no_op_validator)] if group == plugins.VALIDATOR_GROUP else [],
    )
    loaded = plugins.load_validator_plugins(refresh=True)
    assert loaded == [no_op_validator]


def test_load_validator_plugins_is_cached_until_refresh(monkeypatch):
    calls = {"n": 0}

    def fake_entry_points(group):
        calls["n"] += 1
        return []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    plugins.load_validator_plugins(refresh=True)
    plugins.load_validator_plugins()
    plugins.load_validator_plugins()
    assert calls["n"] == 1  # only the first (refresh=True) call actually re-scanned

    plugins.load_validator_plugins(refresh=True)
    assert calls["n"] == 2


def test_load_validator_plugins_skips_a_plugin_that_fails_to_load(monkeypatch):
    def good_validator(item):
        return []

    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeBrokenEntryPoint("broken"), _FakeEntryPoint("good", good_validator)],
    )
    loaded = plugins.load_validator_plugins(refresh=True)
    assert loaded == [good_validator]  # the broken one is skipped, not fatal


def test_run_validator_plugins_aggregates_errors_from_multiple_plugins(monkeypatch):
    def flags_short_text(item):
        return ["too short"] if len(item.get_classification_text()) < 5 else []

    def flags_missing_label(item):
        return ["no label"] if item.labels is None else []

    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeEntryPoint("a", flags_short_text), _FakeEntryPoint("b", flags_missing_label)],
    )
    plugins.load_validator_plugins(refresh=True)

    item = DatasetItem.from_dict({"text": "hi"})
    errors = plugins.run_validator_plugins(item)
    assert set(errors) == {"too short", "no label"}


def test_run_validator_plugins_survives_a_plugin_that_raises_at_call_time(monkeypatch):
    def raises_when_called(item):
        raise ValueError("boom")

    def well_behaved(item):
        return ["expected error"]

    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeEntryPoint("bad", raises_when_called), _FakeEntryPoint("ok", well_behaved)],
    )
    plugins.load_validator_plugins(refresh=True)

    item = DatasetItem.from_dict({"text": "hello world"})
    errors = plugins.run_validator_plugins(item)
    assert errors == ["expected error"]  # the raising plugin didn't take the run down


def test_no_plugins_registered_yields_no_extra_errors(monkeypatch):
    monkeypatch.setattr(plugins, "entry_points", lambda group: [])
    plugins.load_validator_plugins(refresh=True)
    item = DatasetItem.from_dict({"text": "hello world"})
    assert plugins.run_validator_plugins(item) == []


# --- PII recognizer plugin loading ---------------------------------------------------------

class _FakeRecognizer:
    """Mimics presidio_analyzer.PatternRecognizer's shape closely enough for the loader's
    callable-vs-instance detection (has an `analyze` attribute)."""
    def analyze(self):
        pass


def test_load_pii_recognizer_plugins_uses_instance_directly(monkeypatch):
    instance = _FakeRecognizer()
    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeEntryPoint("custom-pii", instance)] if group == plugins.PII_RECOGNIZER_GROUP else [],
    )
    loaded = plugins.load_pii_recognizer_plugins(refresh=True)
    assert loaded == [instance]


def test_load_pii_recognizer_plugins_calls_zero_arg_factory(monkeypatch):
    marker = _FakeRecognizer()

    def factory():
        return marker

    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeEntryPoint("factory-pii", factory)] if group == plugins.PII_RECOGNIZER_GROUP else [],
    )
    loaded = plugins.load_pii_recognizer_plugins(refresh=True)
    assert loaded == [marker]


# --- end-to-end: DatasetValidator actually consults the plugin registry --------------------

def test_dataset_validator_includes_plugin_errors(monkeypatch):
    def rejects_everything(item):
        return ["rejected by policy plugin"]

    monkeypatch.setattr(
        plugins, "entry_points",
        lambda group: [_FakeEntryPoint("policy", rejects_everything)] if group == plugins.VALIDATOR_GROUP else [],
    )
    plugins.load_validator_plugins(refresh=True)

    item = DatasetItem.from_dict({"text": "a perfectly normal, valid-looking row"})
    errors = DatasetValidator.validate_item(item)
    assert "rejected by policy plugin" in errors


def test_dataset_validator_unaffected_when_no_plugins_installed(monkeypatch):
    monkeypatch.setattr(plugins, "entry_points", lambda group: [])
    plugins.load_validator_plugins(refresh=True)

    item = DatasetItem.from_dict({"text": "a perfectly normal, valid-looking row"})
    assert DatasetValidator.validate_item(item) == []
