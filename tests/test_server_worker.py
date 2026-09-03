import pytest
from buffdata.server.worker import secrets_for_project


def test_empty_secrets_file_yields_nothing():
    assert secrets_for_project({}, "alpha") == {}


def test_flat_shape_is_shared_across_every_project():
    flat = {"GEMINI_API_KEY": "shared-key"}
    assert secrets_for_project(flat, "alpha") == flat
    assert secrets_for_project(flat, "beta") == flat


def test_nested_shape_is_scoped_per_project():
    nested = {"alpha": {"GEMINI_API_KEY": "alpha-key"}, "beta": {"OPENAI_API_KEY": "beta-key"}}
    assert secrets_for_project(nested, "alpha") == {"GEMINI_API_KEY": "alpha-key"}
    assert secrets_for_project(nested, "beta") == {"OPENAI_API_KEY": "beta-key"}
    # A project with no entry of its own gets nothing -- not another project's secrets, and
    # not an error.
    assert secrets_for_project(nested, "gamma") == {}


def test_unknown_secret_key_is_rejected():
    with pytest.raises(ValueError, match="Invalid provider-secret file"):
        secrets_for_project({"NOT_A_REAL_KEY": "x"}, "alpha")
    with pytest.raises(ValueError, match="Invalid provider-secret file"):
        secrets_for_project({"alpha": {"NOT_A_REAL_KEY": "x"}}, "alpha")


def test_mixed_flat_and_nested_shapes_are_rejected():
    with pytest.raises(ValueError, match="Invalid provider-secret file"):
        secrets_for_project({"alpha": {"GEMINI_API_KEY": "x"}, "GOOGLE_API_KEY": "y"}, "alpha")


def test_consecutive_projects_never_see_each_others_secrets(monkeypatch):
    """The actual leak scenario worker.py's clear-then-set sequence guards against: a worker
    that just ran project alpha's claim must not carry alpha's key into beta's environment."""
    import os
    nested = {"alpha": {"GEMINI_API_KEY": "alpha-key"}, "beta": {}}
    from buffdata.server.worker import ALLOWED_SECRETS

    for key in ALLOWED_SECRETS:
        monkeypatch.delenv(key, raising=False)
    for key in ALLOWED_SECRETS:
        os.environ.pop(key, None)
    os.environ.update(secrets_for_project(nested, "alpha"))
    assert os.environ.get("GEMINI_API_KEY") == "alpha-key"

    for key in ALLOWED_SECRETS:
        os.environ.pop(key, None)
    os.environ.update(secrets_for_project(nested, "beta"))
    assert "GEMINI_API_KEY" not in os.environ
