import json

import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.governance.sbom import _normalize_purl_name, generate_sbom


def test_generate_sbom_has_correct_cyclonedx_envelope():
    sbom = generate_sbom()
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["serialNumber"].startswith("urn:uuid:")
    assert sbom["version"] == 1
    assert sbom["metadata"]["component"]["name"] == "buffdata"
    assert sbom["metadata"]["component"]["type"] == "application"


def test_generate_sbom_lists_real_installed_dependencies():
    sbom = generate_sbom()
    names = {c["name"].lower() for c in sbom["components"]}
    # pydantic and typer are hard dependencies of buffdata itself (pyproject.toml) -- if
    # they're not visible here, the SBOM isn't actually reflecting the real environment.
    assert "pydantic" in names
    assert any("typer" in name for name in names)


def test_generate_sbom_excludes_the_root_package_from_components():
    sbom = generate_sbom()
    names = {c["name"].lower() for c in sbom["components"]}
    assert "buffdata" not in names


def test_generate_sbom_components_have_required_fields_and_are_sorted():
    sbom = generate_sbom()
    components = sbom["components"]
    assert len(components) > 5  # buffdata has plenty of real dependencies
    for component in components:
        assert component["type"] == "library"
        assert component["name"]
        assert component["version"]
        assert component["purl"].startswith("pkg:pypi/")
    sorted_names = [c["name"].lower() for c in components]
    assert sorted_names == sorted(sorted_names)


def test_generate_sbom_has_no_duplicate_components():
    sbom = generate_sbom()
    names = [c["name"].lower() for c in sbom["components"]]
    assert len(names) == len(set(names))


def test_generate_sbom_serializes_to_json_cleanly():
    sbom = generate_sbom()
    payload = json.dumps(sbom)  # must not raise; every value must be JSON-safe
    assert json.loads(payload) == sbom


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pydantic", "pydantic"),
        ("Pillow_SIMD", "pillow-simd"),
        ("google-genai", "google-genai"),
        ("python.dotenv", "python-dotenv"),
        ("A__B..C", "a-b-c"),
    ],
)
def test_normalize_purl_name(raw, expected):
    assert _normalize_purl_name(raw) == expected


def test_generate_sbom_falls_back_when_root_package_not_installed():
    sbom = generate_sbom(root_package="definitely-not-a-real-installed-package-xyz")
    assert sbom["metadata"]["component"]["version"] == "0.0.0"


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_sbom_writes_valid_json(tmp_path):
    output = tmp_path / "sbom.json"
    result = CliRunner().invoke(app, ["sbom", "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["bomFormat"] == "CycloneDX"
    assert len(data["components"]) > 5


def test_cli_sbom_default_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["sbom"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "sbom.json").exists()
