import json

import pytest
import yaml
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.governance import DataContract, check_contract


def _write_report(path, metrics=None, profile=None, extra=None):
    payload = {
        "profile": profile or {},
        "metrics": metrics or {},
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- DataContract loading -----------------------------------------------------------------

def test_data_contract_loads_from_yaml(tmp_path):
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(yaml.safe_dump({
        "contract_version": "1",
        "name": "support-tickets",
        "description": "Minimum bar for support ticket data",
        "requirements": {"min_relative_accuracy_gain": 0.1, "min_quality_score": 7.0},
    }), encoding="utf-8")

    contract = DataContract.from_yaml(contract_file)
    assert contract.name == "support-tickets"
    assert contract.requirements.min_relative_accuracy_gain == 0.1
    assert contract.requirements.min_quality_score == 7.0
    assert contract.requirements.required_classes is None


def test_data_contract_defaults_to_no_requirements(tmp_path):
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(yaml.safe_dump({"name": "anything-goes"}), encoding="utf-8")
    contract = DataContract.from_yaml(contract_file)
    assert contract.requirements.min_relative_accuracy_gain is None


# --- check_contract: accuracy gain ---------------------------------------------------------

def test_check_contract_passes_with_sufficient_accuracy_gain_in_report(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "accuracy_gate": {"original_accuracy_mean": 0.80, "accuracy_gain": 0.10},
    })
    contract = DataContract(name="c", requirements={"min_relative_accuracy_gain": 0.10})
    result = check_contract(contract, report_path=report)
    assert result.passed is True
    assert result.violations == []


def test_check_contract_fails_with_insufficient_accuracy_gain(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "accuracy_gate": {"original_accuracy_mean": 0.80, "accuracy_gain": 0.02},
    })
    contract = DataContract(name="c", requirements={"min_relative_accuracy_gain": 0.10})
    result = check_contract(contract, report_path=report)
    assert result.passed is False
    assert result.violations[0].requirement == "min_relative_accuracy_gain"


def test_check_contract_uses_separate_accuracy_gate_file_from_generate(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={})
    gate_file = tmp_path / "out.accuracy_gate.json"
    gate_file.write_text(json.dumps({"baseline_accuracy": 0.5, "relative_gain": 0.15}), encoding="utf-8")

    contract = DataContract(name="c", requirements={"min_relative_accuracy_gain": 0.10})
    result = check_contract(contract, report_path=report, accuracy_gate_path=gate_file)
    assert result.passed is True


def test_check_contract_fails_when_no_accuracy_evidence_exists(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={})
    contract = DataContract(name="c", requirements={"min_relative_accuracy_gain": 0.10})
    result = check_contract(contract, report_path=report)
    assert result.passed is False
    assert "no accuracy-gate evidence" in result.violations[0].actual


# --- check_contract: other requirements -----------------------------------------------------

def test_check_contract_min_quality_score_pass_and_fail(tmp_path):
    good = _write_report(tmp_path / "good.report.json", metrics={
        "stages": {"score_refine": {"average_overall_score": 8.5}},
    })
    bad = _write_report(tmp_path / "bad.report.json", metrics={
        "stages": {"score_refine": {"average_overall_score": 5.0}},
    })
    contract = DataContract(name="c", requirements={"min_quality_score": 7.0})

    assert check_contract(contract, report_path=good).passed is True
    result = check_contract(contract, report_path=bad)
    assert result.passed is False
    assert result.violations[0].requirement == "min_quality_score"


def test_check_contract_max_rejected_fraction(tmp_path):
    within_bound = _write_report(tmp_path / "ok.report.json", metrics={
        "accepted_records": 90, "rejected_records": 10,
    })
    over_bound = _write_report(tmp_path / "over.report.json", metrics={
        "accepted_records": 70, "rejected_records": 30,
    })
    contract = DataContract(name="c", requirements={"max_rejected_fraction": 0.15})

    assert check_contract(contract, report_path=within_bound).passed is True
    result = check_contract(contract, report_path=over_bound)
    assert result.passed is False
    assert result.violations[0].requirement == "max_rejected_fraction"


def test_check_contract_required_pii_policy(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "stages": {"pii": {"policy": "all"}},
    })
    contract = DataContract(name="c", requirements={"required_pii_policy": "identifiers"})
    result = check_contract(contract, report_path=report)
    assert result.passed is False
    assert result.violations[0].actual == "all"


def test_check_contract_required_network_policy(tmp_path):
    strict_report = _write_report(tmp_path / "strict.report.json", metrics={"network_policy": "strict"})
    open_report = _write_report(tmp_path / "open.report.json", metrics={"network_policy": "unrestricted"})
    contract = DataContract(name="c", requirements={"required_network_policy": "strict"})

    assert check_contract(contract, report_path=strict_report).passed is True
    assert check_contract(contract, report_path=open_report).passed is False


def test_check_contract_required_classes(tmp_path):
    report = _write_report(
        tmp_path / "out.report.json",
        metrics={},
        profile={"classes": ["World", "Sports", "Business"]},
    )
    satisfied = DataContract(name="c", requirements={"required_classes": ["World", "Sports"]})
    unsatisfied = DataContract(name="c", requirements={"required_classes": ["World", "Sci/Tech"]})

    assert check_contract(satisfied, report_path=report).passed is True
    result = check_contract(unsatisfied, report_path=report)
    assert result.passed is False
    assert "Sci/Tech" in result.violations[0].actual


def test_check_contract_reports_every_violation_at_once(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "accepted_records": 50, "rejected_records": 50,
        "stages": {"pii": {"policy": "off"}},
    })
    contract = DataContract(name="c", requirements={
        "max_rejected_fraction": 0.10,
        "required_pii_policy": "identifiers",
    })
    result = check_contract(contract, report_path=report)
    assert result.passed is False
    assert {v.requirement for v in result.violations} == {"max_rejected_fraction", "required_pii_policy"}


def test_check_contract_missing_report_raises(tmp_path):
    contract = DataContract(name="c")
    with pytest.raises(FileNotFoundError):
        check_contract(contract, report_path=tmp_path / "nonexistent.report.json")


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_contract_check_passes(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "accepted_records": 95, "rejected_records": 5,
    })
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(yaml.safe_dump({
        "name": "demo", "requirements": {"max_rejected_fraction": 0.10},
    }), encoding="utf-8")

    result = CliRunner().invoke(app, [
        "contract", "check", str(contract_file), "--report", str(report),
    ])
    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_cli_contract_check_fails_with_nonzero_exit(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "accepted_records": 50, "rejected_records": 50,
    })
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(yaml.safe_dump({
        "name": "demo", "requirements": {"max_rejected_fraction": 0.10},
    }), encoding="utf-8")

    result = CliRunner().invoke(app, [
        "contract", "check", str(contract_file), "--report", str(report),
    ])
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "max_rejected_fraction" in result.output
