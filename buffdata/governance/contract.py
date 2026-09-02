"""Data Contracts: a versioned, declarative bar a dataset must clear, checked against the
artifacts an ordinary buffdata run already produces -- report.json, and either the
.accuracy_gate.json sibling `buffdata generate` writes or the metrics.accuracy_gate block
`buffdata optimize --require-positive-gain` writes into report.json itself.

This turns "however this team happened to set --min-relative-gain that one time" into a
reviewable, versioned YAML file a data-governance team owns, and a single
`buffdata contract check` command a CI pipeline can gate a merge/deploy on -- checking
already-produced artifacts, not re-running the pipeline, so it stays fast and deterministic
and doesn't need provider credentials at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field


class ContractRequirements(BaseModel):
    min_relative_accuracy_gain: Optional[float] = Field(
        None, ge=0.0, description="Requires accuracy-gate evidence showing at least this relative gain"
    )
    min_quality_score: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Requires a sampled/llm quality audit averaging at least this score"
    )
    max_rejected_fraction: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Rejected / (accepted + rejected) must not exceed this"
    )
    required_pii_policy: Optional[str] = Field(
        None, description="PII redaction policy the run must have used: identifiers, all, or off"
    )
    required_network_policy: Optional[str] = Field(
        None, description="Network policy the run must have used: unrestricted or strict"
    )
    required_classes: Optional[list[str]] = Field(
        None, description="Every one of these classes must appear in the dataset's detected/assigned classes"
    )


class DataContract(BaseModel):
    contract_version: str = "1"
    name: str
    description: str = ""
    requirements: ContractRequirements = Field(default_factory=ContractRequirements)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DataContract":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(**data)


class Violation(BaseModel):
    requirement: str
    expected: str
    actual: str


class ContractResult(BaseModel):
    contract_name: str
    contract_version: str
    passed: bool
    violations: list[Violation] = Field(default_factory=list)


def _load_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def check_contract(
    contract: DataContract,
    *,
    report_path: Union[str, Path],
    accuracy_gate_path: Optional[Union[str, Path]] = None,
) -> ContractResult:
    """Check a completed buffdata run's artifacts against a contract.

    report_path must exist (any buffdata command's *.report.json). accuracy_gate_path is
    only required when the contract sets min_relative_accuracy_gain and the evidence isn't
    already inside report.json's metrics.accuracy_gate block -- pass the .accuracy_gate.json
    that buffdata generate writes alongside its output in that case.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    report = _load_json(report_path)
    metrics = report.get("metrics", {})
    req = contract.requirements
    violations: list[Violation] = []

    if req.min_relative_accuracy_gain is not None:
        gate = _load_json(Path(accuracy_gate_path)) if accuracy_gate_path else metrics.get("accuracy_gate")
        relative_gain = None
        if gate:
            relative_gain = gate.get("relative_gain")
            if relative_gain is None:
                original = gate.get("original_accuracy_mean", gate.get("baseline_accuracy"))
                gain = gate.get("accuracy_gain")
                if gain is not None and original:
                    relative_gain = gain / original
        if relative_gain is None:
            violations.append(Violation(
                requirement="min_relative_accuracy_gain",
                expected=f">= {req.min_relative_accuracy_gain:.1%}",
                actual="no accuracy-gate evidence found (pass accuracy_gate_path, or run via "
                       "buffdata generate / optimize --require-positive-gain)",
            ))
        elif relative_gain < req.min_relative_accuracy_gain:
            violations.append(Violation(
                requirement="min_relative_accuracy_gain",
                expected=f">= {req.min_relative_accuracy_gain:.1%}",
                actual=f"{relative_gain:+.1%}",
            ))

    if req.min_quality_score is not None:
        score = metrics.get("stages", {}).get("score_refine", {}).get("average_overall_score")
        if score is None:
            violations.append(Violation(
                requirement="min_quality_score",
                expected=f">= {req.min_quality_score:.2f}",
                actual="unavailable (quality_mode was off, or used llm mode without a sampled average)",
            ))
        elif score < req.min_quality_score:
            violations.append(Violation(
                requirement="min_quality_score", expected=f">= {req.min_quality_score:.2f}", actual=f"{score:.2f}",
            ))

    if req.max_rejected_fraction is not None:
        accepted = metrics.get("accepted_records", 0)
        rejected = metrics.get("rejected_records", 0)
        total = accepted + rejected
        fraction = (rejected / total) if total else 0.0
        if fraction > req.max_rejected_fraction:
            violations.append(Violation(
                requirement="max_rejected_fraction",
                expected=f"<= {req.max_rejected_fraction:.1%}",
                actual=f"{fraction:.1%}",
            ))

    if req.required_pii_policy is not None:
        actual_policy = metrics.get("stages", {}).get("pii", {}).get("policy")
        if actual_policy != req.required_pii_policy:
            violations.append(Violation(
                requirement="required_pii_policy", expected=req.required_pii_policy, actual=str(actual_policy),
            ))

    if req.required_network_policy is not None:
        actual_network = metrics.get("network_policy")
        if actual_network != req.required_network_policy:
            violations.append(Violation(
                requirement="required_network_policy",
                expected=req.required_network_policy,
                actual=str(actual_network),
            ))

    if req.required_classes is not None:
        actual_classes = set(report.get("profile", {}).get("classes", []))
        missing = set(req.required_classes) - actual_classes
        if missing:
            violations.append(Violation(
                requirement="required_classes",
                expected=f"includes {sorted(req.required_classes)}",
                actual=f"missing {sorted(missing)}",
            ))

    return ContractResult(
        contract_name=contract.name,
        contract_version=contract.contract_version,
        passed=not violations,
        violations=violations,
    )
