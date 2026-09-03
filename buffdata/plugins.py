"""Extension points for custom validators and PII recognizers, discovered via standard
Python entry points -- so a team's custom rule lives in their own installable package
(pyproject.toml's [project.entry-points."buffdata.validators"] /
"buffdata.pii_recognizers"), not a fork of this one.

Scoped deliberately to these two extension points, not the whole pipeline: STAGES in
engine/pipeline.py stays the fixed, tested sequence it already is -- making the entire stage
order itself pluggable would mean arbitrary third-party code running inside every pipeline
stage, a much larger and riskier change than what's built here. Validators and PII
recognizers are the safe, additive case instead: a plugin can only ever contribute an extra
check or an extra thing to redact, layered onto the built-in behavior -- never remove or
reorder what buffdata already does on its own.

A plugin that fails to load or raises while running is logged and skipped, never allowed to
take down a run that doesn't otherwise depend on it.
"""

from __future__ import annotations

import atexit
import logging
from importlib.metadata import entry_points
from typing import Any, Callable, List, Optional, Protocol
from buffdata.security.policy import approved_plugins, current_context, SecurityError, SecurityPolicy

logger = logging.getLogger("buffdata.plugins")

VALIDATOR_GROUP = "buffdata.validators"
PII_RECOGNIZER_GROUP = "buffdata.pii_recognizers"

# Set to True only by buffdata/security/sandbox_worker.py, in its own process, before it
# touches any plugin code -- guards against the sandbox worker (which calls the real loader
# functions below) trying to spawn a sandbox of its own inside the sandbox.
_SANDBOX_ACTIVE = False
_sandbox = None


def _group_needs_sandbox(group: str) -> bool:
    return not _SANDBOX_ACTIVE and any(name.startswith(group + ":") for name in approved_plugins())


def _get_sandbox():
    global _sandbox
    if _sandbox is None:
        from buffdata.security.sandbox import PluginSandbox
        ctx = current_context()
        base_policy = ctx.policy if ctx else SecurityPolicy()
        # approved_plugins() is the single source of truth _group_needs_sandbox() already used
        # to decide to get here (execution-context policy, or BUFFDATA_APPROVED_PLUGINS when
        # there's no context) -- re-deriving it from ctx.policy alone here could disagree with
        # that decision (e.g. no context at all) and hand the child a stale/empty approval list.
        policy = base_policy.model_copy(update={"approved_plugins": sorted(approved_plugins())})
        _sandbox = PluginSandbox(policy)
        atexit.register(_sandbox.close)
    return _sandbox


class ValidatorPlugin(Protocol):
    """A callable taking a DatasetItem and returning a list of error strings (empty when
    the item is valid by this plugin's rule). Register it under the buffdata.validators
    entry-point group; buffdata calls it with no arguments beyond the item.
    """

    def __call__(self, item: Any) -> List[str]: ...


def _load_group(group: str) -> List[Any]:
    loaded: List[Any] = []
    try:
        points = entry_points(group=group)
    except Exception as exc:  # pragma: no cover -- defensive against unexpected environments
        logger.warning("Could not read entry-point group '%s': %s", group, exc)
        return loaded
    approvals = approved_plugins()
    points = list(points)
    required = {name for name in approvals if name.startswith(group + ":")}
    present = {group + ":" + ep.name for ep in points}
    if required - present:
        raise SecurityError("Approved plugin is missing: " + ", ".join(sorted(required - present)))
    for ep in points:
        if group + ":" + ep.name not in approvals:
            raise SecurityError("Installed plugin requires approval: " + group + ":" + ep.name)
        try:
            loaded.append(ep.load())
        except Exception as exc:
            raise SecurityError("Approved plugin failed to load: " + ep.name) from exc
    return loaded


_validator_cache: Optional[List[Callable[[Any], List[str]]]] = None
_pii_recognizer_cache: Optional[List[Any]] = None
_approval_key = None


def _check_approval_cache():
    global _approval_key, _validator_cache, _pii_recognizer_cache
    key = frozenset(approved_plugins())
    if key != _approval_key:
        _validator_cache = _pii_recognizer_cache = None
        _approval_key = key


def load_validator_plugins(*, refresh: bool = False) -> List[Callable[[Any], List[str]]]:
    """Cached (loaded once per process, matching how installed entry points don't change
    mid-run); pass refresh=True to force a re-scan, mainly for tests."""
    global _validator_cache
    _check_approval_cache()
    if _validator_cache is None or refresh:
        _validator_cache = _load_group(VALIDATOR_GROUP)
    return _validator_cache


def load_pii_recognizer_plugins(*, refresh: bool = False) -> List[Any]:
    """Each entry point should resolve to a presidio_analyzer.PatternRecognizer instance, a
    zero-argument factory returning one, or the class itself (instantiated with no
    arguments). Cached the same way as validator plugins.

    When any PII recognizer plugin is approved, construction happens inside the sandbox
    subprocess (see buffdata/security/sandbox.py) and this returns lightweight local proxies
    -- not the real recognizer objects -- that forward every analyze() call back to it.
    """
    global _pii_recognizer_cache
    if _group_needs_sandbox(PII_RECOGNIZER_GROUP):
        from buffdata.security.sandbox import build_sandboxed_recognizer
        sandbox = _get_sandbox()
        response = sandbox.call("pii_construct", {})
        return [build_sandboxed_recognizer(sandbox, metadata) for metadata in response["recognizers"]]
    _check_approval_cache()
    if _pii_recognizer_cache is not None and not refresh:
        return _pii_recognizer_cache
    recognizers: List[Any] = []
    for factory in _load_group(PII_RECOGNIZER_GROUP):
        try:
            recognizer = factory() if callable(factory) and not hasattr(factory, "analyze") else factory
            recognizers.append(recognizer)
        except Exception as exc:
            raise SecurityError("Approved PII plugin failed to initialize") from exc
    _pii_recognizer_cache = recognizers
    return _pii_recognizer_cache


def run_validator_plugins(item: Any) -> List[str]:
    """Run every loaded validator plugin against `item`, collecting all errors. A plugin
    that raises is logged and treated as producing no errors -- it must not fail the item
    (or the run) on its own account.

    When any validator plugin is approved, this call is routed into the sandbox subprocess
    (see buffdata/security/sandbox.py); the real loop below only ever runs there, or in-process
    when no validator plugin is approved.
    """
    if _group_needs_sandbox(VALIDATOR_GROUP):
        return _get_sandbox().call("validate", {"item": item.to_dict()})["errors"]
    errors: List[str] = []
    for plugin in load_validator_plugins():
        try:
            errors.extend(plugin(item) or [])
        except Exception as exc:
            raise SecurityError("Approved validator plugin failed") from exc
    return errors
