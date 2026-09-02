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

import logging
from importlib.metadata import entry_points
from typing import Any, Callable, List, Optional, Protocol

logger = logging.getLogger("buffdata.plugins")

VALIDATOR_GROUP = "buffdata.validators"
PII_RECOGNIZER_GROUP = "buffdata.pii_recognizers"


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
    for ep in points:
        try:
            loaded.append(ep.load())
        except Exception as exc:
            logger.warning("Failed to load buffdata plugin '%s' from group '%s': %s", ep.name, group, exc)
    return loaded


_validator_cache: Optional[List[Callable[[Any], List[str]]]] = None
_pii_recognizer_cache: Optional[List[Any]] = None


def load_validator_plugins(*, refresh: bool = False) -> List[Callable[[Any], List[str]]]:
    """Cached (loaded once per process, matching how installed entry points don't change
    mid-run); pass refresh=True to force a re-scan, mainly for tests."""
    global _validator_cache
    if _validator_cache is None or refresh:
        _validator_cache = _load_group(VALIDATOR_GROUP)
    return _validator_cache


def load_pii_recognizer_plugins(*, refresh: bool = False) -> List[Any]:
    """Each entry point should resolve to a presidio_analyzer.PatternRecognizer instance, a
    zero-argument factory returning one, or the class itself (instantiated with no
    arguments). Cached the same way as validator plugins.
    """
    global _pii_recognizer_cache
    if _pii_recognizer_cache is not None and not refresh:
        return _pii_recognizer_cache
    recognizers: List[Any] = []
    for factory in _load_group(PII_RECOGNIZER_GROUP):
        try:
            recognizer = factory() if callable(factory) and not hasattr(factory, "analyze") else factory
            recognizers.append(recognizer)
        except Exception as exc:
            logger.warning("Failed to instantiate PII recognizer plugin '%s': %s", factory, exc)
    _pii_recognizer_cache = recognizers
    return _pii_recognizer_cache


def run_validator_plugins(item: Any) -> List[str]:
    """Run every loaded validator plugin against `item`, collecting all errors. A plugin
    that raises is logged and treated as producing no errors -- it must not fail the item
    (or the run) on its own account.
    """
    errors: List[str] = []
    for plugin in load_validator_plugins():
        try:
            errors.extend(plugin(item) or [])
        except Exception as exc:
            logger.warning("Validator plugin '%s' raised while checking an item: %s", plugin, exc)
    return errors
