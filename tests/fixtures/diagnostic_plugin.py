"""Test-only plugin. Never installed into the real dev environment -- tests/test_plugin_sandbox.py
copies this module next to a hand-built .dist-info directory in a temp dir and points a
PluginSandbox's PYTHONPATH at it, so a real subprocess does a real importlib.metadata lookup
against it. Behavior is controlled by a magic marker embedded in the item text it's given.
"""
import os
import socket
import time


def diagnostic_validator(item):
    text = item.get_classification_text() or ""
    if "DIAG:RAISE" in text:
        raise RuntimeError("diagnostic plugin failed")
    if "DIAG:SLEEP" in text:
        time.sleep(10)
        return []
    if "DIAG:CRASH" in text:
        os._exit(1)
    if "DIAG:NETWORK" in text:
        try:
            socket.create_connection(("93.184.216.34", 80), timeout=3)
            return ["network reachable"]
        except Exception as exc:
            return ["network blocked: " + type(exc).__name__]
    if "DIAG:ENV" in text:
        return ["env:" + ",".join(sorted(os.environ.keys()))]
    return []


def build_diagnostic_recognizer():
    # A zero-argument factory, not the bare class -- sidesteps load_pii_recognizer_plugins()'s
    # callable-vs-instance branching entirely by always handing back a real instance.
    from presidio_analyzer import Pattern, PatternRecognizer
    return PatternRecognizer(supported_entity="DIAGNOSTIC",
        patterns=[Pattern(name="diag", regex=r"DIAG-HIT", score=0.9)], name="diagnostic-pii")
