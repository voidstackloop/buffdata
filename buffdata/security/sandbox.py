"""Parent side of the plugin sandbox: a long-lived subprocess that runs approved
validator/PII-recognizer plugins outside the main process's address space.

Lazily started on first use (``buffdata/plugins.py`` only reaches here when at least one
plugin is approved) and never auto-respawned after a failure -- a systemic plugin crash
should surface as a run failure, not hide as intermittent flakiness. Reuses
``install_worker_network_guard`` (unconditional network denial, ``network="strict"``) and
the approval/caching/fail-closed logic already in ``buffdata/plugins.py`` -- this module
only adds the process boundary, not new plugin-loading logic.
"""
from __future__ import annotations

import itertools
import os
import queue
import signal
import subprocess
import sys
import threading

from buffdata.security.policy import SecurityError, SecurityPolicy
from buffdata.security.sandbox_protocol import read_frame, write_frame

# Allowlist, not runner.supervise()'s denylist: this subprocess runs lower-trust code than
# BuffData's own optimizer subprocess and must never see provider API keys or any other
# secret sitting in the parent's environment. Same precedent already used for a lower-trust
# external subprocess in buffdata/integrations/graphify.py.
_ENV_ALLOWLIST = {"PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP"}


class PluginSandbox:
    def __init__(self, policy: SecurityPolicy, *, python_executable: str | None = None,
                 worker_argv: list[str] | None = None, extra_env: dict[str, str] | None = None):
        self.policy = policy
        self._python_executable = python_executable or sys.executable
        self._worker_argv = worker_argv or ["-m", "buffdata.security.sandbox_worker"]
        # extra_env is a testing seam only -- buffdata/plugins.py's _get_sandbox() never
        # passes it, so the production allowlist below is the only thing that ever governs
        # what a real caller's sandbox sees.
        self._extra_env = dict(extra_env or {})
        self._process: subprocess.Popen | None = None
        self._poisoned = False
        self._lock = threading.Lock()
        self._next_id = itertools.count(1)

    def _start(self):
        env = {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}
        env.update(self._extra_env)
        self._process = subprocess.Popen([self._python_executable, *self._worker_argv],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix", env=env)
        write_frame(self._process.stdin, {"kind": "init", "approved_plugins": sorted(self.policy.approved_plugins)})

    def call(self, kind: str, payload: dict) -> dict:
        with self._lock:
            if self._poisoned:
                raise SecurityError("Sandbox is unavailable after a previous failure")
            try:
                # The first-ever spawn lives inside this same try/except (not before it) so a
                # bad python_executable/worker_argv fails once, poisons, and raises the same
                # SecurityError every later call would -- not a raw OSError retried forever.
                if self._process is None:
                    self._start()
                elif self._process.poll() is not None:
                    raise RuntimeError("Sandboxed plugin process exited unexpectedly")
                request_id = next(self._next_id)
                write_frame(self._process.stdin, {"id": request_id, "kind": kind, "payload": payload})
                response = self._read_response_with_timeout()
            except Exception as exc:
                self._poison_locked()
                raise SecurityError("Sandboxed plugin process failed: " + type(exc).__name__) from exc
            if response is None:
                self._poison_locked()
                raise SecurityError("Sandboxed plugin call timed out")
            if response.get("id") != request_id:
                self._poison_locked()
                raise SecurityError("Sandbox protocol desynchronized")
            if not response.get("ok"):
                error = response.get("error", {})
                raise SecurityError(error.get("message", "Sandboxed plugin call failed"))
            return response["result"]

    def _read_response_with_timeout(self):
        # A queue-fed reader thread, not select(): select() only supports sockets on
        # Windows, not pipes, and this has to work on the user's own machine.
        result: queue.Queue = queue.Queue(maxsize=1)

        def reader():
            try:
                result.put(("ok", read_frame(self._process.stdout)))
            except Exception as exc:
                result.put(("error", exc))

        threading.Thread(target=reader, daemon=True).start()
        try:
            kind, value = result.get(timeout=self.policy.subprocess_seconds)
        except queue.Empty:
            return None
        if kind == "error":
            raise value
        return value

    def _poison_locked(self):
        self._poisoned = True
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            except Exception:
                pass

    def close(self):
        with self._lock:
            if self._process is None:
                return
            process, self._process = self._process, None
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


_proxy_class = None


def _get_proxy_class():
    # Lazy: importing this module must not force a presidio_analyzer import for a
    # validator-only sandbox (no PII recognizer plugins approved). By the time a PII proxy
    # is actually being built, PIIScrubber.__init__ has already imported presidio_analyzer
    # successfully -- this is never the first place it's imported in the process.
    global _proxy_class
    if _proxy_class is None:
        from presidio_analyzer import EntityRecognizer, RecognizerResult

        class _SandboxedRecognizerProxy(EntityRecognizer):
            """Construction and every ``analyze()`` call actually happen inside the sandbox
            subprocess. Presidio's ``RecognizerRegistry.add_recognizer()`` requires a real,
            local ``EntityRecognizer`` instance (an ``isinstance`` check), so a pure RPC
            handle can't be registered directly -- this proxy carries the real recognizer's
            static metadata locally (so engine-side context-word score enhancement, a
            property lookup on whatever sits in the registry, behaves identically to the
            unsandboxed case) and forwards every ``analyze()`` call to the resident
            recognizer object living in the child.

            Known limitation, not silently absorbed: ``nlp_artifacts`` (a live spaCy ``Doc``-
            derived object, computed engine-side) never crosses the process boundary --
            ``PatternRecognizer``'s base ``analyze()`` doesn't use it, but a plugin overriding
            ``analyze()`` to consume it directly will see ``nlp_artifacts=None`` here.
            """

            def __init__(self, sandbox: PluginSandbox, metadata: dict):
                self._sandbox = sandbox
                super().__init__(supported_entities=metadata["supported_entities"], name=metadata["name"],
                    supported_language=metadata["supported_language"], version=metadata["version"],
                    context=metadata["context"] or None, country_code=metadata["country_code"],
                    score_thresholds=metadata["score_thresholds"] or None)

            def load(self) -> None:
                pass

            def analyze(self, text, entities, nlp_artifacts=None):
                response = self._sandbox.call("pii_analyze",
                    {"name": self.name, "text": text, "entities": list(entities or [])})
                return [RecognizerResult(entity_type=r["entity_type"], start=r["start"], end=r["end"], score=r["score"])
                        for r in response["results"]]

        _proxy_class = _SandboxedRecognizerProxy
    return _proxy_class


def build_sandboxed_recognizer(sandbox: PluginSandbox, metadata: dict):
    return _get_proxy_class()(sandbox, metadata)
