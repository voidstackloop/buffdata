"""Plugin sandbox child entrypoint. Run only as ``python -I -m buffdata.security.sandbox_worker``
-- never imported for its own sake, and never spawns a sandbox of its own (see the
``_SANDBOX_ACTIVE`` guard this sets on ``buffdata.plugins`` below).

Network denial is installed before anything else runs, including before the handshake is
read: whatever this process does next, it can never reach a socket.
"""
from __future__ import annotations

import os
import sys

from buffdata.security.policy import ExecutionContext, SecurityError, SecurityPolicy, execution_context
from buffdata.security.sandbox_protocol import read_frame, write_frame


def _install_guard():
    from buffdata.security.network import install_worker_network_guard
    install_worker_network_guard(ExecutionContext(network="strict"))


def main():
    _install_guard()
    if os.name == "posix":
        os.umask(0o077)

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    handshake = read_frame(stdin)
    if handshake.get("kind") != "init":
        raise ValueError("Sandbox worker expected an init handshake")

    import buffdata.plugins as plugins
    plugins._SANDBOX_ACTIVE = True

    context = ExecutionContext(policy=SecurityPolicy(approved_plugins=handshake.get("approved_plugins", [])),
                               network="strict")
    resident_recognizers: dict[str, object] = {}

    def handle(kind: str, payload: dict) -> dict:
        if kind == "validate":
            from buffdata.models.schemas import DatasetItem
            item = DatasetItem.from_dict(payload["item"])
            return {"errors": plugins.run_validator_plugins(item)}
        if kind == "pii_construct":
            recognizers = plugins.load_pii_recognizer_plugins()
            resident_recognizers.clear()
            metadata = []
            for recognizer in recognizers:
                resident_recognizers[recognizer.name] = recognizer
                metadata.append({"name": recognizer.name, "supported_entities": list(recognizer.supported_entities),
                    "supported_language": recognizer.supported_language, "version": recognizer.version,
                    "context": list(recognizer.context or []), "country_code": recognizer.country_code(),
                    "score_thresholds": dict(recognizer.score_thresholds or {})})
            return {"recognizers": metadata}
        if kind == "pii_analyze":
            recognizer = resident_recognizers.get(payload["name"])
            if recognizer is None:
                raise SecurityError("Unknown sandboxed recognizer: " + str(payload.get("name")))
            findings = recognizer.analyze(text=payload["text"], entities=payload.get("entities") or [], nlp_artifacts=None)
            return {"results": [{"entity_type": f.entity_type, "start": f.start, "end": f.end, "score": f.score}
                                for f in findings]}
        raise ValueError("Unknown sandbox request kind: " + str(kind))

    with execution_context(context):
        while True:
            try:
                request = read_frame(stdin)
            except EOFError:
                return
            try:
                result = handle(request["kind"], request.get("payload", {}))
                write_frame(stdout, {"id": request["id"], "ok": True, "result": result})
            except Exception as exc:
                write_frame(stdout, {"id": request["id"], "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)}})


if __name__ == "__main__":
    main()
