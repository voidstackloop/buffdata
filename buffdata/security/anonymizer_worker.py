"""Standalone presidio-anonymizer sandbox worker. Deliberately has NO ``buffdata`` imports --
even something as small as ``from buffdata.security.sandbox_protocol import ...`` first runs
``buffdata/__init__.py``, which eagerly imports most of the base dependency closure (orjson,
polars, pandas, pyarrow, huggingface_hub, datasets, PyMuPDF, aiohttp, sqlalchemy, ...),
defeating the point of an isolated, minimal venv holding only ``presidio-anonymizer`` and its
older-pinned ``cryptography``. Run this file directly by path
(``<isolated-venv-python> /path/to/anonymizer_worker.py``), never as ``-m
buffdata.security.anonymizer_worker`` -- direct script execution never imports the would-be
parent packages, which is exactly what keeps this venv's install to two packages.

Wire format matches ``buffdata/security/sandbox_protocol.py`` exactly (duplicated here on
purpose, not imported -- see above): a 4-byte big-endian length prefix + UTF-8 JSON body,
request/response, one at a time. Network denial is installed before anything else runs,
including before the very first frame is read.
"""
import json
import os
import struct
import sys

MAX_FRAME_BYTES = 64 * 1024**2
_LENGTH = struct.Struct(">I")


def write_frame(stream, obj):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    stream.write(_LENGTH.pack(len(body)))
    stream.write(body)
    stream.flush()


def read_frame(stream):
    header = _read_exact(stream, _LENGTH.size)
    if header is None:
        raise EOFError("Anonymizer worker stream closed before a frame header")
    (length,) = _LENGTH.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError("Anonymizer worker frame exceeds the size limit")
    body = _read_exact(stream, length)
    if body is None:
        raise EOFError("Anonymizer worker stream closed mid-frame")
    return json.loads(body.decode("utf-8"))


def _read_exact(stream, count):
    chunks, remaining = [], count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def install_network_deny():
    # Unconditional denial -- this worker never legitimately needs a socket, so there's no
    # allowlist to consult (unlike buffdata.security.network's install_worker_network_guard,
    # which supports one; that generality isn't needed or importable here).
    def audit(event, args):
        if event.startswith("socket.") or event in {"urllib.Request", "ftplib.connect"}:
            raise PermissionError("Anonymizer worker network access is not permitted")
    sys.addaudithook(audit)


def main():
    install_network_deny()
    if os.name == "posix":
        os.umask(0o077)

    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import RecognizerResult

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    handshake = read_frame(stdin)
    if handshake.get("kind") != "init":
        raise ValueError("Anonymizer worker expected an init handshake")

    engine = AnonymizerEngine()

    def handle(kind, payload):
        if kind != "anonymize":
            raise ValueError("Unknown anonymizer worker request kind: " + str(kind))
        findings = [RecognizerResult(entity_type=f["entity_type"], start=f["start"], end=f["end"], score=f["score"])
                    for f in payload["analyzer_results"]]
        result = engine.anonymize(text=payload["text"], analyzer_results=findings)
        return {"text": result.text}

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
