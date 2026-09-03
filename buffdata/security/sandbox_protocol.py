"""Length-prefixed JSON framing shared by the plugin sandbox's parent and child.

A single source of truth for the wire format so the parent (`sandbox.py`) and the child
(`sandbox_worker.py`) can never drift out of sync with each other.
"""
from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

MAX_FRAME_BYTES = 64 * 1024**2
_LENGTH = struct.Struct(">I")


def write_frame(stream: BinaryIO, obj: Any) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError("Sandbox frame exceeds the size limit")
    stream.write(_LENGTH.pack(len(body)))
    stream.write(body)
    stream.flush()


def read_frame(stream: BinaryIO) -> Any:
    header = _read_exact(stream, _LENGTH.size)
    if header is None:
        raise EOFError("Sandbox stream closed before a frame header")
    (length,) = _LENGTH.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError("Sandbox frame exceeds the size limit")
    body = _read_exact(stream, length)
    if body is None:
        raise EOFError("Sandbox stream closed mid-frame")
    return json.loads(body.decode("utf-8"))


def _read_exact(stream: BinaryIO, count: int) -> bytes | None:
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
