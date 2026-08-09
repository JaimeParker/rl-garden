"""Binary protocol shared by the Cal-QL JAX client and Minari env server."""

from __future__ import annotations

import struct
from typing import BinaryIO


MAGIC = b"CQM1"
OP_RESET = 1
OP_STEP = 2
OP_CLOSE = 3
STATUS_OK = 1
STATUS_ERROR = 0

HANDSHAKE = struct.Struct("<4sIII")
RESET_REQUEST = struct.Struct("<Bq")
STEP_REQUEST = struct.Struct("<B")
CLOSE_REQUEST = struct.Struct("<B")
STATUS = struct.Struct("<B")
STEP_RESULT = struct.Struct("<fBB")
ERROR_LENGTH = struct.Struct("<I")


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("environment bridge closed unexpectedly")
        chunks.extend(chunk)
    return bytes(chunks)


def write_error(stream: BinaryIO, message: str) -> None:
    encoded = message.encode("utf-8", errors="replace")
    stream.write(STATUS.pack(STATUS_ERROR))
    stream.write(ERROR_LENGTH.pack(len(encoded)))
    stream.write(encoded)
    stream.flush()


def read_status(stream: BinaryIO) -> None:
    (status,) = STATUS.unpack(read_exact(stream, STATUS.size))
    if status == STATUS_OK:
        return
    (length,) = ERROR_LENGTH.unpack(read_exact(stream, ERROR_LENGTH.size))
    message = read_exact(stream, length).decode("utf-8", errors="replace")
    raise RuntimeError("environment bridge error: " + message)
