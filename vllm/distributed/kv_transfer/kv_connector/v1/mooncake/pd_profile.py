# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated PD disaggregation timing marks for Mooncake profiling.

When ``VLLM_MOONCAKE_PD_PROFILE`` is set, each ``mark()`` call appends one
JSONL record with both ``time.time_ns()`` (wall clock, for cross-process
alignment) and ``time.perf_counter()`` (monotonic, for same-process deltas).

Records are correlated by ``transfer_id`` across proxy / prefill / decode.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any

_lock = threading.Lock()
_file_handles: dict[str, Any] = {}
_enabled: bool | None = None
_role: str | None = None
_outdir: str | None = None
_host: str | None = None
_pid: int | None = None


def enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = os.getenv("VLLM_MOONCAKE_PD_PROFILE", "").lower() in (
            "1",
            "true",
            "yes",
        )
    return _enabled


def set_role(role: str) -> None:
    """Override profile role (proxy / prefill / decode)."""
    global _role
    _role = role


def _get_role() -> str:
    global _role
    if _role is None:
        _role = os.getenv("VLLM_MOONCAKE_PD_PROFILE_ROLE", "unknown")
    return _role


def _get_outdir() -> str:
    global _outdir
    if _outdir is None:
        _outdir = os.getenv("VLLM_MOONCAKE_PD_PROFILE_DIR", "/tmp/vllm_pd_profile")
        os.makedirs(_outdir, exist_ok=True)
    return _outdir


def _get_handle(role: str):
    path = os.path.join(_get_outdir(), f"{role}.jsonl")
    if path not in _file_handles:
        with _lock:
            if path not in _file_handles:
                _file_handles[path] = open(path, "a", buffering=1)
    return _file_handles[path]


def mark(
    name: str,
    transfer_id: str | None = None,
    role: str | None = None,
    **extra: Any,
) -> None:
    """Record a timing mark. No-op when profiling is disabled."""
    if not enabled():
        return
    if not transfer_id:
        return

    global _host, _pid
    if _host is None:
        _host = socket.gethostname()
    if _pid is None:
        _pid = os.getpid()

    record: dict[str, Any] = {
        "mark": name,
        "transfer_id": transfer_id,
        "role": role or _get_role(),
        "time_ns": time.time_ns(),
        "perf_counter": time.perf_counter(),
        "pid": _pid,
        "host": _host,
    }
    if extra:
        record.update(extra)

    try:
        fh = _get_handle(record["role"])
        with _lock:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


def transfer_id_from_params(params: dict[str, Any] | None) -> str | None:
    if not params:
        return None
    tid = params.get("transfer_id")
    return str(tid) if tid else None
