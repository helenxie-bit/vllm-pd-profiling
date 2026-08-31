#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze Mooncake PD profile JSONL marks per IETF AI Fabric Inference Bench.

Maps instrumentation to draft-calabria-bmwg-ai-fabric-inference-bench-04
Test Category 2 (Prefill/Decode Disaggregation) KPIs where applicable.

Usage:
  python analyze_pd_profile.py --dir /tmp/vllm_pd_profile_1p1d
  python analyze_pd_profile.py --dir /tmp/run --csv results.csv --json results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Per-request IETF-aligned metric keys (Section 6.1 decomposition + Section 4).
IETF_METRIC_KEYS = (
    "ttft_ms",
    "t_prefill_ms",
    # Prefill stage decomposition: A0→B→C_sched→C_ready→I0 / C_ready→I1
    "a0_to_b_ms",
    "a0_to_d0_ms",
    "a0_to_b_ttft_fraction",
    "plan_ttft_fraction",
    "ttft_fabric_ttft_fraction",
    "promote_lag_ttft_fraction",
    "b_to_c_sched_ms",
    "c_sched_to_c_ready_ms",
    "c_ready_to_i0_ms",
    "c_ready_to_i1_ms",
    "h0_to_h1_ms",
    "h1_to_i0_ms",
    "m_to_n0_ms",
    "n0_to_n1_ms",
    "k1_to_l1_ms",
    "i1_to_j0_ms",
    "j1_to_k0_ms",
    "t_transfer_ms",
    "ttft_fabric_ms",
    "t_decode_init_ms",
    "fabric_fraction",
    "kv_xfer_latency_us",
    "kv_xfer_bandwidth_gbps",
    "fabric_fct_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "itl_mean_ms",
    "itl_count",
    "wait_ready_ms",
    "plan_ms",
    "rdma_ms",
    "promote_lag_ms",
    "num_prompt_tokens",
    "kv_bytes",
    "num_descriptors",
    "bytes_per_descriptor",
)


def load_marks(profile_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_xfer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(profile_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec.get("transfer_id")
                if tid:
                    by_xfer[tid].append(rec)
    for tid in by_xfer:
        by_xfer[tid].sort(key=lambda r: r["time_ns"])
    return by_xfer


def first_mark(
    marks: list[dict[str, Any]], name: str, role: str | None = None
) -> dict[str, Any] | None:
    for m in marks:
        if m["mark"] != name:
            continue
        if role is not None and m.get("role") != role:
            continue
        return m
    return None


def all_marks(
    marks: list[dict[str, Any]], name: str, role: str | None = None
) -> list[dict[str, Any]]:
    out = []
    for m in marks:
        if m["mark"] != name:
            continue
        if role is not None and m.get("role") != role:
            continue
        out.append(m)
    return out


def delta_ns(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if a is None or b is None:
        return None
    return (b["time_ns"] - a["time_ns"]) / 1e6


def delta_perf(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if a is None or b is None:
        return None
    if a.get("pid") != b.get("pid") or a.get("host") != b.get("host"):
        return delta_ns(a, b)
    return (b["perf_counter"] - a["perf_counter"]) * 1e3


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def clean_floats(vals: list[float | None]) -> list[float]:
    return [v for v in vals if v is not None and not math.isnan(v)]


def summary_stats(
    vals: list[float | None], *, include_p95: bool = True
) -> dict[str, float]:
    """Aggregate per-request values into mean and percentile summary."""
    nums = clean_floats(vals)
    if not nums:
        return {}
    stats: dict[str, float] = {
        "mean": statistics.mean(nums),
        "p50": percentile(nums, 0.5),
        "p99": percentile(nums, 0.99),
    }
    if include_p95:
        stats["p95"] = percentile(nums, 0.95)
    return stats


def xfer_granularity_from_k1(
    k1: dict[str, Any] | None,
) -> tuple[int | None, float | None]:
    """Read Mooncake transfer granularity from prefill K1.

    ``num_descriptors`` is ``len(src_ptrs)`` for the ``batch_transfer_sync_write``
    call (same value on K0/K1). ``bytes_per_descriptor`` is the mean payload
    size per descriptor: ``total_bytes / num_descriptors``.

    When several requests share one batched RDMA call, each request's K1 mark
    carries the **batch** totals (profiling attribution), so
    ``bytes_per_descriptor`` is batch bytes divided by batch descriptor count,
    not per-request KV size alone.
    """
    if k1 is None:
        return None, None
    raw_desc = k1.get("num_descriptors")
    if raw_desc is None:
        return None, None
    num_descriptors = int(raw_desc)
    if num_descriptors <= 0:
        return num_descriptors, None
    raw_bytes = k1.get("total_bytes")
    if raw_bytes is None:
        return num_descriptors, None
    return num_descriptors, int(raw_bytes) / num_descriptors


def itl_samples(marks: list[dict[str, Any]]) -> list[float]:
    """Inter-token latency (ITL) samples from proxy N1 and O marks."""
    n1 = first_mark(marks, "N1", "proxy")
    o_marks = all_marks(marks, "O", "proxy")
    vals: list[float] = []
    prev = n1
    for o in o_marks:
        d = delta_perf(prev, o)
        if d is not None and d >= 0:
            vals.append(d)
        prev = o
    return vals


def compute_ietf_metrics(marks: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compute per-request metrics aligned with IETF Cat-2 / fabric KPIs."""
    a0 = first_mark(marks, "A0", "proxy")
    a1 = first_mark(marks, "A1", "proxy")
    b = first_mark(marks, "B", "prefill")
    c_sched = first_mark(marks, "C_sched", "prefill")
    c_ready = first_mark(marks, "C_ready", "prefill")
    # Fallback for older traces that only emitted B at the sched handoff.
    c_sched_or_b = c_sched or b
    d0 = first_mark(marks, "D0", "proxy")
    h0 = first_mark(marks, "H0", "decode")
    h1 = first_mark(marks, "H1", "prefill")
    i0 = first_mark(marks, "I0", "prefill")
    i1 = first_mark(marks, "I1", "prefill")
    j0 = first_mark(marks, "J0", "prefill")
    j1 = first_mark(marks, "J1", "prefill")
    k0 = first_mark(marks, "K0", "prefill")
    k1 = first_mark(marks, "K1", "prefill")
    l1 = first_mark(marks, "L1", "decode")
    m = first_mark(marks, "M", "decode")
    n0 = first_mark(marks, "N0", "decode")
    n1 = first_mark(marks, "N1", "proxy")
    g0 = first_mark(marks, "G0", "decode")
    g1 = first_mark(marks, "G1", "decode")

    ttft_ms = delta_ns(a0, n1)
    # Section 6.1 TTFT decomposition (SUT-E / DUT-PD proxies).
    t_prefill_ms = delta_ns(a0, a1) or delta_ns(a0, c_sched_or_b)
    t_transfer_ms = delta_ns(h0, l1)  # D-side KV pull stall (H0→L1)
    ttft_fabric_ms = delta_perf(k0, k1)  # RDMA segment (DUT-PD core)
    t_decode_init_ms = delta_ns(l1, n1) or delta_ns(m, n1) or delta_ns(n0, n1)

    # Prefill stage chain (may be negative if marks race / reorder across paths).
    a0_to_b_ms = delta_ns(a0, b)  # proxy → prefill (wall clock)
    a0_to_d0_ms = delta_perf(a0, d0)  # proxy: request start → decode HTTP start
    b_to_c_sched_ms = delta_perf(b, c_sched)  # same process; often ~0
    c_sched_to_c_ready_ms = delta_perf(c_sched, c_ready)
    c_ready_to_i0_ms = delta_perf(c_ready, i0)
    c_ready_to_i1_ms = delta_perf(c_ready, i1)
    h0_to_h1_ms = delta_ns(h0, h1)  # decode pull start → prefill recv handshake
    h1_to_i0_ms = delta_perf(h1, i0)  # handshake → wait-ready start (same process)
    m_to_n0_ms = delta_perf(m, n0)  # decode: promote/sched → first decode step
    n0_to_n1_ms = delta_ns(n0, n1)  # decode first token → proxy first token
    k1_to_l1_ms = delta_ns(k1, l1)  # prefill RDMA done → decode pull complete
    i1_to_j0_ms = delta_perf(i1, j0)  # wait-ready done → plan start
    j1_to_k0_ms = delta_perf(j1, k0)  # plan done → RDMA start

    fabric_fraction: float | None = None
    if ttft_ms and ttft_ms > 0 and t_transfer_ms is not None:
        fabric_fraction = t_transfer_ms / ttft_ms

    a0_to_b_ttft_fraction: float | None = None
    if ttft_ms and ttft_ms > 0 and a0_to_b_ms is not None:
        a0_to_b_ttft_fraction = a0_to_b_ms / ttft_ms

    plan_ms = delta_perf(j0, j1)
    plan_ttft_fraction: float | None = None
    if ttft_ms and ttft_ms > 0 and plan_ms is not None:
        plan_ttft_fraction = plan_ms / ttft_ms

    ttft_fabric_ttft_fraction: float | None = None
    if ttft_ms and ttft_ms > 0 and ttft_fabric_ms is not None:
        ttft_fabric_ttft_fraction = ttft_fabric_ms / ttft_ms

    promote_lag_ms = delta_ns(l1, m)
    promote_lag_ttft_fraction: float | None = None
    if ttft_ms and ttft_ms > 0 and promote_lag_ms is not None:
        promote_lag_ttft_fraction = promote_lag_ms / ttft_ms

    rdma_ms = delta_perf(k0, k1)
    kv_xfer_latency_us = rdma_ms * 1000.0 if rdma_ms is not None else None

    kv_bytes = k1.get("total_bytes") if k1 else None
    num_descriptors, bytes_per_descriptor = xfer_granularity_from_k1(k1)
    kv_xfer_bandwidth_gbps: float | None = None
    if kv_bytes and rdma_ms and rdma_ms > 0:
        kv_xfer_bandwidth_gbps = (kv_bytes * 8.0) / (rdma_ms * 1e6)

    itls = itl_samples(marks)
    num_prompt_tokens = None
    if c_sched_or_b and c_sched_or_b.get("num_prompt_tokens") is not None:
        num_prompt_tokens = int(c_sched_or_b["num_prompt_tokens"])
    elif a0 and a0.get("prompt_chars") is not None:
        # Rough fallback when token count unavailable (~4 chars/token).
        num_prompt_tokens = int(a0["prompt_chars"]) // 4

    return {
        "ttft_ms": ttft_ms,
        "t_prefill_ms": t_prefill_ms,
        "a0_to_b_ms": a0_to_b_ms,
        "a0_to_d0_ms": a0_to_d0_ms,
        "a0_to_b_ttft_fraction": a0_to_b_ttft_fraction,
        "plan_ttft_fraction": plan_ttft_fraction,
        "ttft_fabric_ttft_fraction": ttft_fabric_ttft_fraction,
        "promote_lag_ttft_fraction": promote_lag_ttft_fraction,
        "b_to_c_sched_ms": b_to_c_sched_ms,
        "c_sched_to_c_ready_ms": c_sched_to_c_ready_ms,
        "c_ready_to_i0_ms": c_ready_to_i0_ms,
        "c_ready_to_i1_ms": c_ready_to_i1_ms,
        "h0_to_h1_ms": h0_to_h1_ms,
        "h1_to_i0_ms": h1_to_i0_ms,
        "m_to_n0_ms": m_to_n0_ms,
        "n0_to_n1_ms": n0_to_n1_ms,
        "k1_to_l1_ms": k1_to_l1_ms,
        "i1_to_j0_ms": i1_to_j0_ms,
        "j1_to_k0_ms": j1_to_k0_ms,
        "t_transfer_ms": t_transfer_ms,
        "ttft_fabric_ms": ttft_fabric_ms,
        "t_decode_init_ms": t_decode_init_ms,
        "fabric_fraction": fabric_fraction,
        "kv_xfer_latency_us": kv_xfer_latency_us,
        "kv_xfer_bandwidth_gbps": kv_xfer_bandwidth_gbps,
        "fabric_fct_ms": t_transfer_ms,  # Flow completion H0→L1 (Section 4 Fabric_FCT)
        "itl_p50_ms": percentile(itls, 0.5) if itls else None,
        "itl_p95_ms": percentile(itls, 0.95) if itls else None,
        "itl_p99_ms": percentile(itls, 0.99) if itls else None,
        "itl_mean_ms": statistics.mean(itls) if itls else None,
        "itl_count": len(itls) if itls else 0,
        "wait_ready_ms": delta_perf(i0, i1),
        "plan_ms": plan_ms,
        "rdma_ms": rdma_ms,
        "promote_lag_ms": promote_lag_ms,
        "d_setup_ms": delta_ns(d0, h0),
        "bootstrap_ms": delta_perf(g0, g1),
        "ready_to_rdma_ms": delta_perf(i1, k0),
        "first_decode_sched_ms": delta_ns(m, n0),
        "pull_stall_ms": t_transfer_ms,
        "num_prompt_tokens": num_prompt_tokens,
        "kv_bytes": int(kv_bytes) if kv_bytes is not None else None,
        "num_descriptors": num_descriptors,
        "bytes_per_descriptor": bytes_per_descriptor,
    }


def summarize_us(vals: list[float | None], label: str) -> str:
    stats = summary_stats(vals)
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: n={len(clean_floats(vals))} "
        f"mean={stats['mean']:.1f} "
        f"p50={stats['p50']:.1f} "
        f"p95={stats['p95']:.1f} "
        f"p99={stats['p99']:.1f} us"
    )


def summarize(vals: list[float | None], label: str) -> str:
    stats = summary_stats(vals)
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: n={len(clean_floats(vals))} "
        f"mean={stats['mean']:.3f} "
        f"p50={stats['p50']:.3f} "
        f"p95={stats['p95']:.3f} "
        f"p99={stats['p99']:.3f} ms"
    )


def summarize_gbps(vals: list[float | None], label: str) -> str:
    stats = summary_stats(vals, include_p95=False)
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: n={len(clean_floats(vals))} "
        f"mean={stats['mean']:.3f} "
        f"p50={stats['p50']:.3f} "
        f"p99={stats['p99']:.3f} GB/s"
    )


def summarize_descriptors(vals: list[float | None], label: str) -> str:
    stats = summary_stats(vals)
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: n={len(clean_floats(vals))} "
        f"mean={stats['mean']:.1f} "
        f"p50={stats['p50']:.1f} "
        f"p95={stats['p95']:.1f} "
        f"p99={stats['p99']:.1f}"
    )


def summarize_bytes_per_descriptor(vals: list[float | None], label: str) -> str:
    stats = summary_stats(vals)
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: n={len(clean_floats(vals))} "
        f"mean={stats['mean']:.0f} "
        f"p50={stats['p50']:.0f} "
        f"p95={stats['p95']:.0f} "
        f"p99={stats['p99']:.0f} bytes/descriptor"
    )


def summarize_fraction(vals: list[float | None], label: str) -> str:
    stats = summary_stats(vals, include_p95=False)
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: mean={stats['mean']:.3f} "
        f"p50={stats['p50']:.3f} p99={stats['p99']:.3f}"
    )


def group_key(row: dict[str, Any]) -> str:
    n = row.get("num_prompt_tokens")
    return f"prompt_tokens={n}" if n is not None else "prompt_tokens=unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("/tmp/vllm_pd_profile"),
        help="Directory containing proxy/prefill/decode.jsonl",
    )
    parser.add_argument("--per-request", action="store_true")
    parser.add_argument("--csv", type=Path, help="Write per-request CSV")
    parser.add_argument("--json", type=Path, help="Write summary JSON")
    args = parser.parse_args()

    by_xfer = load_marks(args.dir)
    if not by_xfer:
        print(f"No marks found in {args.dir}")
        return

    rows: list[dict[str, Any]] = []
    for tid, marks in sorted(by_xfer.items()):
        metrics = compute_ietf_metrics(marks)
        metrics["transfer_id"] = tid
        rows.append(metrics)
        if args.per_request:
            parts = [tid]
            for k in IETF_METRIC_KEYS:
                v = metrics.get(k)
                if v is None:
                    parts.append(f"{k}=n/a")
                elif isinstance(v, float):
                    parts.append(f"{k}={v:.3f}")
                else:
                    parts.append(f"{k}={v}")
            print(" ".join(parts))

    print(f"\n=== IETF Test Category 2 metrics (n={len(rows)}, dir={args.dir}) ===")
    for key in IETF_METRIC_KEYS:
        if key.endswith("_gbps"):
            print(summarize_gbps([r.get(key) for r in rows if r.get(key)], key))
        elif key in ("itl_count", "num_prompt_tokens", "kv_bytes"):
            nums = [r.get(key) for r in rows if r.get(key) is not None]
            if nums:
                print(f"{key}: mean={statistics.mean(nums):.1f} n={len(nums)}")
            else:
                print(f"{key}: n/a")
        elif key in (
            "fabric_fraction",
            "a0_to_b_ttft_fraction",
            "plan_ttft_fraction",
            "ttft_fabric_ttft_fraction",
            "promote_lag_ttft_fraction",
        ):
            print(summarize_fraction([r.get(key) for r in rows], key))
        elif key == "kv_xfer_latency_us":
            print(
                summarize_us(
                    [r.get(key) for r in rows if r.get(key) is not None], key
                )
            )
        elif key == "num_descriptors":
            print(
                summarize_descriptors(
                    [r.get(key) for r in rows if r.get(key) is not None], key
                )
            )
        elif key == "bytes_per_descriptor":
            print(
                summarize_bytes_per_descriptor(
                    [r.get(key) for r in rows if r.get(key) is not None], key
                )
            )
        else:
            print(
                summarize(
                    [r.get(key) for r in rows if r.get(key) is not None], key
                )
            )

    # Group by prompt length (Section 6.1 reporting).
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[group_key(row)].append(row)

    if len(by_prompt) > 1 or (
        len(by_prompt) == 1 and "unknown" not in next(iter(by_prompt))
    ):
        print("\n=== By prompt length (Section 6.1) ===")
        for gk in sorted(by_prompt, key=lambda k: int(k.split("=")[-1]) if "=" in k else 0):
            grp = by_prompt[gk]
            print(f"\n{gk} (n={len(grp)})")
            print(summarize([r.get("ttft_ms") for r in grp if r.get("ttft_ms")], "ttft_ms"))
            for stage_key in (
                "a0_to_b_ms",
                "a0_to_d0_ms",
                "b_to_c_sched_ms",
                "c_sched_to_c_ready_ms",
                "c_ready_to_i0_ms",
                "c_ready_to_i1_ms",
                "h0_to_h1_ms",
                "h1_to_i0_ms",
                "m_to_n0_ms",
                "n0_to_n1_ms",
                "k1_to_l1_ms",
                "i1_to_j0_ms",
                "j1_to_k0_ms",
            ):
                print(
                    summarize(
                        [r.get(stage_key) for r in grp if r.get(stage_key) is not None],
                        stage_key,
                    )
                )
            fr_keys = (
                "a0_to_b_ttft_fraction",
                "plan_ttft_fraction",
                "ttft_fabric_ttft_fraction",
                "promote_lag_ttft_fraction",
            )
            for fr_key in fr_keys:
                print(summarize_fraction([r.get(fr_key) for r in grp], fr_key))
            print(
                summarize(
                    [r.get("t_transfer_ms") for r in grp if r.get("t_transfer_ms")],
                    "t_transfer_ms",
                )
            )
            print(
                summarize_gbps(
                    [r.get("kv_xfer_bandwidth_gbps") for r in grp if r.get("kv_xfer_bandwidth_gbps")],
                    "kv_xfer_bandwidth_gbps",
                )
            )

    if args.csv:
        fieldnames = ["transfer_id", *IETF_METRIC_KEYS, "d_setup_ms", "bootstrap_ms"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV: {args.csv}")

    if args.json:
        summary = {
            "n_requests": len(rows),
            "profile_dir": str(args.dir),
            "metrics": {},
            "by_prompt_length": {},
        }
        for key in IETF_METRIC_KEYS:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if not vals:
                continue
            if key in ("itl_count", "num_prompt_tokens", "kv_bytes"):
                summary["metrics"][key] = {"mean": statistics.mean(vals), "n": len(vals)}
                continue
            include_p95 = not (
                key.endswith("_gbps")
                or key
                in (
                    "fabric_fraction",
                    "a0_to_b_ttft_fraction",
                    "plan_ttft_fraction",
                    "ttft_fabric_ttft_fraction",
                    "promote_lag_ttft_fraction",
                )
            )
            stats = summary_stats(vals, include_p95=include_p95)
            if stats:
                summary["metrics"][key] = stats
        for gk, grp in by_prompt.items():
            ttfts = [r.get("ttft_ms") for r in grp]
            transfers = [r.get("t_transfer_ms") for r in grp]
            summary["by_prompt_length"][gk] = {
                "n": len(grp),
                "ttft_ms": summary_stats(ttfts) or None,
                "t_transfer_ms": summary_stats(transfers, include_p95=False) or None,
            }
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
