#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import subprocess
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import orjson

    def loads(line: bytes | str) -> dict:
        return orjson.loads(line)

except ModuleNotFoundError:

    def loads(line: bytes | str) -> dict:
        return json.loads(line)


MASK64 = (1 << 64) - 1
INF_NEXT_USE = 1 << 60


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def mix_u64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def key_to_u64(value: int) -> int:
    return value & MASK64


def prefix_signatures(keys: list[int]) -> array:
    sigs = array("Q")
    h = 1469598103934665603
    for key in keys:
        h ^= mix_u64(key_to_u64(key))
        h = (h * 1099511628211) & MASK64
        sigs.append(h)
    return sigs


def open_lines(path: Path) -> Iterable[bytes]:
    if path.suffix == ".zst":
        proc = subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stdout is None:
            raise RuntimeError("zstdcat did not provide stdout")
        try:
            for line in proc.stdout:
                if line.strip():
                    yield line
        finally:
            proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                raise RuntimeError(f"zstdcat failed rc={rc}: {stderr.strip()}")
        return
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                yield line


@dataclass
class RequestEntry:
    timestamp_ns: int
    input_tokens: int
    prefixes: array


@dataclass
class CheckpointRecord:
    signature: int
    prefix_blocks: int
    prefix_sigs: tuple[int, ...]
    next_use: int


class OracleCheckpointBelady:
    def __init__(self, requests: list[RequestEntry], group_count: int, block_size: int) -> None:
        self.requests = requests
        self.group_count = group_count
        self.block_size = block_size
        self.candidate_signatures = self._build_candidate_signatures()
        self.occurrences = self._build_occurrences()

    def _branch_end_checkpoint_lengths(self, prefixes: array, history: set[int]) -> list[int]:
        if not prefixes:
            return []
        lengths: list[int] = []
        for prefix_len in range(len(prefixes) - 1, 0, -1):
            if prefixes[prefix_len - 1] in history:
                lengths.append(prefix_len)
                break
        request_end = len(prefixes)
        if not lengths or lengths[-1] != request_end:
            lengths.append(request_end)
        return lengths

    def _build_candidate_signatures(self) -> set[int]:
        history: set[int] = set()
        candidates: set[int] = set()
        for idx, req in enumerate(self.requests):
            for prefix_len in self._branch_end_checkpoint_lengths(req.prefixes, history):
                candidates.add(req.prefixes[prefix_len - 1])
            if len(req.prefixes) > 1:
                history.update(req.prefixes[:-1])
            if idx and idx % 100000 == 0:
                print(
                    f"[{now()}] oracle candidate pass requests={idx} candidates={len(candidates)}",
                    flush=True,
                )
        return candidates

    def _build_occurrences(self) -> dict[int, list[int]]:
        occurrences: dict[int, list[int]] = defaultdict(list)
        candidates = self.candidate_signatures
        for idx, req in enumerate(self.requests):
            for sig in req.prefixes:
                if sig in candidates:
                    occurrences[sig].append(idx)
            if idx and idx % 100000 == 0:
                print(
                    f"[{now()}] oracle occurrence pass requests={idx} tracked={len(occurrences)}",
                    flush=True,
                )
        return dict(occurrences)

    def run_capacity(self, capacity_blocks: int, warmup_ns: int) -> dict:
        resident: dict[int, CheckpointRecord] = {}
        full_ref: dict[int, int] = {}
        occurrence_pos: dict[int, int] = defaultdict(int)
        victim_heap: list[tuple[int, int]] = []
        used_blocks = 0
        history: set[int] = set()

        first_ts = self.requests[0].timestamp_ns if self.requests else 0
        cutoff_ts = first_ts + warmup_ns
        full_input_tokens = 0
        full_hit_tokens = 0
        full_read_blocks = 0
        full_hit_blocks = 0
        window_input_tokens = 0
        window_hit_tokens = 0
        window_read_blocks = 0
        window_hit_blocks = 0
        admissions = 0
        rejected_admissions = 0
        evictions = 0
        max_used_blocks = 0

        def next_use(sig: int) -> int:
            occ = self.occurrences.get(sig)
            if not occ:
                return INF_NEXT_USE
            pos = occurrence_pos.get(sig, 0)
            return occ[pos] if pos < len(occ) else INF_NEXT_USE

        def push_victim(record: CheckpointRecord) -> None:
            heapq.heappush(victim_heap, (-record.next_use, record.signature))

        def evict_checkpoint(sig: int) -> bool:
            nonlocal used_blocks, evictions
            record = resident.pop(sig, None)
            if record is None:
                return False
            used_blocks -= self.group_count
            for prefix_sig in record.prefix_sigs:
                ref = full_ref[prefix_sig] - 1
                if ref == 0:
                    del full_ref[prefix_sig]
                    used_blocks -= 1
                else:
                    full_ref[prefix_sig] = ref
            evictions += 1
            return True

        def incremental_cost(records: list[CheckpointRecord]) -> int:
            missing_full: set[int] = set()
            for record in records:
                for prefix_sig in record.prefix_sigs:
                    if prefix_sig not in full_ref:
                        missing_full.add(prefix_sig)
            return len(missing_full) + self.group_count * len(records)

        def select_victim(pinned: set[int]) -> int | None:
            while victim_heap:
                neg_next, sig = heapq.heappop(victim_heap)
                record = resident.get(sig)
                if record is None or -neg_next != record.next_use:
                    continue
                if sig in pinned:
                    continue
                return sig
            return None

        for idx, req in enumerate(self.requests):
            # Move all candidate prefixes of the current request past this read.
            for sig in req.prefixes:
                if sig not in self.candidate_signatures:
                    continue
                occ = self.occurrences.get(sig)
                pos = occurrence_pos.get(sig, 0)
                while occ is not None and pos < len(occ) and occ[pos] <= idx:
                    pos += 1
                occurrence_pos[sig] = pos
                record = resident.get(sig)
                if record is not None:
                    record.next_use = next_use(sig)
                    push_victim(record)

            hit_blocks = 0
            for prefix_len in range(len(req.prefixes), 0, -1):
                if req.prefixes[prefix_len - 1] in resident:
                    hit_blocks = prefix_len
                    break
            hit_tokens = min(req.input_tokens, hit_blocks * self.block_size)

            full_input_tokens += req.input_tokens
            full_hit_tokens += hit_tokens
            full_read_blocks += len(req.prefixes)
            full_hit_blocks += hit_blocks
            if req.timestamp_ns >= cutoff_ts:
                window_input_tokens += req.input_tokens
                window_hit_tokens += hit_tokens
                window_read_blocks += len(req.prefixes)
                window_hit_blocks += hit_blocks

            checkpoint_lengths = self._branch_end_checkpoint_lengths(req.prefixes, history)
            new_records: list[CheckpointRecord] = []
            seen_new: set[int] = set()
            for prefix_len in checkpoint_lengths:
                sig = req.prefixes[prefix_len - 1]
                if sig in resident or sig in seen_new:
                    continue
                seen_new.add(sig)
                new_records.append(
                    CheckpointRecord(
                        signature=sig,
                        prefix_blocks=prefix_len,
                        prefix_sigs=tuple(req.prefixes[:prefix_len]),
                        next_use=next_use(sig),
                    )
                )

            if new_records:
                pinned_new = {record.signature for record in new_records}
                while used_blocks + incremental_cost(new_records) > capacity_blocks:
                    victim = select_victim(pinned_new)
                    if victim is None:
                        break
                    evict_checkpoint(victim)

                if used_blocks + incremental_cost(new_records) <= capacity_blocks:
                    for record in new_records:
                        for prefix_sig in record.prefix_sigs:
                            ref = full_ref.get(prefix_sig, 0)
                            if ref == 0:
                                used_blocks += 1
                            full_ref[prefix_sig] = ref + 1
                        used_blocks += self.group_count
                        resident[record.signature] = record
                        push_victim(record)
                    admissions += len(new_records)
                else:
                    rejected_admissions += len(new_records)

            if len(req.prefixes) > 1:
                history.update(req.prefixes[:-1])
            max_used_blocks = max(max_used_blocks, used_blocks)
            if idx and idx % 100000 == 0:
                print(
                    f"[{now()}] oracle sim cap={capacity_blocks} requests={idx} "
                    f"window_hit_rate={(window_hit_tokens / window_input_tokens) if window_input_tokens else 0:.9f} "
                    f"resident={len(resident)} used={used_blocks}",
                    flush=True,
                )

        return {
            "capacity_blocks": capacity_blocks,
            "window_input_tokens": window_input_tokens,
            "window_hit_tokens": window_hit_tokens,
            "window_hit_rate": window_hit_tokens / window_input_tokens if window_input_tokens else 0.0,
            "window_read_blocks": window_read_blocks,
            "window_hit_blocks": window_hit_blocks,
            "window_block_hit_rate": window_hit_blocks / window_read_blocks if window_read_blocks else 0.0,
            "full_input_tokens": full_input_tokens,
            "full_hit_tokens": full_hit_tokens,
            "full_hit_rate": full_hit_tokens / full_input_tokens if full_input_tokens else 0.0,
            "full_read_blocks": full_read_blocks,
            "full_hit_blocks": full_hit_blocks,
            "full_block_hit_rate": full_hit_blocks / full_read_blocks if full_read_blocks else 0.0,
            "max_used_blocks": max_used_blocks,
            "resident_checkpoints": len(resident),
            "stored_full_blocks": len(full_ref),
            "admitted_checkpoints": admissions,
            "rejected_checkpoints": rejected_admissions,
            "evicted_checkpoints": evictions,
        }


def capacity_blocks_from_tib(capacity_tib: float) -> int:
    return int(round(capacity_tib * 1024 * 1024 / 16))


def capacity_label(capacity_tib: float, capacity_blocks: int) -> str:
    if abs(capacity_tib - round(capacity_tib)) < 1e-9:
        cap = f"{int(round(capacity_tib))}tib"
    else:
        cap = f"{capacity_tib:.6f}".rstrip("0").rstrip(".").replace(".", "p") + "tib"
    return f"phys{cap}_blocks_{capacity_blocks}"


def load_requests(trace_path: Path) -> list[RequestEntry]:
    requests: list[RequestEntry] = []
    started = time.time()
    for idx, line in enumerate(open_lines(trace_path)):
        obj = loads(line)
        if obj.get("type", "request") != "request":
            continue
        keys = obj.get("keys") or []
        if not keys:
            continue
        requests.append(
            RequestEntry(
                timestamp_ns=int(obj["timestamp_ns"]),
                input_tokens=int(obj.get("input_len") or len(keys)),
                prefixes=prefix_signatures([int(key) for key in keys]),
            )
        )
        if len(requests) % 100000 == 0:
            print(
                f"[{now()}] loaded requests={len(requests)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    if not requests:
        raise RuntimeError(f"no request rows loaded from {trace_path}")
    return requests


def append_summary(summary_path: Path, row: dict) -> None:
    fields = [
        "capacity_tib",
        "capacity_blocks",
        "target_window_hit_rate",
        "reached_target",
        "window_hit_rate",
        "window_block_hit_rate",
        "full_hit_rate",
        "full_block_hit_rate",
        "window_input_tokens",
        "window_hit_tokens",
        "window_read_blocks",
        "window_hit_blocks",
        "max_used_blocks",
        "resident_checkpoints",
        "stored_full_blocks",
        "admitted_checkpoints",
        "rejected_checkpoints",
        "evicted_checkpoints",
    ]
    exists = summary_path.exists()
    with summary_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-A Belady upper bound for branch_end Mamba checkpoints")
    parser.add_argument("--trace", type=Path, required=True, help="optimizer request JSONL/JSONL.zst trace")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--group-count", type=int, default=12)
    parser.add_argument("--warmup-ns", type=int, default=int(2 * 3600 * 1e9))
    parser.add_argument("--start-capacity-tib", type=float, default=2.0)
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--target-window-hit-rate", type=float, default=0.7861)
    parser.add_argument("--max-capacity-tib", type=float, default=0.0, help="0 means no explicit cap")
    parser.add_argument("--capacities-tib", nargs="*", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "trace": str(args.trace),
        "block_size": args.block_size,
        "group_count": args.group_count,
        "warmup_ns": args.warmup_ns,
        "start_capacity_tib": args.start_capacity_tib,
        "growth_factor": args.growth_factor,
        "target_window_hit_rate": args.target_window_hit_rate,
        "method": "oracle_a_branch_end_checkpoint_belady_next_hit_time",
    }
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    print(f"[{now()}] loading trace {args.trace}", flush=True)
    requests = load_requests(args.trace)
    print(f"[{now()}] loaded {len(requests)} requests", flush=True)
    oracle = OracleCheckpointBelady(requests, group_count=args.group_count, block_size=args.block_size)
    print(
        f"[{now()}] prepared oracle candidates={len(oracle.candidate_signatures)} "
        f"occurrence_keys={len(oracle.occurrences)}",
        flush=True,
    )

    summary_path = args.output_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()
    status_path = args.output_dir / "status.json"
    if status_path.exists():
        status_path.unlink()
    completed: list[dict] = []
    if args.capacities_tib:
        capacities = list(args.capacities_tib)
    else:
        capacities = None

    capacity_tib = args.start_capacity_tib
    run_count = 0
    while True:
        if capacities is not None:
            if run_count >= len(capacities):
                break
            capacity_tib = capacities[run_count]
        capacity_blocks = capacity_blocks_from_tib(capacity_tib)
        print(f"[{now()}] start oracle capacity={capacity_tib}T blocks={capacity_blocks}", flush=True)
        metrics = oracle.run_capacity(capacity_blocks, args.warmup_ns)
        reached = metrics["window_hit_rate"] >= args.target_window_hit_rate
        row = {
            "capacity_tib": capacity_tib,
            "capacity_blocks": capacity_blocks,
            "target_window_hit_rate": args.target_window_hit_rate,
            "reached_target": reached,
            **metrics,
        }
        append_summary(summary_path, row)
        completed.append(row)
        status_path.write_text(
            json.dumps(
                {
                    "state": "done" if reached else "running",
                    "current_capacity_tib": capacity_tib,
                    "completed": completed,
                    "updated_at": now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"[{now()}] done oracle capacity={capacity_tib}T "
            f"window={metrics['window_hit_rate']:.9f} reached={reached}",
            flush=True,
        )
        if reached and not args.capacities_tib:
            break
        run_count += 1
        if capacities is None:
            if args.max_capacity_tib > 0 and capacity_tib >= args.max_capacity_tib:
                break
            if run_count >= 64:
                raise RuntimeError("refusing to run more than 64 growth steps without reaching target")
            capacity_tib *= args.growth_factor

    print(f"[{now()}] oracle summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
