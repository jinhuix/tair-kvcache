#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import orjson

    def dumps(obj: dict) -> str:
        return orjson.dumps(obj).decode("utf-8")

    def loads(line: bytes) -> dict:
        return orjson.loads(line)

except ModuleNotFoundError:

    def dumps(obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":"))

    def loads(line: bytes) -> dict:
        return json.loads(line)


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE = SCRIPT_DIR / "oracle_checkpoint_belady.py"
RUN_ROOT = Path("/mnt/xujinhui/results/qwen_37_max/qwen37_branch_end_oracle_a_until99_20260630_h06_h12")
RAW_TRACE = Path(
    "/mnt/xujinhui/modelboard/qwen_37_max/20260630_h06_to_20260630_h12/"
    "qwen3.7-max_20260630_h06_to_20260630_h12_service_"
    "qwen3.7-max-2026-05-20-think-model-a186_block512.jsonl.zst"
)
TRACE = Path(
    "/mnt/xujinhui/results/qwen_37_max/"
    "qwen37_branch_end_checkpoint_value_score_until99_rerun_20260712/"
    "traces/qwen37_a186_block512_optimizer_request.jsonl"
)
INSTANCE_ID = "qwen37_a186"
BLOCK_SIZE = 512
GROUP_COUNT = 12
START_CAPACITY_TIB = 2.0
GROWTH_FACTOR = 2.0
TARGET_WINDOW_HIT_RATE = 0.7861
WARMUP_NS = int(2 * 3600 * 1e9)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def hash_to_i64(value: str) -> int:
    raw = bytes.fromhex(value)
    unsigned = int.from_bytes(raw[:8], "big", signed=False)
    if unsigned >= (1 << 63):
        unsigned -= 1 << 64
    return unsigned


def open_zst_lines(path: Path):
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


def build_optimizer_trace() -> None:
    meta_path = TRACE.with_suffix(".meta.json")
    if TRACE.exists() and meta_path.exists():
        print(f"[{now()}] reuse optimizer trace: {TRACE}", flush=True)
        return

    TRACE.parent.mkdir(parents=True, exist_ok=True)
    tmp_trace = TRACE.with_suffix(".jsonl.tmp")
    rows = 0
    written = 0
    skipped_no_hash = 0
    truncated_hashes = 0
    input_tokens = 0
    key_count = 0
    prev_ts = 0
    started = time.time()

    with tmp_trace.open("wb") as out:
        for line in open_zst_lines(RAW_TRACE):
            rows += 1
            obj = loads(line)
            hashes = obj.get("input_block_hash_ids") or []
            input_len = int(obj.get("input_length") or obj.get("token_length") or len(hashes) * BLOCK_SIZE)
            max_keys = max(0, input_len // BLOCK_SIZE)
            if len(hashes) > max_keys:
                truncated_hashes += len(hashes) - max_keys
                hashes = hashes[:max_keys]
            if not hashes:
                skipped_no_hash += 1
                continue

            ts = int(float(obj["timestamp"]) * 1_000_000_000)
            if ts <= prev_ts:
                ts = prev_ts + 1
            prev_ts = ts

            rec = {
                "type": "request",
                "instance_id": INSTANCE_ID,
                "trace_id": "trace_" + str(obj.get("request_id", written)),
                "timestamp_ns": ts,
                "keys": [hash_to_i64(str(value)) for value in hashes],
                "input_len": input_len,
                "query_type": "prefix_match",
                "block_mask": [],
                "sw_size": 0,
                "location_spec_names": [],
                "ttl_us": 0,
            }
            out.write((dumps(rec) + "\n").encode("utf-8"))
            written += 1
            input_tokens += input_len
            key_count += len(hashes)
            if written % 100000 == 0:
                print(
                    f"[{now()}] converted written={written} rows={rows} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    tmp_trace.rename(TRACE)
    meta = {
        "raw_trace": str(RAW_TRACE),
        "optimizer_trace": str(TRACE),
        "rows_scanned": rows,
        "requests_written": written,
        "skipped_no_hash": skipped_no_hash,
        "truncated_hashes": truncated_hashes,
        "input_tokens": input_tokens,
        "key_count": key_count,
        "block_size": BLOCK_SIZE,
        "instance_id": INSTANCE_ID,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[{now()}] wrote optimizer trace: {json.dumps(meta, sort_keys=True)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    parser.add_argument("--start-capacity-tib", type=float, default=START_CAPACITY_TIB)
    parser.add_argument("--growth-factor", type=float, default=GROWTH_FACTOR)
    parser.add_argument("--target-window-hit-rate", type=float, default=TARGET_WINDOW_HIT_RATE)
    parser.add_argument("--max-capacity-tib", type=float, default=0.0)
    parser.add_argument("--capacities-tib", nargs="*", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RAW_TRACE.exists():
        raise FileNotFoundError(RAW_TRACE)
    if not ORACLE.exists():
        raise FileNotFoundError(ORACLE)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_optimizer_trace()
    cmd = [
        sys.executable,
        str(ORACLE),
        "--trace",
        str(TRACE),
        "--output-dir",
        str(args.output_dir),
        "--block-size",
        str(BLOCK_SIZE),
        "--group-count",
        str(GROUP_COUNT),
        "--warmup-ns",
        str(WARMUP_NS),
        "--start-capacity-tib",
        str(args.start_capacity_tib),
        "--growth-factor",
        str(args.growth_factor),
        "--target-window-hit-rate",
        str(args.target_window_hit_rate),
    ]
    if args.max_capacity_tib > 0:
        cmd += ["--max-capacity-tib", str(args.max_capacity_tib)]
    if args.capacities_tib:
        cmd += ["--capacities-tib", *[str(v) for v in args.capacities_tib]]
    print(f"[{now()}] run oracle: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
