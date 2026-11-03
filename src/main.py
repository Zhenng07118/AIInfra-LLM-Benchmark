#!/usr/bin/env python3
"""Learning-grade vLLM serving benchmark (no third-party dependencies).

Reconstructed and improved from a 2026 hands-on learning session; not an
exact copy of historical scripts. Client-visible timing; not GPU kernel timing.
"""
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "请介绍人工智能的主要应用领域。"
MAX_TOKENS = 256
REPEATS = 5
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
TIMEOUT_SECONDS = 120
COLUMNS = ["concurrency", "throughput", "throughput_std", "avg_latency",
           "p95_latency", "avg_ttft_ms", "avg_tpot_ms", "failed_requests",
           "successful_requests", "repeats"]


def send_request(i):
    """Send one SSE streamed request; return timings and token count."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"用户{i}：{PROMPT}"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    first_text_at = None
    last_text_at = None
    output_tokens = None
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            for raw_line in response:
                if not raw_line.startswith(b"data: "):
                    continue
                text = raw_line[6:].decode("utf-8").strip()
                if text == "[DONE]":
                    break
                if not text:
                    continue
                chunk = json.loads(text)
                for choice in chunk.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        now = time.perf_counter()
                        if first_text_at is None:
                            first_text_at = now
                        last_text_at = now
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
        http_elapsed = time.perf_counter() - start
        if output_tokens is None:
            raise ValueError("stream response missing usage.completion_tokens")
        if output_tokens < 1 or first_text_at is None or last_text_at is None:
            raise ValueError("no nonempty streamed text content received")
        # The observable completion latency is last text chunk arrival, not
        # necessarily HTTP connection close. Keep HTTP elapsed for debugging.
        text_latency = last_text_at - start
        ttft = first_text_at - start
        tpot = ((last_text_at - first_text_at) / (output_tokens - 1)
                if output_tokens > 1 else None)
        return {"ok": True, "tokens": output_tokens, "latency": text_latency,
                "http_elapsed": http_elapsed, "ttft": ttft, "tpot": tpot,
                "error": None}
    except (urllib.error.URLError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as exc:
        return {"ok": False, "tokens": 0,
                "latency": time.perf_counter() - start, "http_elapsed": None,
                "ttft": None, "tpot": None, "error": str(exc)}


def nearest_rank_p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def run_benchmark(concurrency, repeats=REPEATS):
    throughputs, latencies, ttfts, tpots = [], [], [], []
    failed, successful_count = 0, 0
    for repetition in range(repeats):
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(send_request, range(concurrency)))
        batch_elapsed = time.perf_counter() - batch_start
        ok = [r for r in results if r["ok"]]
        bad = [r for r in results if not r["ok"]]
        failed += len(bad)
        successful_count += len(ok)
        if bad:
            print(f"  并发{concurrency} 第{repetition + 1}轮失败{len(bad)}次；示例错误：{bad[0]['error']}")
        if ok:
            throughputs.append(sum(r["tokens"] for r in ok) / batch_elapsed)
            latencies.extend(r["latency"] for r in ok)
            ttfts.extend(r["ttft"] for r in ok if r["ttft"] is not None)
            tpots.extend(r["tpot"] for r in ok if r["tpot"] is not None)
    if not throughputs:
        raise RuntimeError("所有轮次均无成功请求，请先检查 vLLM 服务端与 /health")
    return {
        "concurrency": concurrency,
        "throughput": round(statistics.mean(throughputs), 2),
        "throughput_std": (round(statistics.stdev(throughputs), 2)
                           if len(throughputs) >= 2 else None),
        "avg_latency": round(statistics.mean(latencies), 3),
        "p95_latency": round(nearest_rank_p95(latencies), 3),
        "avg_ttft_ms": (round(statistics.mean(ttfts) * 1000, 2)
                        if ttfts else None),
        "avg_tpot_ms": (round(statistics.mean(tpots) * 1000, 2)
                        if tpots else None),
        "failed_requests": failed,
        "successful_requests": successful_count,
        "repeats": repeats,
    }


def save_results(rows, levels):
    folder = Path("results")
    folder.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"benchmark_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "recorded_at_local": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL, "url": URL, "prompt": PROMPT,
        "max_tokens": MAX_TOKENS, "temperature": 0,
        "stream": True, "include_usage": True,
        "concurrency_levels": levels, "repeats": REPEATS,
        "measurement_notes": [
            "TTFT = client request start to first nonempty streamed text chunk",
            "Latency = client request start to last nonempty streamed text chunk",
            "TPOT = (last text - first text) / (output tokens - 1)",
            "Throughput = sum successful output tokens / batch wall-clock time",
            "Not GPU kernel metrics; prefix caching and client overhead not controlled",
        ],
        "server_config": {
            "note": "Fill from actual vllm serve logs; unverified here",
            "vllm_version": None, "gpu": None, "driver": None,
            "max_num_seqs": None, "gpu_memory_utilization": None,
        },
    }
    config_path = path.with_name(path.stem + "_config.json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"结果已保存：{path}")
    print(f"实验配置已保存：{config_path}")
    return path


def main():
    print("=== LLM Inference Benchmark ===")
    print("1. 单请求测试")
    print("2. 并发测试")
    choice = input("请选择测试模式（1/2）：").strip()
    if choice == "1":
        levels = [1]
    elif choice == "2":
        levels = CONCURRENCY_LEVELS
    else:
        raise ValueError("请输入 1 或 2")
    # Warm-up; still inspect status to avoid hiding unavailable server.
    print("预热中...")
    warmup = send_request(0)
    if not warmup["ok"]:
        raise RuntimeError(f"预热失败：{warmup['error']}")
    rows = []
    for level in levels:
        row = run_benchmark(level)
        rows.append(row)
        print(
            f"并发={level} | 吞吐量={row['throughput']}±{row['throughput_std']} "
            f"Tokens/s | 平均延迟={row['avg_latency']}s | "
            f"P95={row['p95_latency']}s | TTFT={row['avg_ttft_ms']}ms | "
            f"TPOT={row['avg_tpot_ms']}ms | 失败={row['failed_requests']}"
        )
    save_results(rows, levels)


if __name__ == "__main__":
    main()
