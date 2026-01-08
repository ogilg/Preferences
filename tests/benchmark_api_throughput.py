"""
Benchmark API throughput for Cerebras and Hyperbolic.

Run with: python -m tests.benchmark_api_throughput

Measures requests/second, latency distribution, and error rates
across different concurrency levels to find optimal max_concurrent.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

from dotenv import load_dotenv
import numpy as np

load_dotenv()
from openai import AsyncOpenAI


@dataclass
class BenchmarkResult:
    provider: str
    max_concurrent: int
    num_requests: int
    total_time: float
    latencies: list[float]
    errors: int
    rate_limit_errors: int

    @property
    def throughput(self) -> float:
        return self.num_requests / self.total_time

    @property
    def success_rate(self) -> float:
        return (self.num_requests - self.errors) / self.num_requests

    @property
    def latency_mean(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def latency_p50(self) -> float:
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0

    @property
    def latency_p95(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    @property
    def latency_p99(self) -> float:
        return float(np.percentile(self.latencies, 99)) if self.latencies else 0.0


PROVIDERS = {
    "hyperbolic": {
        "base_url": "https://api.hyperbolic.xyz/v1",
        "env_var": "HYPERBOLIC_API_KEY",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_var": "CEREBRAS_API_KEY",
        "model": "llama3.1-8b",
    },
}

TEST_PROMPT = [{"role": "user", "content": "Say 'hello' and nothing else."}]


async def run_benchmark(
    provider: str,
    max_concurrent: int,
    num_requests: int,
    timeout: float = 30.0,
) -> BenchmarkResult:
    import os

    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env_var"])
    if not api_key:
        raise ValueError(f"Missing {config['env_var']} environment variable")

    client = AsyncOpenAI(api_key=api_key, base_url=config["base_url"])
    semaphore = asyncio.Semaphore(max_concurrent)

    latencies: list[float] = []
    errors = 0
    rate_limit_errors = 0
    lock = asyncio.Lock()

    async def make_request() -> None:
        nonlocal errors, rate_limit_errors
        start = time.perf_counter()
        async with semaphore:
            try:
                await asyncio.wait_for(
                    client.chat.completions.create(
                        model=config["model"],
                        messages=TEST_PROMPT,
                        temperature=0.0,
                        max_tokens=8,
                    ),
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - start
                async with lock:
                    latencies.append(elapsed)
            except asyncio.TimeoutError:
                async with lock:
                    errors += 1
            except Exception as e:
                async with lock:
                    errors += 1
                    if "rate" in str(e).lower() or "429" in str(e):
                        rate_limit_errors += 1

    start_time = time.perf_counter()
    await asyncio.gather(*[make_request() for _ in range(num_requests)])
    total_time = time.perf_counter() - start_time

    await client.close()

    return BenchmarkResult(
        provider=provider,
        max_concurrent=max_concurrent,
        num_requests=num_requests,
        total_time=total_time,
        latencies=latencies,
        errors=errors,
        rate_limit_errors=rate_limit_errors,
    )


def print_result(result: BenchmarkResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"Provider: {result.provider} | Concurrency: {result.max_concurrent}")
    print(f"{'=' * 60}")
    print(f"Requests:      {result.num_requests}")
    print(f"Total time:    {result.total_time:.2f}s")
    print(f"Throughput:    {result.throughput:.2f} req/s")
    print(f"Success rate:  {result.success_rate * 100:.1f}%")
    print(f"Rate limits:   {result.rate_limit_errors}")
    print(f"Latency mean:  {result.latency_mean * 1000:.0f}ms")
    print(f"Latency p50:   {result.latency_p50 * 1000:.0f}ms")
    print(f"Latency p95:   {result.latency_p95 * 1000:.0f}ms")
    print(f"Latency p99:   {result.latency_p99 * 1000:.0f}ms")


def print_summary(results: list[BenchmarkResult]) -> None:
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Provider':<12} {'Concur':>8} {'Throughput':>12} {'Success':>10} "
        f"{'RateLim':>8} {'p50':>8} {'p95':>8}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.provider:<12} {r.max_concurrent:>8} {r.throughput:>10.2f}/s "
            f"{r.success_rate * 100:>9.1f}% {r.rate_limit_errors:>8} "
            f"{r.latency_p50 * 1000:>6.0f}ms {r.latency_p95 * 1000:>6.0f}ms"
        )

    # Find optimal concurrency for each provider
    print(f"\n{'=' * 80}")
    print("OPTIMAL CONCURRENCY (highest throughput with <5% errors)")
    print(f"{'=' * 80}")

    for provider in PROVIDERS:
        provider_results = [r for r in results if r.provider == provider]
        valid = [r for r in provider_results if r.success_rate >= 0.95]
        if valid:
            best = max(valid, key=lambda r: r.throughput)
            print(
                f"{provider}: max_concurrent={best.max_concurrent} "
                f"({best.throughput:.1f} req/s)"
            )
        else:
            print(f"{provider}: No valid results (all had >5% errors)")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark API throughput")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=list(PROVIDERS.keys()),
        default=list(PROVIDERS.keys()),
        help="Providers to test",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[10, 25, 50, 75, 100, 150, 200],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests per test",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per request in seconds",
    )
    args = parser.parse_args()

    results: list[BenchmarkResult] = []

    for provider in args.providers:
        print(f"\n>>> Testing {provider}...")
        for concurrency in args.concurrency:
            print(f"  Running with max_concurrent={concurrency}...", end=" ", flush=True)
            try:
                result = await run_benchmark(
                    provider=provider,
                    max_concurrent=concurrency,
                    num_requests=args.requests,
                    timeout=args.timeout,
                )
                results.append(result)
                print(f"{result.throughput:.1f} req/s, {result.errors} errors")

                # Early stop if we're hitting too many rate limits
                if result.rate_limit_errors > args.requests * 0.2:
                    print(f"  Stopping {provider}: too many rate limit errors")
                    break
            except Exception as e:
                print(f"Failed: {e}")

    for result in results:
        print_result(result)

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
