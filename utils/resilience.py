"""
Resilience utilities — per-node timeout + retry with exponential backoff.

Used to wrap agent calls (fact_check, media_check) so a slow/hanging external
API (search, model inference, detection API) can't stall the whole graph, and
transient failures get retried before being marked as a permanent failure.

IMPORTANT implementation note: this uses raw daemon threading.Thread, NOT
concurrent.futures.ThreadPoolExecutor. ThreadPoolExecutor threads are
non-daemon by default and Python registers an atexit hook that joins all of
them before the process can exit — so a genuinely hung call (e.g. a network
request that never returns) would prevent the whole program from exiting even
after this wrapper "gives up" and returns a timeout result. Daemon threads are
killed automatically when the main program exits, avoiding that trap. The
trade-off: the hung call's thread is still technically alive and consuming
resources until it errors out or the process exits — Python cannot forcibly
kill a thread — so a leaked hung API call is still a leaked hung API call.
This decorator prevents it from blocking the graph or the process, not from
existing at all.
"""
import time
import queue
import threading
import functools


def with_timeout_retry(timeout_sec: float = 20.0, max_retries: int = 2, backoff_base: float = 1.5):
    """Decorator: runs `func` in a daemon worker thread with a hard timeout,
    retrying on failure/timeout with exponential backoff. If all attempts are
    exhausted, returns a degraded result dict (status: 'timeout' or 'failed')
    instead of raising, so the graph can continue with partial data."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wrapper_start = time.time()
            last_exception = None

            for attempt in range(max_retries + 1):
                result_q = queue.Queue(maxsize=1)

                def _run():
                    try:
                        result_q.put(("ok", func(*args, **kwargs)))
                    except Exception as e:
                        result_q.put(("error", e))

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                t.join(timeout=timeout_sec)

                if t.is_alive():
                    # Timed out — thread keeps running in the background (daemon,
                    # so it won't block process exit), but we stop waiting on it.
                    last_exception = TimeoutError(f"{func.__name__} exceeded {timeout_sec}s timeout")
                else:
                    status, payload = result_q.get()
                    if status == "ok":
                        return payload
                    last_exception = payload

                if attempt < max_retries:
                    time.sleep(backoff_base ** attempt)

            is_timeout = isinstance(last_exception, TimeoutError)
            elapsed_ms = int((time.time() - wrapper_start) * 1000)
            return {
                "verdict": "Unverified",
                "visual_verdict": "Inconclusive",
                "audio_verdict": "Inconclusive",
                "confidence": 0.0,
                "visual_confidence": 0.0,
                "audio_confidence": 0.0,
                "evidence": [f"[node failed after {max_retries + 1} attempts: {last_exception}]"],
                "visual_evidence": [f"[node failed after {max_retries + 1} attempts: {last_exception}]"],
                "audio_evidence": [],
                "status": "timeout" if is_timeout else "failed",
                "latency_ms": elapsed_ms,
                "cost_usd": 0.0,
            }
        return wrapper
    return decorator
