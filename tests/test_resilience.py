"""Tests for utils/resilience.py — including a regression test for the real bug
found during manual testing: the original ThreadPoolExecutor-based implementation
used non-daemon threads, which would prevent the whole process from exiting if
a wrapped call genuinely hung. Fixed with raw daemon threading.Thread."""
import time
import threading
from utils.resilience import with_timeout_retry


def test_successful_call_returns_immediately():
    @with_timeout_retry(timeout_sec=2.0, max_retries=1)
    def fast_func():
        return {"verdict": "ok"}

    result = fast_func()
    assert result == {"verdict": "ok"}


def test_timeout_returns_degraded_result_not_raises():
    @with_timeout_retry(timeout_sec=0.5, max_retries=1, backoff_base=1.0)
    def slow_func():
        time.sleep(999)

    result = slow_func()  # must not raise, must not hang the test
    assert result["status"] == "timeout"
    assert result["confidence"] == 0.0


def test_timeout_uses_daemon_thread_process_can_exit():
    """Regression test for the ThreadPoolExecutor non-daemon-thread bug.
    If this were still using non-daemon threads, the CI job itself would hang
    at teardown — the fact that the overall test run completes is part of the
    proof. This assertion additionally confirms the leaked thread is daemon."""
    @with_timeout_retry(timeout_sec=0.3, max_retries=0, backoff_base=1.0)
    def slow_func():
        time.sleep(999)

    result = slow_func()
    assert any(t.daemon for t in threading.enumerate() if t.is_alive())
    assert result["status"] == "timeout"


def test_retries_before_giving_up():
    attempt_count = {"n": 0}

    @with_timeout_retry(timeout_sec=2.0, max_retries=2, backoff_base=1.0)
    def flaky_func():
        attempt_count["n"] += 1
        raise ValueError("simulated transient failure")

    result = flaky_func()
    assert attempt_count["n"] == 3  # initial attempt + 2 retries
    assert result["status"] == "failed"


def test_eventual_success_after_retry():
    attempt_count = {"n": 0}

    @with_timeout_retry(timeout_sec=2.0, max_retries=2, backoff_base=1.0)
    def flaky_then_ok():
        attempt_count["n"] += 1
        if attempt_count["n"] < 2:
            raise ValueError("fails once")
        return {"verdict": "recovered"}

    result = flaky_then_ok()
    assert result == {"verdict": "recovered"}
