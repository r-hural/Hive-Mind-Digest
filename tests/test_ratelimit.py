from pipeline.llm.ratelimit import RateLimiter


class Clock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += s


def test_rpm_blocks_the_n_plus_first_request():
    c = Clock()
    rl = RateLimiter(rpm=2, tpm=0, clock=c.now, sleep=c.sleep)
    assert rl.acquire() == 0
    assert rl.acquire() == 0
    waited = rl.acquire()
    assert waited == 60.0
    assert c.t == 60.0


def test_tpm_blocks_on_estimated_tokens():
    c = Clock()
    rl = RateLimiter(rpm=0, tpm=100, clock=c.now, sleep=c.sleep)
    assert rl.acquire(60) == 0
    assert rl.acquire(60) == 60.0        # 120 > 100 within the window


def test_single_oversized_request_still_passes_on_empty_window():
    c = Clock()
    rl = RateLimiter(rpm=0, tpm=10, clock=c.now, sleep=c.sleep)
    assert rl.acquire(1000) == 0


def test_zero_means_unlimited():
    c = Clock()
    rl = RateLimiter(rpm=0, tpm=0, clock=c.now, sleep=c.sleep)
    for _ in range(100):
        assert rl.acquire(10_000) == 0


def test_penalize_sets_a_shared_barrier():
    c = Clock()
    rl = RateLimiter(rpm=0, tpm=0, clock=c.now, sleep=c.sleep)
    rl.penalize(30)
    assert rl.acquire() == 30.0
    assert rl.acquire() == 0


def test_lock_is_released_while_sleeping():
    c = Clock()
    locked_states = []

    def sleep(s):
        locked_states.append(rl._lock.locked())
        c.sleep(s)

    rl = RateLimiter(rpm=1, tpm=0, clock=c.now, sleep=sleep)
    rl.acquire()
    rl.acquire()          # rpm=1 forces a wait here, invoking the sleep closure above

    assert locked_states
    assert all(state is False for state in locked_states)


def test_wait_accumulates_across_loops():
    c = Clock()
    rl = RateLimiter(rpm=0, tpm=100, clock=c.now, sleep=c.sleep)
    assert rl.acquire(10) == 0
    c.t = 1
    assert rl.acquire(80) == 0
    waited = rl.acquire(50)              # waits 59s then 1s across two loop iterations
    assert waited == 60.0


def test_window_pruning_restores_capacity():
    c = Clock()
    rl = RateLimiter(rpm=2, tpm=0, clock=c.now, sleep=c.sleep)
    assert rl.acquire() == 0
    assert rl.acquire() == 0
    c.t = 61
    assert rl.acquire() == 0.0


def test_a_rate_limit_wait_stops_at_once_on_interrupt():
    # the default sleep waits on the interrupt flag: claude_cli's 6 calls/minute never holds Ctrl+C for 60 s
    import threading
    import time
    import pytest
    from pipeline.llm.base import LLMInterrupted, request_interrupt
    rl = RateLimiter(rpm=1)
    rl.acquire()
    threading.Timer(0.3, request_interrupt).start()
    t0 = time.monotonic()
    with pytest.raises(LLMInterrupted):
        rl.acquire()
    assert time.monotonic() - t0 < 5
