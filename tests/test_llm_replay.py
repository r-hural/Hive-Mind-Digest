import json, os, threading, time
import pytest
from pipeline.llm.base import LLMError, LLMResult
from pipeline.llm.replay import ReplayProvider, CassetteMissing, cassette_key


class Fake:
    name = "anthropic_api"
    def __init__(self, delay=0.0):
        self.calls = 0
        self.delay = delay
    def complete(self, prompt, *, model, system=None, max_tokens=16000):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return LLMResult(text='{"x": 1}', model=model, provider=self.name, input_tokens=3, output_tokens=2,
                         cost_usd=None, stop_reason="end_turn", duration_ms=5)


def test_key_depends_on_model_system_and_prompt():
    k = cassette_key("m", "s", "p")
    assert k != cassette_key("m2", "s", "p") and k != cassette_key("m", None, "p") and k != cassette_key("m", "s", "p2")
    assert len(k) == 64


def test_record_then_replay(tmp_path):
    inner = Fake()
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    r1 = rec.complete("p", model="m", system="s")
    assert inner.calls == 1 and rec.misses == 1 and rec.recorded == 1
    files = os.listdir(tmp_path / "anthropic_api")
    assert len(files) == 1 and files[0].endswith(".json")
    saved = json.load(open(tmp_path / "anthropic_api" / files[0], encoding="utf-8"))
    assert saved["prompt"] == "p" and saved["response"]["text"] == '{"x": 1}'
    assert saved["cassette_version"] == 1

    rep = ReplayProvider(str(tmp_path / "anthropic_api"))
    r2 = rep.complete("p", model="m", system="s")
    assert r2.text == r1.text and rep.hits == 1
    assert r2.provider == "replay:anthropic_api" and r2.cost_usd == 0.0
    assert r2.input_tokens == 3 and r2.duration_ms == 0
    assert rep.name == "anthropic_api"


def test_replay_without_cassette_raises(tmp_path):
    rep = ReplayProvider(str(tmp_path / "claude_cli"))
    with pytest.raises(CassetteMissing, match="claude_cli"):
        rep.complete("p", model="m")
    assert rep.misses == 1


def test_record_reuses_existing_cassette(tmp_path):
    inner = Fake()
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    rec.complete("p", model="m"); rec.complete("p", model="m")
    assert inner.calls == 1 and rec.hits == 1 and rec.misses == 1 and rec.recorded == 1


def test_record_requires_inner(tmp_path):
    with pytest.raises(ValueError):
        ReplayProvider(str(tmp_path / "anthropic_api"), record=True)


def test_corrupt_cassette_raises_llm_error(tmp_path):
    d = tmp_path / "anthropic_api"
    d.mkdir()
    key = cassette_key("m", None, "p")
    (d / f"{key}.json").write_text("{", encoding="utf-8")
    rep = ReplayProvider(str(d))
    with pytest.raises(LLMError, match="re-record"):
        rep.complete("p", model="m")


def test_cassette_with_extra_field_still_replays(tmp_path):
    d = tmp_path / "anthropic_api"
    d.mkdir()
    key = cassette_key("m", None, "p")
    payload = {
        "cassette_version": 1,
        "model": "m",
        "system": None,
        "prompt": "p",
        "response": {
            "text": '{"x": 1}', "model": "m", "provider": "anthropic_api",
            "input_tokens": 3, "output_tokens": 2, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "cost_usd": None, "stop_reason": "end_turn", "duration_ms": 5,
            "future_field": "unexpected",
        },
    }
    (d / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    rep = ReplayProvider(str(d))
    r = rep.complete("p", model="m")
    assert r.text == '{"x": 1}'
    assert r.duration_ms == 0
    assert r.cost_usd == 0.0


def test_concurrent_recording_writes_exactly_once(tmp_path):
    inner = Fake(delay=0.05)
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    results = []

    def worker():
        results.append(rec.complete("p", model="m"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    files = os.listdir(tmp_path / "anthropic_api")
    assert len(files) == 1
    with open(tmp_path / "anthropic_api" / files[0], encoding="utf-8") as f:
        json.load(f)
    assert rec.recorded + rec.hits == 2


class GatedFake:
    """Inner provider whose calls block on test-controlled synchronisation primitives."""
    name = "anthropic_api"

    def __init__(self, barrier=None, release=None):
        self.barrier = barrier
        self.release = release
        self.entered = threading.Event()
        self.calls = 0
        self._calls_lock = threading.Lock()

    def complete(self, prompt, *, model, system=None, max_tokens=16000):
        with self._calls_lock:
            self.calls += 1
        self.entered.set()
        if self.barrier is not None:
            self.barrier.wait()  # raises BrokenBarrierError on timeout if calls are serialised
        if self.release is not None:
            assert self.release.wait(timeout=5)
        return LLMResult(text=f'{{"p": "{prompt}"}}', model=model, provider=self.name, input_tokens=3,
                         output_tokens=2, cost_usd=0.01, stop_reason="end_turn", duration_ms=5)


def _run_threads(fns):
    errors, threads = [], []

    def wrap(fn):
        def run():
            try:
                fn()
            except BaseException as e:  # surfaced to the test thread below
                errors.append(e)
        return run

    for fn in fns:
        t = threading.Thread(target=wrap(fn))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    return errors


def test_recording_different_keys_runs_in_parallel(tmp_path):
    # Both inner calls must be inside the provider at the same time to pass the barrier;
    # a provider-wide lock would leave one waiting and break the barrier after the timeout.
    inner = GatedFake(barrier=threading.Barrier(2, timeout=5))
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    results = {}
    errors = _run_threads([lambda: results.__setitem__("a", rec.complete("a", model="m")),
                           lambda: results.__setitem__("b", rec.complete("b", model="m"))])
    assert errors == []
    assert inner.calls == 2 and rec.recorded == 2 and rec.misses == 2 and rec.hits == 0
    assert results["a"].text == '{"p": "a"}' and results["b"].text == '{"p": "b"}'
    assert results["a"].cost_usd == 0.01  # a fresh recording keeps the real cost
    assert len(os.listdir(tmp_path / "anthropic_api")) == 2


def test_recording_same_key_concurrently_calls_inner_once(tmp_path):
    release = threading.Event()
    inner = GatedFake(release=release)
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    results = []

    first = threading.Thread(target=lambda: results.append(rec.complete("p", model="m")))
    first.start()
    assert inner.entered.wait(timeout=5)  # the first caller now owns the key and is mid-call
    second = threading.Thread(target=lambda: results.append(rec.complete("p", model="m")))
    second.start()
    release.set()
    first.join(timeout=10); second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()

    assert inner.calls == 1
    assert rec.recorded == 1 and rec.misses == 1 and rec.hits == 1
    assert sorted(r.cost_usd for r in results) == [0.0, 0.01]  # the waiter replays at zero cost
    assert len(os.listdir(tmp_path / "anthropic_api")) == 1


def test_recording_failure_does_not_poison_the_key(tmp_path):
    class FailOnce(GatedFake):
        def complete(self, prompt, **kw):
            if self.calls == 0:
                self.calls += 1
                raise LLMError("boom")
            return super().complete(prompt, **kw)

    inner = FailOnce()
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    with pytest.raises(LLMError, match="boom"):
        rec.complete("p", model="m")
    assert rec.complete("p", model="m").text == '{"p": "p"}'
    assert rec.recorded == 1 and rec.misses == 2
    assert [f for f in os.listdir(tmp_path / "anthropic_api") if f.endswith(".tmp")] == []


def test_counters_are_consistent_under_many_threads(tmp_path):
    inner = GatedFake()
    rec = ReplayProvider(str(tmp_path / "anthropic_api"), inner=inner, record=True)
    keys = [f"k{i}" for i in range(8)]
    errors = _run_threads([(lambda k=k: rec.complete(k, model="m")) for k in keys for _ in range(4)])
    assert errors == []
    assert inner.calls == 8 and rec.recorded == 8 and rec.misses == 8 and rec.hits == 24
