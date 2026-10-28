import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from pipeline.llm.anthropic_batch import (BATCH_PROVIDER, build_request, entry_result, submit_batch,
                                          wait_batch)
from pipeline.llm.base import LLMError, LLMFatalError, LLMInterrupted, LLMTransientError
from pipeline.pricing import estimate_cost
from tests.helpers import FakeBatchClient, fake_message

KEY = "a" * 64


def _errored(etype, message="boom"):
    return SimpleNamespace(type="errored", error=SimpleNamespace(type="error",
                           error=SimpleNamespace(type=etype, message=message)))


def _entry(result):
    return SimpleNamespace(custom_id=KEY, result=result)


def _api_error(status, etype, message):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    return anthropic.APIStatusError(message, response=resp, body={"error": {"type": etype, "message": message}})


# --- requests and results ---------------------------------------------------------------------

def test_build_request_with_and_without_system():
    r = build_request(KEY, "hi", model="claude-haiku-5-5", system="sys", max_tokens=1000)
    assert r == {"custom_id": KEY, "params": {"model": "claude-haiku-5-5", "max_tokens": 1000,
                                              "messages": [{"role": "user", "content": "hi"}], "system": "sys"}}
    assert "system" not in build_request(KEY, "hi", model="m", system=None, max_tokens=10)["params"]


def test_succeeded_entry_is_priced_at_the_batch_rate():
    msg = fake_message("ok", model="claude-sonnet-5-5", input_tokens=1000, output_tokens=200)
    r = entry_result(_entry(SimpleNamespace(type="succeeded", message=msg)), "claude-sonnet-5-5", 1234)
    assert r.text == "ok" and r.provider == BATCH_PROVIDER == "anthropic_batch" and r.duration_ms == 1234
    assert r.cost_usd == pytest.approx(estimate_cost("claude-sonnet-5-5", 1000, 200, batch=True))


def test_errored_entry_maps_like_a_direct_call():
    with pytest.raises(LLMFatalError):
        entry_result(_entry(_errored("authentication_error")), "m", 0)
    with pytest.raises(LLMFatalError):
        entry_result(_entry(_errored("invalid_request_error", "Your credit balance is too low")), "m", 0)
    with pytest.raises(LLMTransientError):
        entry_result(_entry(_errored("overloaded_error")), "m", 0)
    with pytest.raises(LLMError) as e:
        entry_result(_entry(_errored("invalid_request_error", "prompt is too long")), "m", 0)
    assert not isinstance(e.value, (LLMFatalError, LLMTransientError))


def test_expired_and_canceled_entries_are_errors():
    with pytest.raises(LLMError, match="expired"):
        entry_result(_entry(SimpleNamespace(type="expired")), "m", 0)
    with pytest.raises(LLMError, match="canceled"):
        entry_result(_entry(SimpleNamespace(type="canceled")), "m", 0)


# --- submit and wait --------------------------------------------------------------------------

def test_submit_and_wait_return_entries_by_custom_id():
    client = FakeBatchClient(lambda p: fake_message("echo " + p["messages"][0]["content"]), polls_until_end=3)
    bid = submit_batch(client, [build_request(KEY, "x", model="m", system=None, max_tokens=10)])
    entries = wait_batch(client, bid, poll_seconds=0, max_wait_seconds=60, log=lambda m: None)
    assert list(entries) == [KEY]
    assert entries[KEY].result.message.content[0].text == "echo x"


def test_submit_credit_error_is_fatal():
    client = FakeBatchClient(lambda p: fake_message("x"))

    def refuse(*, requests):
        raise _api_error(400, "invalid_request_error", "Your credit balance is too low")

    client.messages.batches.create = refuse
    with pytest.raises(LLMFatalError):
        submit_batch(client, [build_request(KEY, "x", model="m", system=None, max_tokens=10)])


def test_wait_past_max_wait_cancels_and_raises():
    client = FakeBatchClient(lambda p: fake_message("x"), polls_until_end=10**6)
    bid = submit_batch(client, [build_request(KEY, "x", model="m", system=None, max_tokens=10)])
    with pytest.raises(LLMError, match="cancelled"):
        wait_batch(client, bid, poll_seconds=0, max_wait_seconds=0, log=lambda m: None)
    assert client.cancelled == [bid]


def test_wait_retries_a_transient_poll_error():
    client = FakeBatchClient(lambda p: fake_message("x"), polls_until_end=1)
    bid = submit_batch(client, [build_request(KEY, "x", model="m", system=None, max_tokens=10)])
    real, calls = client.messages.batches.retrieve, []

    def flaky(batch_id):
        calls.append(batch_id)
        if len(calls) == 1:
            raise _api_error(529, "overloaded_error", "overloaded")
        return real(batch_id)

    client.messages.batches.retrieve = flaky
    assert KEY in wait_batch(client, bid, poll_seconds=0, max_wait_seconds=60, log=lambda m: None)
    assert len(calls) == 2


def _results_failing_once(client):
    real, calls = client.messages.batches.results, []

    def results(batch_id):
        calls.append(batch_id)
        if len(calls) == 1:
            def broken():
                raise httpx.ReadError("connection reset while streaming results")
                yield
            return broken()
        return real(batch_id)
    client.messages.batches.results = results
    return calls


def test_a_network_error_reading_results_is_transient():
    client = FakeBatchClient(lambda p: fake_message("x"))
    _results_failing_once(client)
    bid = submit_batch(client, [build_request(KEY, "x", model="m", system=None, max_tokens=10)])
    with pytest.raises(LLMTransientError, match="batch results"):
        wait_batch(client, bid, poll_seconds=0, max_wait_seconds=60, log=lambda m: None)


def test_a_retried_results_read_reuses_the_batch_instead_of_sending_it_again(tmp_path):
    from pipeline.llm.anthropic_batch import AnthropicBatchProvider, BatchJournal
    from pipeline.llm.client import LLMClient
    from pydantic import BaseModel

    class Out(BaseModel):
        value: int
    client = FakeBatchClient(lambda p: fake_message('{"value": 1}'))
    calls = _results_failing_once(client)
    p = AnthropicBatchProvider(client, log=lambda m: None, min_requests=1, gather_seconds=0.05,
                               poll_seconds=0.01, journal=BatchJournal(str(tmp_path / "batches.json")))
    llm = LLMClient(p, db_path=str(tmp_path / "s.sqlite"), run_id=None, notebook_id=None,
                    retry_base_delay=0, sleep=lambda s: None, log=lambda m: None)
    from pipeline import db
    db.init_db(str(tmp_path / "s.sqlite"))
    assert llm.call_json("p", Out, model="claude-haiku-5-5", stage="s").value == 1
    assert len(client.created) == 1 and len(calls) == 2      # read twice, sent once


def test_wait_stops_on_interrupt_and_cancels():
    from pipeline.llm.base import request_interrupt
    client = FakeBatchClient(lambda p: fake_message("x"), polls_until_end=10**6)
    bid = submit_batch(client, [build_request(KEY, "x", model="m", system=None, max_tokens=10)])
    request_interrupt()
    with pytest.raises(LLMInterrupted, match="interrupted"):
        wait_batch(client, bid, poll_seconds=0, max_wait_seconds=60, log=lambda m: None)
    assert client.cancelled == [bid]


import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline.llm.anthropic_batch import AnthropicBatchProvider
from pipeline.llm.base import request_interrupt


def _echo(params):
    return fake_message("echo " + params["messages"][0]["content"])


def _provider(client, **kw):
    kw.setdefault("gather_seconds", 0.1)
    kw.setdefault("poll_seconds", 0.01)
    return AnthropicBatchProvider(client, log=lambda m: None, **kw)


def _call_all(p, prompts, model="claude-haiku-5-5"):
    """Every prompt from its own thread at once, like a stage pool; exceptions are returned, not raised."""
    def one(prompt):
        try:
            return p.complete(prompt, model=model, system="sys", max_tokens=1000)
        except Exception as e:
            return e
    with ThreadPoolExecutor(len(prompts)) as pool:
        return list(pool.map(one, prompts))


def test_provider_keeps_the_api_name_and_says_it_batches():
    p = _provider(FakeBatchClient(_echo))
    assert p.name == "anthropic_api" and p.batching is True


def test_concurrent_calls_share_one_batch():
    client = FakeBatchClient(_echo)
    results = _call_all(_provider(client), ["a", "b", "c"])
    assert [r.text for r in results] == ["echo a", "echo b", "echo c"]
    assert [len(b) for b in client.created] == [3] and client.direct_calls == 0
    assert {r.provider for r in results} == {"anthropic_batch"}
    assert results[0].cost_usd == pytest.approx(estimate_cost("claude-haiku-5-5", 10, 5, batch=True))


def test_a_wave_below_min_requests_goes_direct_at_full_price():
    client = FakeBatchClient(_echo)
    r = _provider(client).complete("solo", model="claude-haiku-5-5", system="sys", max_tokens=1000)
    assert r.text == "echo solo" and r.provider == "anthropic_api" and r.cost_usd is None
    assert client.created == [] and client.direct_calls == 1


def test_max_wave_splits_a_large_wave():
    client = FakeBatchClient(_echo)
    _call_all(_provider(client, max_wave=2, min_requests=1), ["a", "b", "c"])
    assert sorted(len(b) for b in client.created) == [1, 2]


def test_a_wave_over_the_request_size_limit_is_split(monkeypatch):
    import pipeline.llm.anthropic_batch as mod
    one = len(json.dumps(build_request(KEY, "x" * 1000, model="claude-haiku-5-5", system="sys", max_tokens=1000)))
    monkeypatch.setattr(mod, "MAX_BATCH_BYTES", int(one * 2.5))     # two requests fit, three do not
    client = FakeBatchClient(_echo)
    prompts = [c * 1000 for c in "abcde"]
    results = _call_all(_provider(client), prompts)
    assert [r.text for r in results] == ["echo " + p for p in prompts]
    assert sorted(len(b) for b in client.created) == [1, 2, 2]


def test_interrupting_a_split_wave_cancels_every_batch_it_sent(monkeypatch):
    import pipeline.llm.anthropic_batch as mod
    one = len(json.dumps(build_request(KEY, "x" * 1000, model="claude-haiku-5-5", system="sys", max_tokens=1000)))
    monkeypatch.setattr(mod, "MAX_BATCH_BYTES", int(one * 1.5))     # one request per batch
    client = FakeBatchClient(_echo, polls_until_end=10**6)
    threading.Timer(0.5, request_interrupt).start()
    results = _call_all(_provider(client), ["a" * 1000, "b" * 1000])
    assert all(isinstance(r, LLMInterrupted) for r in results)
    assert sorted(client.cancelled) == ["msgbatch_1", "msgbatch_2"]


def test_split_requests_keeps_order_and_an_oversized_request_alone():
    from pipeline.llm.anthropic_batch import split_requests
    reqs = [{"custom_id": str(i), "params": {"p": "x" * n}} for i, n in enumerate([10, 10, 500, 10])]
    chunks = split_requests(reqs, 100)
    assert [[r["custom_id"] for r in c] for c in chunks] == [["0", "1"], ["2"], ["3"]]


def test_identical_requests_are_sent_once_and_billed_once():
    client = FakeBatchClient(_echo)
    results = _call_all(_provider(client), ["same", "same", "other"])
    assert [r.text for r in results] == ["echo same", "echo same", "echo other"]
    assert len(client.created[0]) == 2
    assert sum(r.input_tokens for r in results[:2]) == 10      # one of the two carries the tokens
    assert sum(r.cost_usd for r in results[:2]) == pytest.approx(estimate_cost("claude-haiku-5-5", 10, 5, batch=True))


def test_one_failed_request_fails_only_its_caller():
    def answer(params):
        if params["messages"][0]["content"] == "bad":
            return _errored("overloaded_error")
        return _echo(params)
    results = _call_all(_provider(FakeBatchClient(answer)), ["ok", "bad"])
    assert results[0].text == "echo ok"
    assert isinstance(results[1], LLMTransientError)


def test_fatal_create_error_reaches_every_caller():
    client = FakeBatchClient(_echo)

    def refuse(*, requests):
        raise _api_error(400, "invalid_request_error", "Your credit balance is too low")

    client.messages.batches.create = refuse
    results = _call_all(_provider(client), ["a", "b"])
    assert all(isinstance(r, LLMFatalError) for r in results)


def test_interrupt_cancels_the_batch_and_releases_callers():
    client = FakeBatchClient(_echo, polls_until_end=10**6)
    threading.Timer(0.5, request_interrupt).start()
    t0 = time.monotonic()
    results = _call_all(_provider(client), ["a", "b"])
    assert time.monotonic() - t0 < 5
    # LLMInterrupted, not a plain LLMError: the analyze worker leaves these sources pending
    assert all(isinstance(r, LLMInterrupted) for r in results)
    assert client.cancelled == ["msgbatch_1"]


def test_same_prompt_with_different_max_tokens_is_not_merged():
    client = FakeBatchClient(_echo)
    p = _provider(client)

    def one(max_tokens):
        return p.complete("same", model="claude-haiku-5-5", system="sys", max_tokens=max_tokens)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(one, [1000, 2000]))
    assert [r.input_tokens for r in results] == [10, 10]           # each billed: two requests sent
    assert sorted(r["params"]["max_tokens"] for r in client.created[0]) == [1000, 2000]


def test_request_key_is_a_valid_custom_id_and_covers_max_tokens():
    import re
    from pipeline.llm.anthropic_batch import request_key
    k = request_key("m", "s", "p", 1000)
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", k)                 # the API's custom_id format
    assert k != request_key("m", "s", "p", 2000) and k != request_key("m", None, "p", 1000)


def test_calls_after_a_wave_form_the_next_wave():
    client = FakeBatchClient(_echo)
    p = _provider(client)
    _call_all(p, ["a", "b"])
    _call_all(p, ["c", "d"])
    assert [len(b) for b in client.created] == [2, 2]


import json
import os
from datetime import datetime, timedelta

from pipeline.llm.anthropic_batch import RESULTS_DAYS, BatchJournal, request_key


def test_journal_records_finds_and_forgets(tmp_path):
    j = BatchJournal(str(tmp_path / "cache" / "batches.json"))
    j.record("b1", ["k1", "k2"])
    j.record("b2", ["k2"])
    assert j.batches_for(["k2"]) == ["b2", "b1"] and j.batches_for(["zz"]) == []
    j.forget(["k2"])
    assert j.batches_for(["k2"]) == [] and j.batches_for(["k1"]) == ["b1"]
    j.forget(["k1"])
    assert json.loads((tmp_path / "cache" / "batches.json").read_text(encoding="utf-8")) == []


def test_journal_prunes_old_batches_and_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "batches.json"
    t0 = datetime(2026, 10, 1)
    BatchJournal(str(path), now=lambda: t0).record("old", ["k"])
    later = BatchJournal(str(path), now=lambda: t0 + timedelta(days=RESULTS_DAYS + 1))
    assert later.batches_for(["k"]) == []
    path.write_text("{not json", encoding="utf-8")
    assert BatchJournal(str(path)).batches_for(["k"]) == []


def test_two_journals_on_one_file_never_lose_each_others_records(tmp_path):
    # two hmd processes sharing one home: each has its own BatchJournal (and thread lock) on one file
    path = str(tmp_path / "cache" / "batches.json")
    journals = [BatchJournal(path), BatchJournal(path)]

    def write(i):
        journals[i % 2].record(f"b{i}", [f"k{i}"])
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(write, range(40)))
    assert sorted(journals[0].batches_for([f"k{i}" for i in range(40)])) == sorted(f"b{i}" for i in range(40))
    assert not os.path.exists(path + ".lock")


def test_a_stale_journal_lock_is_taken_over(tmp_path):
    path = str(tmp_path / "batches.json")
    lock = path + ".lock"
    open(lock, "w").close()
    old = time.time() - 3600
    os.utime(lock, (old, old))                 # left behind by a process that crashed holding it
    j = BatchJournal(path)
    j.record("b1", ["k1"])
    assert j.batches_for(["k1"]) == ["b1"] and not os.path.exists(lock)


def test_a_wave_reuses_an_earlier_batch_and_sends_only_the_rest(tmp_path):
    client = FakeBatchClient(_echo)
    journal = BatchJournal(str(tmp_path / "batches.json"))
    key_a = request_key("claude-haiku-5-5", "sys", "a", 1000)
    old = submit_batch(client, [build_request(key_a, "a", model="claude-haiku-5-5", system="sys", max_tokens=1000)])
    journal.record(old, [key_a])
    results = _call_all(_provider(client, journal=journal), ["a", "b"])
    assert [r.text for r in results] == ["echo a", "echo b"]
    assert [len(b) for b in client.created] == [1, 1]          # the old batch, then "b" alone
    assert journal.batches_for([key_a]) == []                 # delivered, so forgotten


def _journal_with_unreachable_batch(tmp_path, status, etype):
    """A journal naming an earlier batch whose poll now fails with `status`; returns (client, journal, key)."""
    client = FakeBatchClient(_echo)
    journal = BatchJournal(str(tmp_path / "batches.json"))
    key_a = request_key("claude-haiku-5-5", "sys", "a", 1000)
    journal.record("msgbatch_gone", [key_a, "z" * 64])     # a key no later wave asks for
    real = client.messages.batches.retrieve

    def retrieve(batch_id):
        if batch_id == "msgbatch_gone":
            raise _api_error(status, etype, "no such batch" if status == 404 else "forbidden")
        return real(batch_id)
    client.messages.batches.retrieve = retrieve
    return client, journal, key_a


def test_a_journal_batch_that_is_gone_is_dropped_and_the_wave_sent_fresh(tmp_path):
    client, journal, key_a = _journal_with_unreachable_batch(tmp_path, 404, "not_found_error")
    results = _call_all(_provider(client, journal=journal), ["a", "b"])
    assert [r.text for r in results] == ["echo a", "echo b"]
    assert [len(b) for b in client.created] == [2]
    assert journal.batches_for(["z" * 64]) == []          # dropped, so never polled again


def test_a_403_on_reuse_sends_fresh_with_a_warning_instead_of_stopping(tmp_path):
    client, journal, key_a = _journal_with_unreachable_batch(tmp_path, 403, "permission_error")
    msgs = []
    p = AnthropicBatchProvider(client, log=msgs.append, gather_seconds=0.1, poll_seconds=0.01, journal=journal)
    results = _call_all(p, ["a", "b"])
    assert [r.text for r in results] == ["echo a", "echo b"]
    assert [len(b) for b in client.created] == [2]
    assert any("msgbatch_gone" in m and "warning" in m for m in msgs)
    assert journal.batches_for(["z" * 64]) == ["msgbatch_gone"]     # kept: it may be reachable later


def test_an_interrupted_wave_stays_in_the_journal(tmp_path):
    client = FakeBatchClient(_echo, polls_until_end=10**6)
    journal = BatchJournal(str(tmp_path / "batches.json"))
    threading.Timer(0.5, request_interrupt).start()
    _call_all(_provider(client, journal=journal), ["a", "b"])
    assert journal.batches_for([request_key("claude-haiku-5-5", "sys", "a", 1000)]) == ["msgbatch_1"]


def test_interrupt_during_journal_reuse_sends_no_new_batch(tmp_path):
    client = FakeBatchClient(_echo, polls_until_end=10**6)
    journal = BatchJournal(str(tmp_path / "batches.json"))
    key_a = request_key("claude-haiku-5-5", "sys", "a", 1000)
    old = submit_batch(client, [build_request(key_a, "a", model="claude-haiku-5-5", system="sys", max_tokens=1000)])
    journal.record(old, [key_a])
    threading.Timer(0.5, request_interrupt).start()
    results = _call_all(_provider(client, journal=journal), ["a", "b"])
    assert len(client.created) == 1                      # nothing new sent after Ctrl+C
    assert all(isinstance(r, LLMInterrupted) for r in results)


def test_a_failing_dispatch_does_not_strand_callers(monkeypatch):
    import pipeline.llm.anthropic_batch as mod
    real_thread = threading.Thread

    def thread(*a, **k):
        if k.get("name") == "hmd-batch-wave":
            raise RuntimeError("can't start new thread")
        return real_thread(*a, **k)

    monkeypatch.setattr(mod.threading, "Thread", thread)
    p = _provider(FakeBatchClient(_echo))
    out = []
    t = real_thread(target=lambda: out.extend(_call_all(p, ["a", "b"])), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "callers are stranded"
    assert all(isinstance(r, LLMError) and "dispatch" in str(r) for r in out)
    # the dispatcher survives: a later single call still goes direct
    monkeypatch.setattr(mod.threading, "Thread", real_thread)
    assert p.complete("solo", model="claude-haiku-5-5", system="sys", max_tokens=1000).text == "echo solo"


def test_a_call_interrupted_before_its_wave_is_sent_raises_llm_interrupted():
    client = FakeBatchClient(_echo)
    p = _provider(client, gather_seconds=5)
    threading.Timer(0.3, request_interrupt).start()
    results = _call_all(p, ["a", "b"])
    assert all(isinstance(r, LLMInterrupted) for r in results) and client.created == []


def test_an_interrupted_batch_leaves_the_analyze_source_pending(tmp_path):
    # 2026-10-15 review: a cancelled batch marked its sources as analyze errors
    from pipeline import db
    from pipeline.llm.client import LLMClient
    from pipeline.transcription.base import Segment, Transcript
    from pipeline.transcription.cache import TranscriptCache
    from pipeline.workers.analyze import analyze_source
    path = str(tmp_path / "s.sqlite")
    db.init_db(path)
    nid = db.create_notebook(path, "T", "t")
    cache = TranscriptCache(str(tmp_path / "cache"))
    sids = []
    for vid in ("aaaaaaaaaaa", "bbbbbbbbbbb"):
        sid = db.add_source(path, nid, vid, f"https://youtu.be/{vid}")
        tr = Transcript(video_id=vid, language="en", method="youtube_captions",
                        segments=[Segment(start=0.0, end=5.0, text="some words about the topic")])
        cache.save(tr)
        db.update_transcribe_status(path, sid, "done", transcript_path=cache.path(vid, "youtube_captions"),
                                    method="youtube_captions")
        sids.append(sid)
    client = FakeBatchClient(_echo, polls_until_end=10**6)
    llm = LLMClient(_provider(client), db_path=path, run_id=None, notebook_id=nid, log=lambda m: None)
    threading.Timer(0.5, request_interrupt).start()

    def one(sid):
        try:
            return analyze_source(path, sid, "T", llm, "claude-haiku-5-5", cache, log=lambda m: None)
        except Exception as e:
            return e
    with ThreadPoolExecutor(2) as pool:
        out = list(pool.map(one, sids))
    assert all(isinstance(r, LLMInterrupted) for r in out)
    for sid in sids:
        src = db.get_source(path, sid)
        assert src["analyze_status"] == "pending" and not src["error_message"]
