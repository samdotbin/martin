"""
Regression tests for scripts/claim_shards.py's per-shard-file redesign.
The concurrent-claim test directly reproduces the production incident that
motivated the redesign: two contributors both ended up claiming shards
24-27 under the old single-shared-claims.json design.
"""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import claim_shards  # noqa: E402


def test_claim_basic(tmp_path):
    shards = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    assert shards == sorted(shards)
    assert len(shards) == 4
    assert all(0 <= i < 48 for i in shards)


def test_claim_creates_one_file_per_shard(tmp_path):
    shards = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    claims_dir = tmp_path / "claims"
    for idx in shards:
        assert (claims_dir / f"shard{idx}.json").exists()


def test_claim_idempotent_same_want(tmp_path):
    first = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    second = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    assert first == second


def test_claim_incremental_want_keeps_original(tmp_path):
    first = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    more = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=6, name="alice")
    assert set(first).issubset(set(more))
    assert len(more) == 6


def test_claim_two_contributors_get_disjoint_shards(tmp_path):
    alice = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    bob = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="bob")
    assert set(alice).isdisjoint(set(bob))


def test_claim_exhaustion_raises(tmp_path):
    claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=4, name="alice")
    with pytest.raises(RuntimeError, match="no free shards left"):
        claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=1, name="bob")


def test_claim_exhaustion_partial_still_returns_what_it_has(tmp_path):
    # alice already holds 2 of 4 total; asking for 4 should raise (can't
    # fully satisfy), but the two she already has must not be lost.
    claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=2, name="alice")
    claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=2, name="bob")
    with pytest.raises(RuntimeError):
        claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=3, name="alice")
    # alice's original 2 are still intact and re-claimable idempotently
    still_mine = claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=2, name="alice")
    assert len(still_mine) == 2


def test_release_then_reclaim(tmp_path):
    shards = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    released = claim_shards.release(str(tmp_path), "alice")
    assert released == shards
    # released shards are free again
    bob = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=48, name="bob")
    assert set(shards).issubset(set(bob))


def test_release_only_affects_named_contributor(tmp_path):
    alice = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    bob = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="bob")
    claim_shards.release(str(tmp_path), "alice")
    still_bobs = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="bob")
    assert still_bobs == bob


def test_concurrent_claims_never_overlap(tmp_path):
    """Direct reproduction of the production incident: many contributors
    claiming around the same time must never end up with the same shard.
    Under the old single-claims.json design this could lose updates; the
    per-shard-file 'exclusive create' design must not."""
    shared_folder = str(tmp_path)
    names = [f"contributor-{i}" for i in range(10)]
    results = {}
    errors = []

    def _worker(name):
        try:
            results[name] = claim_shards.claim(shared_folder, total_shards=40, n_wanted=4, name=name)
        except Exception as e:  # noqa: BLE001
            errors.append((name, e))

    threads = [threading.Thread(target=_worker, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors: {errors}"
    all_claimed = [idx for shards in results.values() for idx in shards]
    assert len(all_claimed) == len(set(all_claimed)), (
        f"duplicate shard claim detected across contributors: {results}"
    )


def test_claim_preferred_indices_honored_when_free(tmp_path):
    shards = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice",
                                 preferred=[40, 41, 42, 43])
    assert shards == [40, 41, 42, 43]


def test_claim_preferred_falls_back_when_taken(tmp_path):
    claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=1, name="bob", preferred=[40])
    # alice prefers 40-43, but 40 is bob's -- she should get 41-43 plus one
    # more from the normal scan, not fail or collide with bob.
    shards = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice",
                                 preferred=[40, 41, 42, 43])
    assert 40 not in shards
    assert {41, 42, 43}.issubset(set(shards))
    assert len(shards) == 4


def test_claim_preferred_out_of_range_ignored(tmp_path):
    shards = claim_shards.claim(str(tmp_path), total_shards=4, n_wanted=2, name="alice",
                                 preferred=[99, -1, 2])
    assert 2 in shards
    assert len(shards) == 2


def test_gpu_status_reports_claims(tmp_path):
    claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=4, name="alice")
    status = claim_shards.gpu_status(str(tmp_path), total_shards=48)
    assert len(status) == 1
    assert status[0]["name"] == "alice"
    assert len(status[0]["shards"]) == 4
    assert status[0]["online"] is False  # no checkpoint/log activity yet


def test_gpu_status_online_via_recent_file_activity(tmp_path):
    shards = claim_shards.claim(str(tmp_path), total_shards=48, n_wanted=1, name="alice")
    shard_dir = tmp_path / f"shard{shards[0]}"
    (shard_dir).mkdir(parents=True, exist_ok=True)
    (shard_dir / "train_log.txt").write_text("hello")
    status = claim_shards.gpu_status(str(tmp_path), total_shards=48, online_window_minutes=20)
    assert status[0]["online"] is True


def test_corrupt_claim_file_treated_as_absent_not_crash(tmp_path):
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir(parents=True)
    (claims_dir / "shard0.json").write_text("{not valid json")
    # Should not raise -- corrupt file is treated as unreadable/absent for
    # listing purposes (see module docstring on this tradeoff).
    result = claim_shards._list_claims(str(tmp_path), total_shards=4)
    assert 0 not in result
