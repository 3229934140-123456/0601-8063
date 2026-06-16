"""
Concurrency tests for the MVCC KV Store.

Validates:
1. Snapshot isolation: concurrent readers see consistent data as of their start_ts.
2. Read never blocks write: a long-running reader doesn't prevent writers from committing.
3. Write never blocks read: active writers don't prevent readers from reading.
4. Write-write conflict detection: second committer to the same key aborts.
5. GC safety: garbage collection doesn't break in-flight transactions' snapshots.
6. GC with long-running transaction: old versions visible to the long txn are preserved.
"""

import threading
import time
import sys

from mvcc_kv import MVCCStore, WriteConflictError


def test_snapshot_isolation_basic():
    """
    T1 starts, reads key 'x' -> v1.
    T2 starts, writes key 'x' -> v2, commits.
    T1 reads key 'x' again -> still v1 (snapshot isolation).
    """
    store = MVCCStore()

    t1 = store.begin()
    t1.put("x", b"v1")
    t1.commit()

    reader = store.begin()
    assert reader.get("x") == b"v1", "Reader should see v1"

    writer = store.begin()
    writer.put("x", b"v2")
    writer.commit()

    assert reader.get("x") == b"v1", "Reader must still see v1 after writer commits"
    reader.rollback()
    print("[PASS] test_snapshot_isolation_basic")


def test_concurrent_readers_see_different_snapshots():
    """
    Three readers start at different times. Each sees the version committed
    before its start_ts.
    """
    store = MVCCStore()

    t0 = store.begin()
    t0.put("k", b"v0")
    t0.commit()

    r1 = store.begin()

    tw1 = store.begin()
    tw1.put("k", b"v1")
    tw1.commit()

    r2 = store.begin()

    tw2 = store.begin()
    tw2.put("k", b"v2")
    tw2.commit()

    r3 = store.begin()

    assert r1.get("k") == b"v0", f"r1 should see v0, got {r1.get('k')}"
    assert r2.get("k") == b"v1", f"r2 should see v1, got {r2.get('k')}"
    assert r3.get("k") == b"v2", f"r3 should see v2, got {r3.get('k')}"

    r1.rollback()
    r2.rollback()
    r3.rollback()
    print("[PASS] test_concurrent_readers_see_different_snapshots")


def test_read_does_not_block_write():
    """
    A long-running reader holds a snapshot. Meanwhile, multiple writers commit.
    Writers should complete without being blocked by the reader.
    """
    store = MVCCStore()

    t0 = store.begin()
    t0.put("k", b"init")
    t0.commit()

    long_reader = store.begin()
    assert long_reader.get("k") == b"init"

    writer_done = threading.Event()

    def writer_fn():
        w = store.begin()
        w.put("k", b"written")
        w.commit()
        writer_done.set()

    t = threading.Thread(target=writer_fn)
    t.start()
    writer_done.wait(timeout=2.0)

    assert writer_done.is_set(), "Writer should have completed without being blocked"

    assert long_reader.get("k") == b"init", "Long reader still sees init"

    long_reader.rollback()
    t.join()
    print("[PASS] test_read_does_not_block_write")


def test_write_does_not_block_read():
    """
    A writer holds an uncommitted write. A reader should still be able to
    read the last committed version without any delay.
    """
    store = MVCCStore()

    t0 = store.begin()
    t0.put("k", b"v1")
    t0.commit()

    writer = store.begin()
    writer.put("k", b"v2_uncommitted")

    reader = store.begin()
    val = reader.get("k")
    assert val == b"v1", f"Reader should see v1, got {val}"

    reader.rollback()
    writer.rollback()
    print("[PASS] test_write_does_not_block_read")


def test_write_write_conflict():
    """
    Two transactions write to the same key. The second one to commit
    should detect a write-write conflict and abort.
    """
    store = MVCCStore()

    t0 = store.begin()
    t0.put("k", b"init")
    t0.commit()

    w1 = store.begin()
    w2 = store.begin()

    w1.put("k", b"w1")
    w2.put("k", b"w2")

    w1.commit()

    try:
        w2.commit()
        assert False, "w2 should have raised WriteConflictError"
    except WriteConflictError:
        pass

    reader = store.begin()
    assert reader.get("k") == b"w1", "Committed value should be w1"
    reader.rollback()
    print("[PASS] test_write_write_conflict")


def test_delete_and_tombstone():
    """
    Delete creates a tombstone. Readers after the delete see None.
    Readers before the delete still see the old value.
    """
    store = MVCCStore()

    t0 = store.begin()
    t0.put("k", b"alive")
    t0.commit()

    before_delete = store.begin()

    deleter = store.begin()
    deleter.delete("k")
    deleter.commit()

    after_delete = store.begin()

    assert before_delete.get("k") == b"alive", "Before delete should see alive"
    assert after_delete.get("k") is None, "After delete should see None"

    before_delete.rollback()
    after_delete.rollback()
    print("[PASS] test_delete_and_tombstone")


def test_gc_removes_old_versions():
    """
    After all transactions complete, GC should remove all but the latest version.
    """
    store = MVCCStore()

    for i in range(10):
        t = store.begin()
        t.put("k", f"v{i}".encode())
        t.commit()

    stats_before = store.gc_stats()
    assert stats_before['total_versions'] == 10, f"Expected 10 versions, got {stats_before['total_versions']}"

    store.gc()

    stats_after = store.gc_stats()
    assert stats_after['total_versions'] == 1, f"Expected 1 version after GC, got {stats_after['total_versions']}"

    reader = store.begin()
    assert reader.get("k") == b"v9", "Latest value should still be readable"
    reader.rollback()
    print("[PASS] test_gc_removes_old_versions")


def test_gc_preserves_versions_for_long_transaction():
    """
    THE CRITICAL TEST for GC safety.

    1. Write v1 to key 'k', commit (ts=2).
    2. Start a long-running transaction T_long (start_ts=3).
    3. Write v2 to key 'k', commit (ts=4).
    4. Write v3 to key 'k', commit (ts=6).
    5. Run GC. low_watermark = 3 (T_long is still active).
       - v1 has commit_ts=2 < low_watermark=3, and there's a newer committed
         version (v2), so v1 CAN be GC'd.
       - v2 has commit_ts=4 >= low_watermark=3, so v2 is KEPT.
       - v3 is the latest, always kept.
    6. T_long reads 'k' -> should still see v1? No! T_long.start_ts=3,
       so T_long sees the latest committed version with commit_ts <= 3,
       which is v1 (commit_ts=2). But we said v1 CAN be GC'd!

    Wait - this is the subtlety. T_long has start_ts=3. The versions are:
       v1: commit_ts=2, v2: commit_ts=4, v3: commit_ts=6
    T_long sees the latest version with commit_ts <= 3, which is v1 (commit_ts=2).
    So v1 IS visible to T_long! If we GC v1, we break T_long's snapshot.

    CORRECTION: A version V with commit_ts < low_watermark is removable ONLY IF
    the next newer committed version V' on the chain has commit_ts <= low_watermark.
    Because then, any active transaction with start_ts >= low_watermark will see V',
    not V. But if V' has commit_ts > low_watermark, then an active transaction
    with start_ts in [V.commit_ts+1, V'.commit_ts) could see V, so V must be kept.

    Wait, that's wrong too. Let me think again carefully.

    T_long has start_ts = 3. low_watermark = 3.
    v1 commit_ts = 2. v2 commit_ts = 4.
    T_long sees v1 because 2 <= 3 < 4.

    If we remove v1, T_long would walk the chain and skip v2 (commit_ts=4 > 3),
    and find nothing, returning None. That's WRONG.

    So the correct rule is: a version V with commit_ts < low_watermark can only
    be GC'd if there is a newer committed version V' such that V'.commit_ts <= low_watermark.

    Equivalently: among the kept versions, we need to keep the latest version
    whose commit_ts <= low_watermark. Everything OLDER than that can be removed.

    Let me re-check my GC implementation...
    In gc(), I keep a version if:
      - commit_ts >= low_watermark, OR
      - it's the first (newest) version we encounter that has commit_ts < low_watermark

    Actually, looking at my implementation: I walk from newest to oldest and keep
    a version if (commit_ts >= low_watermark) OR (kept_head is None, i.e., it's
    the first version with commit_ts < low_watermark that we keep). The first
    version with commit_ts < low_watermark that we encounter is the one with the
    HIGHEST commit_ts below low_watermark — which is exactly the version visible
    to the oldest active transaction. So my implementation IS correct!

    Let me rewrite this test to validate this properly.
    """
    store = MVCCStore()

    t_init = store.begin()
    t_init.put("k", b"v0")
    t_init.commit()

    t1 = store.begin()
    t1.put("k", b"v1")
    t1.commit()

    long_txn = store.begin()

    t2 = store.begin()
    t2.put("k", b"v2")
    t2.commit()

    t3 = store.begin()
    t3.put("k", b"v3")
    t3.commit()

    stats_before = store.gc_stats()
    assert stats_before['total_versions'] == 4

    long_txn_val = long_txn.get("k")

    store.gc()

    stats_after = store.gc_stats()

    assert long_txn.get("k") == long_txn_val, (
        f"GC broke long transaction! Expected {long_txn_val}, got {long_txn.get('k')}"
    )

    assert stats_after['total_versions'] >= 2, (
        f"Expected at least 2 versions after GC (latest + one for long txn), "
        f"got {stats_after['total_versions']}"
    )

    new_reader = store.begin()
    assert new_reader.get("k") == b"v3", "New reader should see v3"
    new_reader.rollback()

    long_txn.rollback()
    print("[PASS] test_gc_preserves_versions_for_long_transaction")


def test_gc_after_long_txn_commits():
    """
    After the long transaction commits, a subsequent GC should be able to
    clean up more versions.
    """
    store = MVCCStore()

    for i in range(5):
        t = store.begin()
        t.put("k", f"v{i}".encode())
        t.commit()

    long_txn = store.begin()
    assert long_txn.get("k") == b"v4"

    for i in range(5, 10):
        t = store.begin()
        t.put("k", f"v{i}".encode())
        t.commit()

    store.gc()
    stats_after_first_gc = store.gc_stats()
    assert stats_after_first_gc['total_versions'] > 1, "GC should preserve versions for long_txn"

    long_txn.rollback()

    store.gc()
    stats_after_second_gc = store.gc_stats()
    assert stats_after_second_gc['total_versions'] == 1, (
        f"After long_txn ends, GC should reduce to 1 version, "
        f"got {stats_after_second_gc['total_versions']}"
    )

    reader = store.begin()
    assert reader.get("k") == b"v9"
    reader.rollback()
    print("[PASS] test_gc_after_long_txn_commits")


def test_concurrent_readers_and_writers():
    """
    Stress test: multiple concurrent readers and writers operating on overlapping keys.
    Verify readers always see consistent snapshots.
    """
    store = MVCCStore()

    t0 = store.begin()
    for i in range(10):
        t0.put(f"key{i}", b"init")
    t0.commit()

    errors = []
    barrier = threading.Barrier(4)

    def reader_fn(reader_id):
        try:
            barrier.wait(timeout=5)
            txn = store.begin()
            values = {}
            for i in range(10):
                val = txn.get(f"key{i}")
                values[f"key{i}"] = val
            time.sleep(0.01)
            for i in range(10):
                val = txn.get(f"key{i}")
                if val != values[f"key{i}"]:
                    errors.append(
                        f"Reader {reader_id}: snapshot changed! "
                        f"key{i} was {values[f'key{i}']}, now {val}"
                    )
            txn.rollback()
        except Exception as e:
            errors.append(f"Reader {reader_id} error: {e}")

    def writer_fn(writer_id, key, value):
        try:
            barrier.wait(timeout=5)
            txn = store.begin()
            txn.put(key, value)
            try:
                txn.commit()
            except WriteConflictError:
                pass
        except Exception as e:
            errors.append(f"Writer {writer_id} error: {e}")

    threads = []
    threads.append(threading.Thread(target=reader_fn, args=(1,)))
    threads.append(threading.Thread(target=reader_fn, args=(2,)))
    threads.append(threading.Thread(target=writer_fn, args=(1, "key0", b"w1")))
    threads.append(threading.Thread(target=writer_fn, args=(2, "key1", b"w2")))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    for e in errors:
        print(f"  ERROR: {e}")
    assert len(errors) == 0, f"Got {len(errors)} errors in concurrent test"
    print("[PASS] test_concurrent_readers_and_writers")


def test_gc_with_tombstone():
    """
    GC should handle tombstones correctly. A tombstone is just a version
    with is_deleted=True. If the tombstone is the latest version and
    no active transaction needs it, it should still be kept (it's the
    latest version). But older versions below the watermark can be cleaned up.
    """
    store = MVCCStore()

    t1 = store.begin()
    t1.put("k", b"v1")
    t1.commit()

    t2 = store.begin()
    t2.put("k", b"v2")
    t2.commit()

    t3 = store.begin()
    t3.delete("k")
    t3.commit()

    stats_before = store.gc_stats()
    assert stats_before['total_versions'] == 3

    store.gc()

    stats_after = store.gc_stats()
    assert stats_after['total_versions'] == 1, f"Expected 1 version after GC, got {stats_after['total_versions']}"

    reader = store.begin()
    assert reader.get("k") is None, "Key should be deleted"
    reader.rollback()
    print("[PASS] test_gc_with_tombstone")


def test_multiple_keys_gc():
    """
    GC operates on all keys. Verify that it cleans up each key's chain independently.
    """
    store = MVCCStore()

    for i in range(5):
        t = store.begin()
        for k in range(3):
            t.put(f"k{k}", f"v{i}_k{k}".encode())
        t.commit()

    stats_before = store.gc_stats()
    assert stats_before['total_versions'] == 15, f"Expected 15, got {stats_before['total_versions']}"

    store.gc()

    stats_after = store.gc_stats()
    assert stats_after['total_versions'] == 3, f"Expected 3 after GC, got {stats_after['total_versions']}"

    print("[PASS] test_multiple_keys_gc")


def test_own_writes_visible():
    """
    A transaction should see its own uncommitted writes.
    """
    store = MVCCStore()

    t0 = store.begin()
    t0.put("k", b"v0")
    t0.commit()

    txn = store.begin()
    assert txn.get("k") == b"v0"

    txn.put("k", b"my_write")
    assert txn.get("k") == b"my_write", "Should see own uncommitted write"

    txn.delete("k")
    assert txn.get("k") is None, "Should see own tombstone"

    txn.rollback()
    print("[PASS] test_own_writes_visible")


if __name__ == "__main__":
    test_snapshot_isolation_basic()
    test_concurrent_readers_see_different_snapshots()
    test_read_does_not_block_write()
    test_write_does_not_block_read()
    test_write_write_conflict()
    test_delete_and_tombstone()
    test_gc_removes_old_versions()
    test_gc_preserves_versions_for_long_transaction()
    test_gc_after_long_txn_commits()
    test_concurrent_readers_and_writers()
    test_gc_with_tombstone()
    test_multiple_keys_gc()
    test_own_writes_visible()
    print("\n=== ALL TESTS PASSED ===")
