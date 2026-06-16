"""
Concurrency stress tests for MVCC KV Store.

Tests three critical scenarios:
1. Same-key write-write conflict: N threads racing to commit writes to the
   same key. EXACTLY ONE must succeed, all others must get WriteConflictError.
   Must never result in both commits succeeding with random value overwrites.

2. Multi-key atomic visibility: a transaction writes to multiple keys.
   Any new read transaction must see EITHER all old values OR all new values.
   Must never see a partial (half-committed) view where some keys are updated
   and others are not.

3. Read-write non-blocking stress: long-running reads on hot key + high-
   frequency writes. Reads must not block writes, writes must not block reads.
   New read transactions must acquire their snapshot quickly.

All tests are designed to be repeatable and stable across many runs.
"""

import threading
import time
import random
from typing import List, Set, Dict, Tuple

from mvcc_kv import MVCCStore, WriteConflictError


# ============================================================================
# Test 1: Same-Key Write-Write Conflict Stress
# ============================================================================

def test_same_key_conflict_stress(num_rounds: int = 10, num_threads: int = 8):
    """
    Stress test: N threads race to commit a write to the same key.
    In each round, exactly ONE must succeed, all others must fail with
    WriteConflictError.

    This catches the race condition where two transactions both pass the
    conflict check and both commit, resulting in random value overwrites.
    """
    for round_num in range(num_rounds):
        store = MVCCStore()

        t_init = store.begin()
        t_init.put("hot_key", b"initial")
        t_init.commit()

        success_count: List[int] = [0]
        fail_count: List[int] = [0]
        final_value: List[bytes] = [None]
        successful_thread: List[int] = [-1]
        errors: List[str] = []
        counter_lock = threading.Lock()
        begin_barrier = threading.Barrier(num_threads)
        commit_barrier = threading.Barrier(num_threads)

        def writer_fn(writer_id):
            try:
                begin_barrier.wait(timeout=5)
                txn = store.begin()
                txn.put("hot_key", f"writer_{writer_id}".encode())
                commit_barrier.wait(timeout=5)
                try:
                    txn.commit()
                    with counter_lock:
                        success_count[0] += 1
                    final_value[0] = f"writer_{writer_id}".encode()
                    successful_thread[0] = writer_id
                except WriteConflictError:
                    with counter_lock:
                        fail_count[0] += 1
            except Exception as e:
                errors.append(f"Thread {writer_id} error: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=writer_fn, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        for t in threads:
            if t.is_alive():
                errors.append(f"Thread hung!")

        for key in ["hot_key"]:
            lock = store._lock_manager.get_lock(key)
            if lock.locked():
                errors.append(f"Lock for {key} is still held after all threads finished!")

        if success_count[0] != 1:
            errors.append(
                f"Expected exactly 1 successful commit, got {success_count[0]} "
                f"(success={success_count[0]}, fail={fail_count[0]})"
            )

        if success_count[0] + fail_count[0] != num_threads:
            errors.append(
                f"Thread count mismatch! Expected {num_threads} total, "
                f"got {success_count[0] + fail_count[0]} "
                f"(success={success_count[0]}, fail={fail_count[0]})"
            )

        reader = store.begin()
        actual_final = reader.get("hot_key")
        reader.rollback()

        expected_writer_id = successful_thread[0]
        if expected_writer_id >= 0:
            expected_value = f"writer_{expected_writer_id}".encode()
            if actual_final != expected_value:
                errors.append(
                    f"Final value mismatch! Expected {expected_value} "
                    f"(from writer {expected_writer_id}), got {actual_final}"
                )

        if len(errors) > 0:
            print(f"  Round {round_num} ERRORS:")
            for e in errors:
                print(f"    - {e}")
            raise AssertionError(f"Round {round_num} failed with {len(errors)} errors")

    print(f"[PASS] test_same_key_conflict_stress ({num_rounds} rounds, {num_threads} threads each)")


def test_same_key_conflict_many_keys(num_rounds: int = 5, num_threads: int = 6, num_keys: int = 5):
    """
    Variation: multiple keys, each with concurrent writers.
    For each key, exactly one writer should succeed.
    """
    for round_num in range(num_rounds):
        store = MVCCStore()

        for k in range(num_keys):
            t = store.begin()
            t.put(f"key{k}", b"init")
            t.commit()

        results: Dict[str, List[int]] = {f"key{k}": [] for k in range(num_keys)}
        errors: List[str] = []
        lock = threading.Lock()
        commit_barriers: Dict[str, threading.Barrier] = {
            f"key{k}": threading.Barrier(num_threads) for k in range(num_keys)
        }

        def writer_fn(writer_id, key):
            try:
                txn = store.begin()
                txn.put(key, f"w{writer_id}".encode())
                commit_barriers[key].wait(timeout=5)
                try:
                    txn.commit()
                    with lock:
                        results[key].append(writer_id)
                except WriteConflictError:
                    pass
            except Exception as e:
                with lock:
                    errors.append(f"Writer {writer_id} on {key}: {e}")

        threads = []
        writer_counter = 0
        for k in range(num_keys):
            for _ in range(num_threads):
                t = threading.Thread(target=writer_fn, args=(writer_counter, f"key{k}"))
                threads.append(t)
                writer_counter += 1

        random.shuffle(threads)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for key, winners in results.items():
            if len(winners) != 1:
                errors.append(
                    f"Key {key}: expected exactly 1 winner, got {len(winners)}: {winners}"
                )

            reader = store.begin()
            val = reader.get(key)
            reader.rollback()
            expected = f"w{winners[0]}".encode() if winners else b"init"
            if val != expected:
                errors.append(f"Key {key}: value mismatch, got {val}, expected {expected}")

        if errors:
            for e in errors:
                print(f"    - {e}")
            raise AssertionError(f"Round {round_num} failed")

    print(f"[PASS] test_same_key_conflict_many_keys ({num_rounds} rounds, {num_keys} keys, {num_threads} writers/key)")


# ============================================================================
# Test 2: Multi-Key Atomic Visibility
# ============================================================================

def test_multi_key_atomic_visibility(num_rounds: int = 50):
    """
    Test that multi-key transactions are atomically visible.

    A writer transaction repeatedly updates N keys from (v0, v0, ..., v0)
    to (v1, v1, ..., v1) and back. Concurrently, many reader transactions
    start, read all N keys, and verify that they are ALL the same value.

    If there's any time window where a reader sees some keys as v0 and
    some as v1, we have a half-committed visibility bug.
    """
    for round_num in range(num_rounds):
        store = MVCCStore()
        num_keys = 5
        iterations_per_writer = 20
        num_readers = 10

        t_init = store.begin()
        for k in range(num_keys):
            t_init.put(f"k{k}", b"v0")
        t_init.commit()

        errors: List[str] = []
        stop_flag = threading.Event()
        writer_done = threading.Event()

        def writer_fn():
            try:
                for i in range(iterations_per_writer):
                    val = b"v1" if i % 2 == 0 else b"v0"
                    txn = store.begin()
                    for k in range(num_keys):
                        txn.put(f"k{k}", val)
                    try:
                        txn.commit()
                    except WriteConflictError:
                        pass
                writer_done.set()
                stop_flag.set()
            except Exception as e:
                errors.append(f"Writer error: {e}")
                stop_flag.set()

        def reader_fn(reader_id):
            try:
                while not stop_flag.is_set():
                    txn = store.begin()
                    values = []
                    for k in range(num_keys):
                        val = txn.get(f"k{k}")
                        values.append(val)
                    txn.rollback()

                    first_val = values[0]
                    for i, v in enumerate(values):
                        if v != first_val:
                            errors.append(
                                f"Reader {reader_id} saw INCONSISTENT values! "
                                f"k0={first_val}, k{i}={v}, all={values}"
                            )
                            stop_flag.set()
                            return

                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader {reader_id} error: {e}")

        threads = []
        writer_thread = threading.Thread(target=writer_fn)
        threads.append(writer_thread)
        writer_thread.start()

        for i in range(num_readers):
            t = threading.Thread(target=reader_fn, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        for t in threads:
            if t.is_alive():
                errors.append("Thread hung!")

        if errors:
            print(f"  Round {round_num} ERRORS:")
            for e in errors:
                print(f"    - {e}")
            raise AssertionError(f"Round {round_num} failed")

    print(f"[PASS] test_multi_key_atomic_visibility ({num_rounds} rounds)")


def test_multi_key_atomic_with_conflicts(num_rounds: int = 20):
    """
    Multi-key atomic visibility under write-write conflicts.
    Multiple writers are concurrently updating the same set of keys.
    Readers must never see a partial view.
    """
    for round_num in range(num_rounds):
        store = MVCCStore()
        num_keys = 4

        t_init = store.begin()
        for k in range(num_keys):
            t_init.put(f"k{k}", b"v0")
        t_init.commit()

        errors: List[str] = []
        stop_flag = threading.Event()
        writers_done = threading.Event()
        active_writers = [0]
        counter_lock = threading.Lock()

        def writer_fn(writer_id, num_iterations: int):
            try:
                with counter_lock:
                    active_writers[0] += 1
                for i in range(num_iterations):
                    val = f"w{writer_id}_i{i}".encode()
                    txn = store.begin()
                    for k in range(num_keys):
                        txn.put(f"k{k}", val)
                    try:
                        txn.commit()
                    except WriteConflictError:
                        pass
                    time.sleep(0.0005)
            except Exception as e:
                errors.append(f"Writer {writer_id} error: {e}")
            finally:
                with counter_lock:
                    active_writers[0] -= 1
                    if active_writers[0] == 0:
                        writers_done.set()
                        stop_flag.set()

        def reader_fn(reader_id):
            try:
                while not stop_flag.is_set():
                    txn = store.begin()
                    values = []
                    for k in range(num_keys):
                        val = txn.get(f"k{k}")
                        values.append(val)
                    txn.rollback()

                    first_val = values[0]
                    for i, v in enumerate(values):
                        if v != first_val:
                            errors.append(
                                f"Reader {reader_id} saw INCONSISTENT values! "
                                f"k0={first_val}, k{i}={v}, all={values}"
                            )
                            stop_flag.set()
                            return

                    time.sleep(0.0005)
            except Exception as e:
                errors.append(f"Reader {reader_id} error: {e}")

        threads = []
        num_writers = 4
        for i in range(num_writers):
            t = threading.Thread(target=writer_fn, args=(i, 15))
            threads.append(t)
            t.start()

        num_readers = 6
        for i in range(num_readers):
            t = threading.Thread(target=reader_fn, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if errors:
            print(f"  Round {round_num} ERRORS:")
            for e in errors:
                print(f"    - {e}")
            raise AssertionError(f"Round {round_num} failed")

    print(f"[PASS] test_multi_key_atomic_with_conflicts ({num_rounds} rounds)")


# ============================================================================
# Test 3: Read-Write Non-Blocking Stress
# ============================================================================

def test_long_read_does_not_block_write():
    """
    Verify that a long-running read transaction does not block writers.

    A read transaction starts and holds its snapshot. Meanwhile, multiple
    writers commit to the same key. The writers must complete quickly,
    without waiting for the reader to finish.
    """
    store = MVCCStore()

    t_init = store.begin()
    t_init.put("hot", b"init")
    t_init.commit()

    long_reader = store.begin()
    assert long_reader.get("hot") == b"init"

    num_writers = 20
    write_times: List[float] = []
    errors: List[str] = []

    def writer_fn(writer_id):
        start = time.perf_counter()
        try:
            txn = store.begin()
            txn.put("hot", f"w{writer_id}".encode())
            try:
                txn.commit()
            except WriteConflictError:
                pass
        except Exception as e:
            errors.append(f"Writer {writer_id} error: {e}")
        finally:
            elapsed = time.perf_counter() - start
            write_times.append(elapsed)

    threads = []
    for i in range(num_writers):
        t = threading.Thread(target=writer_fn, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=5)

    for t in threads:
        if t.is_alive():
            errors.append("Writer thread hung!")

    max_write_time = max(write_times) if write_times else 0
    if max_write_time > 1.0:
        errors.append(
            f"Writers took too long! Max write time: {max_write_time:.3f}s, "
            f"suggesting they were blocked by the long reader."
        )

    assert long_reader.get("hot") == b"init", "Long reader's snapshot must be preserved"
    long_reader.rollback()

    if errors:
        for e in errors:
            print(f"  - {e}")
        raise AssertionError("test_long_read_does_not_block_write failed")

    print(f"[PASS] test_long_read_does_not_block_write (max write time: {max_write_time:.3f}s)")


def test_high_frequency_writes_dont_block_reads():
    """
    Verify that high-frequency writes don't prevent new read transactions
    from quickly acquiring their snapshot.

    Many writers are continuously committing. New readers must be able to
    start and read quickly, without noticeable delay.
    """
    store = MVCCStore()

    t_init = store.begin()
    t_init.put("hot", b"init")
    t_init.commit()

    stop_flag = threading.Event()
    writers_active = [0]
    counter_lock = threading.Lock()
    errors: List[str] = []
    read_times: List[float] = []

    def writer_fn(writer_id):
        try:
            with counter_lock:
                writers_active[0] += 1
            i = 0
            while not stop_flag.is_set():
                txn = store.begin()
                txn.put("hot", f"w{writer_id}_i{i}".encode())
                try:
                    txn.commit()
                except WriteConflictError:
                    pass
                i += 1
                time.sleep(0.0001)
        except Exception as e:
            errors.append(f"Writer {writer_id} error: {e}")
        finally:
            with counter_lock:
                writers_active[0] -= 1

    num_writers = 4
    writer_threads = []
    for i in range(num_writers):
        t = threading.Thread(target=writer_fn, args=(i,))
        writer_threads.append(t)
        t.start()

    time.sleep(0.1)

    num_readers = 30
    for i in range(num_readers):
        start = time.perf_counter()
        txn = store.begin()
        val = txn.get("hot")
        txn.rollback()
        elapsed = time.perf_counter() - start
        read_times.append(elapsed)

        if val is None:
            errors.append(f"Reader {i} got None!")

        if elapsed > 0.5:
            errors.append(
                f"Reader {i} took too long: {elapsed:.3f}s, "
                f"suggesting reads are blocked by writes."
            )

    stop_flag.set()
    for t in writer_threads:
        t.join(timeout=5)

    max_read_time = max(read_times)
    avg_read_time = sum(read_times) / len(read_times)

    if errors:
        for e in errors:
            print(f"  - {e}")
        raise AssertionError("test_high_frequency_writes_dont_block_reads failed")

    print(
        f"[PASS] test_high_frequency_writes_dont_block_reads "
        f"(max read: {max_read_time*1000:.1f}ms, avg read: {avg_read_time*1000:.1f}ms)"
    )


def test_read_write_stress_mixed(num_rounds: int = 5, duration: float = 2.0):
    """
    Mixed read-write stress test with multiple hot keys.
    Many readers and writers operating concurrently.
    Verifies:
    - No deadlocks
    - No data corruption
    - Writes don't starve reads, reads don't starve writes
    """
    for round_num in range(num_rounds):
        store = MVCCStore()
        num_keys = 10

        for k in range(num_keys):
            t = store.begin()
            t.put(f"k{k}", b"v0")
            t.commit()

        stop_flag = threading.Event()
        errors: List[str] = []
        stats_lock = threading.Lock()
        successful_writes = [0]
        conflicted_writes = [0]
        successful_reads = [0]

        def writer_fn(writer_id):
            try:
                i = 0
                while not stop_flag.is_set():
                    key_idx = i % num_keys
                    key = f"k{key_idx}"
                    txn = store.begin()
                    txn.put(key, f"w{writer_id}_v{i}".encode())
                    try:
                        txn.commit()
                        with stats_lock:
                            successful_writes[0] += 1
                    except WriteConflictError:
                        with stats_lock:
                            conflicted_writes[0] += 1
                    i += 1
                    time.sleep(0.0002)
            except Exception as e:
                errors.append(f"Writer {writer_id} error: {e}")

        def reader_fn(reader_id):
            try:
                while not stop_flag.is_set():
                    key_idx = random.randint(0, num_keys - 1)
                    key = f"k{key_idx}"
                    txn = store.begin()
                    val = txn.get(key)
                    txn.rollback()
                    if val is None:
                        errors.append(f"Reader {reader_id} got None for {key}!")
                    with stats_lock:
                        successful_reads[0] += 1
                    time.sleep(0.0005)
            except Exception as e:
                errors.append(f"Reader {reader_id} error: {e}")

        threads = []
        num_writers = 6
        num_readers = 8

        for i in range(num_writers):
            t = threading.Thread(target=writer_fn, args=(i,))
            threads.append(t)
            t.start()

        for i in range(num_readers):
            t = threading.Thread(target=reader_fn, args=(i,))
            threads.append(t)
            t.start()

        time.sleep(duration)
        stop_flag.set()

        for t in threads:
            t.join(timeout=10)

        for t in threads:
            if t.is_alive():
                errors.append("Thread hung!")

        if errors:
            print(f"  Round {round_num} ERRORS:")
            for e in errors:
                print(f"    - {e}")
            raise AssertionError(f"Round {round_num} failed")

        print(
            f"  Round {round_num}: {successful_writes[0]} writes ok, "
            f"{conflicted_writes[0]} conflicts, "
            f"{successful_reads[0]} reads ok"
        )

    print(f"[PASS] test_read_write_stress_mixed ({num_rounds} rounds, {duration}s each)")


if __name__ == "__main__":
    print("=" * 70)
    print("CONCURRENCY STRESS TESTS")
    print("=" * 70)

    print("\n--- Test Suite 1: Same-Key Write-Write Conflict ---")
    test_same_key_conflict_stress(num_rounds=20, num_threads=10)
    test_same_key_conflict_many_keys(num_rounds=10, num_threads=6, num_keys=5)

    print("\n--- Test Suite 2: Multi-Key Atomic Visibility ---")
    test_multi_key_atomic_visibility(num_rounds=50)
    test_multi_key_atomic_with_conflicts(num_rounds=30)

    print("\n--- Test Suite 3: Read-Write Non-Blocking ---")
    test_long_read_does_not_block_write()
    test_high_frequency_writes_dont_block_reads()
    test_read_write_stress_mixed(num_rounds=5, duration=2.0)

    print("\n" + "=" * 70)
    print("ALL CONCURRENCY TESTS PASSED")
    print("=" * 70)
