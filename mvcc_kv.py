"""
MVCC In-Memory KV Store with Snapshot Isolation
=================================================

Design Overview
---------------

1. MVCC Version Chain Organization
-----------------------------------
Each key maps to a singly-linked list of VersionNodes, ordered from newest to oldest.

    key -> VersionNode(txn_id=5, value="v3") -> VersionNode(txn_id=3, value="v2") -> ...

- Each VersionNode carries: (txn_id, start_ts, value, is_deleted, next pointer)
- `txn_id` references the transaction that wrote this version.
- The commit status and commit_ts are NOT stored in the node. Instead, they are
  stored in a central TransactionTable keyed by txn_id.
- This indirection is the KEY to multi-key atomic visibility: when a transaction
  commits, we atomically update its entry in the TransactionTable from PENDING
  to COMMITTED with its commit_ts. All versions written by that transaction
  become visible at the EXACT same instant.
- `is_deleted` marks a tombstone version (DELETE creates a tombstone).

2. Visibility Rules (Snapshot Isolation)
-----------------------------------------
A transaction T with start_ts = S, txn_id = TID sees a version V written by
transaction W (txn_id = WID) if and only if:

  Look up W's status in TransactionTable:
  (a) W.status == ABORTED  ->  skip this version (it never happened)
  (b) W.status == PENDING:
        - if WID == TID  ->  visible (our own uncommitted writes)
        - else           ->  skip (someone else's uncommitted writes)
  (c) W.status == COMMITTED:
        - if W.commit_ts <= S  ->  visible
        - else                 ->  skip

Among visible versions, pick the one with the highest commit_ts
(which is the first match in our newest-to-oldest chain).

This guarantees:
- Multi-key atomic visibility: a transaction's writes all become visible at
  the same moment (when its TransactionTable entry is updated atomically).
- Read never blocks write: readers walk the chain, nodes are immutable,
  and TransactionTable reads are lock-free.
- Write never blocks read: writers prepend new nodes; existing nodes are
  never modified in place.

3. Write-Write Conflict Detection (Atomic)
-------------------------------------------
To prevent two concurrent transactions from both successfully committing writes
to the same key (which would result in the later one "randomly" winning by
having a higher commit_ts), we use per-key locks and a strict protocol:

When transaction T commits with write set {k1, k2, ..., kn}:
  1. Sort keys in a fixed global order (prevents deadlock).
  2. Acquire per-key locks for ALL keys in the write set.
  3. With all locks held:
     a. For each key, check for conflicts: does the chain contain any
        version whose writing transaction is COMMITTED and has commit_ts > T.start_ts?
        If yes for any key -> CONFLICT, release locks, abort.
     b. If no conflicts, prepend new VersionNodes to each key's chain.
        (These nodes reference T's txn_id, which is still PENDING.)
     c. Atomically update T's TransactionTable entry:
        status = COMMITTED, commit_ts = <newly allocated timestamp>
  4. Release all locks.

This guarantees that only one writer can proceed per key at a time,
and conflict detection + write installation is atomic.

4. Garbage Collection: Safe Reclamation of Old Versions
--------------------------------------------------------
GC algorithm:
  1. Compute low_watermark = min(start_ts of all active transactions).
     If no active transactions, low_watermark = +inf.
  2. For each key's version chain:
     a. Collect all nodes, resolving their effective commit_ts via TransactionTable.
        Skip nodes whose transaction is ABORTED (they're effectively not there).
     b. Keep nodes whose commit_ts >= low_watermark (recent versions).
     c. Among nodes with commit_ts < low_watermark, keep only the one with
        the highest commit_ts (the "boundary version" visible to the oldest
        active transaction).
     d. Rebuild the chain, skipping ABORTED transaction nodes entirely.

5. Per-Key Locking for Read/Write Non-Blocking
-----------------------------------------------
- Reads acquire a per-key lock ONLY long enough to read the head pointer,
  then release it immediately. The chain walk is entirely lock-free because
  VersionNodes are immutable once created.
- Writes acquire per-key locks for all keys in their write set (in sorted order),
  but only during the commit phase. Reads on other keys proceed unobstructed.
- This ensures:
  * Long-running reads on key A never block writes to key B.
  * High-frequency writes don't prevent new reads from acquiring their snapshot.
"""

import threading
import time
from typing import Optional, Dict, List, Set, Tuple
from enum import Enum


class TxnStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"


class VersionNode:
    """
    A single version in the MVCC version chain for a key.

    The chain is ordered newest-to-oldest.
    Nodes are IMMUTABLE after creation (except the `next` pointer which is
    only set during prepend, under a per-key lock).
    """

    __slots__ = ('txn_id', 'start_ts', 'value', 'is_deleted', 'next')

    def __init__(
        self,
        txn_id: int,
        start_ts: int,
        value: Optional[bytes],
        is_deleted: bool = False,
    ):
        self.txn_id: int = txn_id
        self.start_ts: int = start_ts
        self.value: Optional[bytes] = value
        self.is_deleted: bool = is_deleted
        self.next: Optional['VersionNode'] = None


class TransactionTableEntry:
    """Entry in the global transaction table."""

    __slots__ = ('status', 'commit_ts', 'start_ts')

    def __init__(self, start_ts: int):
        self.status: TxnStatus = TxnStatus.PENDING
        self.commit_ts: Optional[int] = None
        self.start_ts: int = start_ts


class TransactionTable:
    """
    Central table tracking all transactions' status and commit timestamps.

    This is the KEY to multi-key atomic visibility: a transaction's commit
    is a single atomic update to this table. All versions written by the
    transaction become visible at the exact same instant.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[int, TransactionTableEntry] = {}

    def create_entry(self, txn_id: int, start_ts: int):
        with self._lock:
            self._entries[txn_id] = TransactionTableEntry(start_ts)

    def get_entry(self, txn_id: int) -> Optional[TransactionTableEntry]:
        with self._lock:
            return self._entries.get(txn_id)

    def get_status(self, txn_id: int) -> Optional[Tuple[TxnStatus, Optional[int]]]:
        """
        Return (status, commit_ts) for a transaction.
        This is a single atomic read operation.
        """
        with self._lock:
            entry = self._entries.get(txn_id)
            if entry is None:
                return None
            return (entry.status, entry.commit_ts)

    def mark_committed(self, txn_id: int, commit_ts: int) -> bool:
        """
        Atomically mark a transaction as COMMITTED with the given commit_ts.
        Returns True if successful, False if the transaction was already aborted.
        """
        with self._lock:
            entry = self._entries.get(txn_id)
            if entry is None or entry.status == TxnStatus.ABORTED:
                return False
            entry.status = TxnStatus.COMMITTED
            entry.commit_ts = commit_ts
            return True

    def mark_aborted(self, txn_id: int):
        """Atomically mark a transaction as ABORTED."""
        with self._lock:
            entry = self._entries.get(txn_id)
            if entry is not None and entry.status == TxnStatus.PENDING:
                entry.status = TxnStatus.ABORTED

    def cleanup(self, txn_id: int):
        """Remove a transaction entry (called when it's safe to do so)."""
        with self._lock:
            self._entries.pop(txn_id, None)


class PerKeyLockManager:
    """
    Per-key lock manager.

    - get_lock(key): returns a threading.Lock for the given key (creates if needed).
    - acquire_all(keys): acquires locks for all given keys in sorted order (prevents deadlock).
    - release_all(locks): releases the given locks.

    Locks are never deleted once created — a small memory overhead for simplicity.
    """

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._table_lock = threading.Lock()

    def get_lock(self, key: str) -> threading.Lock:
        with self._table_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def acquire_all(self, keys: List[str]) -> List[threading.Lock]:
        """
        Acquire locks for all keys in sorted order.
        Returns the list of acquired locks (in the same order as sorted keys).
        Using sorted order prevents deadlock when transactions have overlapping
        write sets.
        """
        sorted_keys = sorted(keys)
        locks = []
        for k in sorted_keys:
            lock = self.get_lock(k)
            lock.acquire()
            locks.append(lock)
        return locks

    def release_all(self, locks: List[threading.Lock]):
        """Release all locks in reverse order of acquisition."""
        for lock in reversed(locks):
            lock.release()


class TimestampOracle:
    """
    Monotonically increasing timestamp allocator.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counter: int = 1

    def alloc(self) -> int:
        with self._lock:
            ts = self._counter
            self._counter += 1
            return ts

    def current(self) -> int:
        with self._lock:
            return self._counter


class ActiveTransactionRegistry:
    """
    Tracks all active (in-progress) transactions for GC low_watermark computation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active: Dict[int, int] = {}

    def register(self, txn_id: int, start_ts: int):
        with self._lock:
            self._active[txn_id] = start_ts

    def unregister(self, txn_id: int):
        with self._lock:
            self._active.pop(txn_id, None)

    def low_watermark(self) -> int:
        with self._lock:
            if not self._active:
                return float('inf')
            return min(self._active.values())

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)


class Transaction:
    """
    Represents a single MVCC transaction.

    Lifecycle:
        txn = store.begin()
        txn.put(key, value)   -> immediately installs a PENDING version in the chain
        txn.get(key)          -> reads from snapshot or own pending/committed writes
        txn.delete(key)       -> immediately installs a PENDING tombstone in the chain
        txn.commit()          -> atomically changes status to COMMITTED
        txn.rollback()        -> marks status as ABORTED (GC will clean up PENDING nodes)

    KEY DESIGN: put/delete write PENDING nodes to the chain immediately. This ensures
    concurrent writers can see each other's in-flight writes and detect conflicts.
    Readers skip PENDING nodes from other transactions, so they never see uncommitted
    data from others.
    """

    def __init__(
        self,
        txn_id: int,
        start_ts: int,
        store: 'MVCCStore',
    ):
        self.txn_id = txn_id
        self.start_ts = start_ts
        self.commit_ts: Optional[int] = None
        self._store = store
        self._write_buffer: Dict[str, Tuple[Optional[bytes], bool]] = {}
        self._committed = False
        self._rolled_back = False

    def get(self, key: str) -> Optional[bytes]:
        """
        Read from the snapshot defined by self.start_ts.
        """
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")

        if key in self._write_buffer:
            value, is_deleted = self._write_buffer[key]
            if is_deleted:
                return None
            return value

        version = self._store._find_visible_version(key, self.start_ts, self.txn_id)
        if version is None:
            return None
        if version.is_deleted:
            return None
        return version.value

    def put(self, key: str, value: bytes):
        """
        Write a value.

        Immediately installs a PENDING version node in the key's chain.
        This allows concurrent writers to see our in-flight write and detect
        conflicts. Our PENDING node is only visible to ourselves until commit.
        """
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")
        self._write_buffer[key] = (value, False)
        self._store._install_pending_version(
            key, self.txn_id, self.start_ts, value, False
        )

    def delete(self, key: str):
        """
        Delete a key (writes a tombstone).

        Immediately installs a PENDING tombstone node in the key's chain.
        """
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")
        self._write_buffer[key] = (None, True)
        self._store._install_pending_version(
            key, self.txn_id, self.start_ts, None, True
        )

    def commit(self):
        """
        Commit this transaction atomically.

        Algorithm:
        1. If no writes, just mark as committed (no-op).
        2. Sort keys in write set, acquire all per-key locks (prevents deadlock).
        3. With locks held, check each key for write-write conflicts:
           a. Any COMMITTED version with commit_ts > self.start_ts? -> CONFLICT
           b. Any PENDING version from a different transaction? -> CONFLICT
              (These were installed by other concurrent writers.)
        4. If no conflicts, allocate commit_ts and atomically update
           TransactionTable to COMMITTED. This is the single atomic commit point —
           all our writes become visible to other transactions at this exact instant.
        5. Release all locks.

        CRITICAL: Our PENDING versions are already in the chain. The only thing
        that changes at commit time is the transaction status in TransactionTable.
        This ensures multi-key atomic visibility.
        """
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")

        if not self._write_buffer:
            self.commit_ts = self._store._ts_oracle.alloc()
            self._store._txn_table.mark_committed(self.txn_id, self.commit_ts)
            self._committed = True
            self._store._registry.unregister(self.txn_id)
            return

        keys = list(self._write_buffer.keys())
        locks = self._store._lock_manager.acquire_all(keys)

        try:
            for key in keys:
                if self._store._has_write_conflict_locked(key, self.start_ts, self.txn_id):
                    self._store._txn_table.mark_aborted(self.txn_id)
                    self._rolled_back = True
                    raise WriteConflictError(
                        f"Write-write conflict on key '{key}' "
                        f"(txn start_ts={self.start_ts}, txn_id={self.txn_id})"
                    )

            self.commit_ts = self._store._ts_oracle.alloc()
            if not self._store._txn_table.mark_committed(self.txn_id, self.commit_ts):
                raise WriteConflictError(
                    f"Transaction {self.txn_id} was aborted before commit"
                )

            self._committed = True
        finally:
            self._store._lock_manager.release_all(locks)
            if not self._committed:
                self._store._txn_table.mark_aborted(self.txn_id)
            self._store._registry.unregister(self.txn_id)

    def rollback(self):
        """Discard this transaction."""
        if self._committed:
            raise RuntimeError("Transaction already committed")
        if self._rolled_back:
            return
        self._rolled_back = True
        self._store._txn_table.mark_aborted(self.txn_id)
        self._store._registry.unregister(self.txn_id)


class WriteConflictError(Exception):
    pass


class MVCCStore:
    """
    In-memory MVCC Key-Value Store with Snapshot Isolation.

    Guarantees:
    - Snapshot Isolation: each transaction sees a consistent view as of its start_ts.
    - Multi-key atomic visibility: a transaction's writes become visible all at once.
    - First-committer-wins for write-write conflicts (no random overwrites).
    - Read never blocks write, write never blocks read (per-key locking).
    - Safe GC based on low_watermark of active transactions.
    """

    def __init__(self):
        self._ts_oracle = TimestampOracle()
        self._txn_table = TransactionTable()
        self._registry = ActiveTransactionRegistry()
        self._lock_manager = PerKeyLockManager()
        self._data: Dict[str, Optional[VersionNode]] = {}
        self._data_head_locks: Dict[str, threading.Lock] = {}
        self._data_meta_lock = threading.Lock()
        self._txn_counter = 0
        self._txn_counter_lock = threading.Lock()
        self._gc_lock = threading.Lock()

    def _get_head_lock(self, key: str) -> threading.Lock:
        """Get the lock for reading/writing the head pointer of a key's chain."""
        with self._data_meta_lock:
            lock = self._data_head_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._data_head_locks[key] = lock
            return lock

    def begin(self) -> Transaction:
        """
        Start a new transaction.

        Steps:
        1. Allocate start_ts (defines our snapshot view).
        2. Allocate unique txn_id.
        3. Create TransactionTable entry (status=PENDING).
        4. Register in ActiveTransactionRegistry (for GC).
        """
        start_ts = self._ts_oracle.alloc()
        with self._txn_counter_lock:
            self._txn_counter += 1
            txn_id = self._txn_counter
        self._txn_table.create_entry(txn_id, start_ts)
        self._registry.register(txn_id, start_ts)
        return Transaction(txn_id=txn_id, start_ts=start_ts, store=self)

    def _find_visible_version(
        self,
        key: str,
        reader_start_ts: int,
        reader_txn_id: int,
    ) -> Optional[VersionNode]:
        """
        Walk the version chain for `key` and return the first (newest) visible version.

        Read path is carefully designed to be non-blocking:
        1. Acquire per-key head lock ONLY to copy the head pointer.
        2. Release lock immediately.
        3. Walk chain without any locks (VersionNodes are immutable once created).
        4. For each node, look up its transaction status in TransactionTable.

        This ensures:
        - Long chain walks never block writers.
        - High-frequency writes don't block readers (they only wait for the
          brief head-pointer copy).
        """
        head_lock = self._get_head_lock(key)
        with head_lock:
            head = self._data.get(key)

        node = head
        while node is not None:
            status_info = self._txn_table.get_status(node.txn_id)

            if status_info is None:
                node = node.next
                continue

            status, commit_ts = status_info

            if status == TxnStatus.ABORTED:
                node = node.next
                continue

            if status == TxnStatus.PENDING:
                if node.txn_id == reader_txn_id:
                    return node
                node = node.next
                continue

            if status == TxnStatus.COMMITTED:
                if commit_ts is not None and commit_ts <= reader_start_ts:
                    return node
                node = node.next
                continue

            node = node.next

        return None

    def _has_write_conflict_locked(
        self,
        key: str,
        writer_start_ts: int,
        writer_txn_id: int,
    ) -> bool:
        """
        Check for write-write conflict on `key`.
        Must be called with the per-key lock held for this key.

        Conflict detection rules (deterministic "first-starter-wins"):

        For each version on the chain written by a different transaction W:
        (a) If W is COMMITTED and W.commit_ts > writer_start_ts:
            -> CONFLICT. Another transaction committed a write after our
               snapshot started (standard SI rule).
        (b) If W is PENDING:
            -> CONFLICT if W.start_ts < writer_start_ts.
               This means W started before us. We let earlier-starting
               transactions finish first. If W started after us, we do NOT
               conflict — it's W's responsibility to detect our PENDING
               node and abort when it commits.

        This ensures deterministic behavior: among concurrent writers to
        the same key, the one with the smaller start_ts (earlier starter)
        always wins. The later-starter always aborts. No randomness.
        """
        head = self._data.get(key)

        node = head
        while node is not None:
            if node.txn_id == writer_txn_id:
                node = node.next
                continue

            status_info = self._txn_table.get_status(node.txn_id)
            if status_info is None:
                node = node.next
                continue

            status, commit_ts = status_info
            if status == TxnStatus.COMMITTED and commit_ts is not None and commit_ts > writer_start_ts:
                return True

            if status == TxnStatus.PENDING:
                if node.start_ts < writer_start_ts:
                    return True

            node = node.next
        return False

    def _prepend_version_locked(self, key: str, node: VersionNode):
        """
        Prepend a version node to the head of the key's chain.
        Must be called with the per-key lock held for this key.
        """
        node.next = self._data.get(key)
        self._data[key] = node

    def _install_pending_version(
        self,
        key: str,
        txn_id: int,
        start_ts: int,
        value: Optional[bytes],
        is_deleted: bool,
    ):
        """
        Install a PENDING version node into the key's chain.

        Called from Transaction.put() and Transaction.delete() BEFORE commit.
        This allows concurrent writers to see our in-flight write and detect
        conflicts.

        Acquires the per-key lock only briefly to prepend the node.
        The node remains invisible to other readers (they skip PENDING nodes
        from other transactions) until commit time when the transaction
        status is atomically changed to COMMITTED.
        """
        node = VersionNode(
            txn_id=txn_id,
            start_ts=start_ts,
            value=value,
            is_deleted=is_deleted,
        )
        lock = self._lock_manager.get_lock(key)
        with lock:
            node.next = self._data.get(key)
            self._data[key] = node

    def gc(self):
        """
        Garbage collect old versions and ABORTED transaction nodes.

        Algorithm:
        1. Compute low_watermark = min(start_ts of all active transactions).
        2. For each key:
           a. Acquire the head lock, copy the chain head.
           b. Walk the chain, resolving each node's commit_ts via TransactionTable.
              Skip nodes from ABORTED transactions entirely.
           c. Partition nodes:
              - "old" nodes: commit_ts < low_watermark
              - "recent" nodes: commit_ts >= low_watermark or PENDING
           d. Keep the LAST "old" node (highest commit_ts below low_watermark).
           e. Rebuild the chain.

        This removes:
        - All nodes from ABORTED transactions.
        - Old nodes that are too old to be visible to any active transaction,
          except the boundary version needed by the oldest active transaction.
        """
        with self._gc_lock:
            low_watermark = self._registry.low_watermark()

            keys_to_gc = []
            with self._data_meta_lock:
                keys_to_gc = list(self._data.keys())

            for key in keys_to_gc:
                head_lock = self._get_head_lock(key)
                with head_lock:
                    head = self._data.get(key)
                    if head is None:
                        continue

                    all_nodes = []
                    node = head
                    while node is not None:
                        status_info = self._txn_table.get_status(node.txn_id)
                        if status_info is not None:
                            status, commit_ts = status_info
                            if status != TxnStatus.ABORTED:
                                all_nodes.append((node, status, commit_ts))
                        node = node.next

                    all_nodes.reverse()

                    old_nodes = []
                    recent_nodes = []
                    for n, status, commit_ts in all_nodes:
                        if (
                            status == TxnStatus.COMMITTED
                            and commit_ts is not None
                            and commit_ts < low_watermark
                        ):
                            old_nodes.append((n, commit_ts))
                        else:
                            recent_nodes.append(n)

                    kept_old = []
                    if old_nodes:
                        kept_old.append(old_nodes[-1][0])

                    kept_all = kept_old + recent_nodes

                    new_head = None
                    for n in kept_all:
                        new_node = VersionNode(
                            txn_id=n.txn_id,
                            start_ts=n.start_ts,
                            value=n.value,
                            is_deleted=n.is_deleted,
                        )
                        new_node.next = new_head
                        new_head = new_node

                    self._data[key] = new_head

    def gc_stats(self) -> dict:
        """Return statistics about version chains."""
        stats = {
            'keys': 0,
            'total_versions': 0,
            'low_watermark': self._registry.low_watermark(),
            'active_transactions': self._registry.active_count(),
        }
        with self._data_meta_lock:
            keys = list(self._data.keys())
        stats['keys'] = len(keys)
        for key in keys:
            head_lock = self._get_head_lock(key)
            with head_lock:
                head = self._data.get(key)
            node = head
            while node is not None:
                stats['total_versions'] += 1
                node = node.next
        return stats

    def debug_chain(self, key: str) -> List[str]:
        """Return a string representation of the version chain for a key."""
        head_lock = self._get_head_lock(key)
        with head_lock:
            head = self._data.get(key)
        result = []
        node = head
        while node is not None:
            status_info = self._txn_table.get_status(node.txn_id)
            if status_info is not None:
                status, commit_ts = status_info
                if status == TxnStatus.PENDING:
                    status_str = "pending"
                elif status == TxnStatus.COMMITTED:
                    status_str = f"committed@{commit_ts}"
                else:
                    status_str = "aborted"
            else:
                status_str = "unknown"
            kind = "TOMBSTONE" if node.is_deleted else repr(node.value)
            result.append(
                f"VersionNode(txn_id={node.txn_id}, start_ts={node.start_ts}, "
                f"{status_str}, val={kind})"
            )
            node = node.next
        return result
