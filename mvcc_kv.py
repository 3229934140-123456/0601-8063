"""
MVCC In-Memory KV Store with Snapshot Isolation
=================================================

Design Overview
---------------

1. MVCC Version Chain Organization
-----------------------------------
Each key maps to a singly-linked list of VersionNodes, ordered from newest to oldest.

    key -> VersionNode(commit_ts=30, value="v3") -> VersionNode(commit_ts=20, value="v2") -> VersionNode(commit_ts=10, value="v1")

- Each VersionNode carries: (commit_ts, start_ts, value, is_deleted, next pointer)
- `commit_ts` is set when the writing transaction commits; before commit it is None (uncommitted).
- `start_ts` is the timestamp of the transaction that wrote this version.
- `is_deleted` marks a tombstone version (DELETE creates a tombstone).
- The chain head is always the latest committed or uncommitted version.

Read path: walk the chain, skip uncommitted versions (commit_ts is None) that don't belong
to the current transaction, then pick the first version whose commit_ts <= reader's start_ts.

Write path: a write under transaction T appends a new VersionNode at the head with commit_ts=None.
On commit, atomically set commit_ts = commit_timestamp. This is the "commit marker".

2. Visibility Rules (Snapshot Isolation)
-----------------------------------------
A transaction T with start_ts = S sees a version V if and only if:
  (a) V.commit_ts is not None   (the version is committed)
  (b) V.commit_ts <= S          (committed before T's snapshot)
  (c) If multiple versions satisfy (a)(b), pick the one with the largest commit_ts
      (which is the first match in our newest-to-oldest chain)

Additionally, a transaction always sees its own uncommitted writes:
  (d) V.start_ts == T.start_ts and V.commit_ts is None

This guarantees:
- Read never blocks write: readers only walk the chain, no locks needed on the chain
  beyond a brief read-lock on the head pointer.
- Write never blocks read: writers prepend new nodes; existing nodes are immutable
  once committed (commit_ts only goes from None to a value, never changes after).

3. Garbage Collection: Safe Reclamation of Old Versions
--------------------------------------------------------
THE CORE PROBLEM: When can we remove an old version from the chain?

A version V_old with commit_ts = C can be safely removed if and only if:
  (1) There exists a newer version V_new on the same key's chain such that
      V_new.commit_ts > C  (so V_old is not the latest version for any snapshot)
  AND
  (2) No active transaction T has T.start_ts >= C  AND  T.start_ts < V_new.commit_ts
      In other words, no active transaction's snapshot falls in the window
      [C, V_new.commit_ts) where V_old would be the visible version.

Simplified rule: A version V_old with commit_ts = C is removable when:
  - There is a newer committed version V_new on the chain, AND
  - C < low_watermark, where low_watermark = min(start_ts of all active transactions)

  If C < low_watermark, then every active transaction has start_ts >= low_watermark > C,
  and they will all see V_new or something even newer, never V_old.

low_watermark (GC watermark) tracking:
  - We maintain an ActiveTransactionRegistry that records each transaction's start_ts
    at Begin() and removes it at Commit()/Rollback().
  - low_watermark = min(all start_ts in the registry), or +inf if no active transactions.
  - During GC, we compute low_watermark and remove versions where commit_ts < low_watermark
    AND there exists a newer committed version on the same chain.

EDGE CASES:
  - A long-running transaction T_old with start_ts = 5 keeps low_watermark at 5.
    Versions committed at ts=4 or earlier can only be GC'd if a newer version exists
    AND commit_ts < 5. But we must NOT remove the latest version visible to T_old.
    So we only remove versions that have a NEWER committed successor on the chain.
    The newest version visible to T_old (commit_ts <= 5) is always preserved because
    it has no newer committed version that T_old's snapshot would skip.
  - Tombstones (is_deleted=True): treated the same as regular versions. A tombstone
    with commit_ts < low_watermark that has a newer committed version can be GC'd.
    But if the tombstone IS the latest version, it must be kept so that readers
    know the key doesn't exist.

4. Concurrency Guarantees
--------------------------
- Read never blocks write: Get() walks the version chain without taking any lock
  on the chain nodes. Only a brief lock on the chain head is needed to get the
  first node pointer.
- Write never blocks read: Put()/Delete() prepend a new node at the head.
  Existing nodes are never modified in place (commit_ts goes None->value once).
- Write-write conflict: detected at commit time. If two transactions write to the
  same key, the second one to commit detects a write-write conflict (a version
  with commit_ts > self.start_ts exists on the key's chain) and aborts.
"""

import threading
import time
from typing import Optional, Dict, List, Set


class VersionNode:
    """
    A single version in the MVCC version chain for a key.

    The chain is a singly-linked list ordered from newest to oldest.
    Once a node is committed (commit_ts is set), it is immutable.
    """

    __slots__ = ('commit_ts', 'start_ts', 'value', 'is_deleted', 'next')

    def __init__(
        self,
        start_ts: int,
        value: Optional[bytes],
        is_deleted: bool = False,
    ):
        self.commit_ts: Optional[int] = None
        self.start_ts: int = start_ts
        self.value: Optional[bytes] = value
        self.is_deleted: bool = is_deleted
        self.next: Optional['VersionNode'] = None

    def __repr__(self):
        status = "uncommitted" if self.commit_ts is None else f"committed@{self.commit_ts}"
        kind = "TOMBSTONE" if self.is_deleted else repr(self.value)
        return f"VersionNode(start_ts={self.start_ts}, {status}, val={kind})"


class TimestampOracle:
    """
    Monotonically increasing timestamp allocator.

    Uses a simple global counter protected by a lock.
    - alloc() returns the next timestamp and advances the counter.
    - current() returns the current timestamp without advancing.
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
    Tracks all active (in-progress) transactions and their start timestamps.

    This is the critical component for determining the GC low watermark:
        low_watermark = min(start_ts of all active transactions)

    When no transactions are active, low_watermark is +inf, meaning all
    old versions with a newer successor can be garbage collected.
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
        """
        Return the minimum start_ts among all active transactions.
        If no transactions are active, return float('inf').

        Any version with commit_ts < low_watermark AND a newer committed
        version exists on its chain can be safely garbage collected.
        """
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
        txn.put(key, value)   # buffer a write
        txn.get(key)          # read from snapshot or own writes
        txn.delete(key)       # buffer a delete
        txn.commit()          # attempt commit; raises on write-write conflict
        # OR
        txn.rollback()        # discard

    Key timestamps:
        start_ts  - assigned at Begin(), defines the snapshot view
        commit_ts - assigned at Commit() only if commit succeeds
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
        self._write_buffer: Dict[str, tuple] = {}
        self._committed = False
        self._rolled_back = False

    def get(self, key: str) -> Optional[bytes]:
        """
        Read from the snapshot defined by self.start_ts.

        Steps:
        1. Check own write buffer first (uncommitted writes are visible to self).
        2. Walk the key's version chain to find the latest committed version
           with commit_ts <= self.start_ts.
        3. If the visible version is a tombstone, return None (key doesn't exist).
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

        version = self._store._find_visible_version(key, self.start_ts, self.start_ts)
        if version is None:
            return None
        if version.is_deleted:
            return None
        return version.value

    def put(self, key: str, value: bytes):
        """Buffer a write. The write is only visible to this transaction until commit."""
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")
        self._write_buffer[key] = (value, False)

    def delete(self, key: str):
        """Buffer a delete (tombstone). Only takes effect on commit."""
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")
        self._write_buffer[key] = (None, True)

    def commit(self):
        """
        Commit this transaction.

        Algorithm:
        1. Allocate a commit_ts from the TimestampOracle.
        2. For each key in the write buffer, check for write-write conflicts:
           if any version on the key's chain has commit_ts in (self.start_ts, +inf),
           another transaction committed a write after our snapshot started -> CONFLICT.
        3. If no conflicts, for each write, create a VersionNode at the head of
           the key's chain with commit_ts = commit_ts (immediately committed).
        4. Unregister from the ActiveTransactionRegistry.

        Write-write conflict detection is first-committer-wins.
        """
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back")
        if self._committed:
            raise RuntimeError("Transaction already committed")

        if not self._write_buffer:
            self.commit_ts = self._store._ts_oracle.alloc()
            self._committed = True
            self._store._registry.unregister(self.txn_id)
            return

        self.commit_ts = self._store._ts_oracle.alloc()

        for key in self._write_buffer:
            if self._store._has_write_conflict(key, self.start_ts):
                self._store._registry.unregister(self.txn_id)
                self._rolled_back = True
                raise WriteConflictError(
                    f"Write-write conflict on key '{key}' "
                    f"(txn start_ts={self.start_ts}, commit_ts={self.commit_ts})"
                )

        for key, (value, is_deleted) in self._write_buffer.items():
            node = VersionNode(
                start_ts=self.start_ts,
                value=value,
                is_deleted=is_deleted,
            )
            node.commit_ts = self.commit_ts
            self._store._prepend_version(key, node)

        self._committed = True
        self._store._registry.unregister(self.txn_id)

    def rollback(self):
        """Discard this transaction. Remove from active registry."""
        if self._committed:
            raise RuntimeError("Transaction already committed")
        self._rolled_back = True
        self._store._registry.unregister(self.txn_id)


class WriteConflictError(Exception):
    pass


class MVCCStore:
    """
    In-memory MVCC Key-Value Store with Snapshot Isolation.

    Guarantees:
    - Snapshot Isolation: each transaction sees a consistent view as of its start_ts.
    - Read never blocks write, write never blocks read.
    - Write-write conflicts are detected at commit time (first-committer-wins).
    - Old versions are garbage collected based on the low_watermark of active transactions.
    """

    def __init__(self):
        self._ts_oracle = TimestampOracle()
        self._registry = ActiveTransactionRegistry()
        self._data: Dict[str, Optional[VersionNode]] = {}
        self._data_lock = threading.Lock()
        self._txn_counter = 0
        self._gc_lock = threading.Lock()

    def begin(self) -> Transaction:
        """
        Start a new transaction.

        1. Allocate a start_ts from the TimestampOracle.
        2. Allocate a unique txn_id.
        3. Register (txn_id, start_ts) in the ActiveTransactionRegistry.
           This registration is what enables GC to know about this transaction
           and keep the low_watermark below our start_ts.
        """
        start_ts = self._ts_oracle.alloc()
        with self._data_lock:
            self._txn_counter += 1
            txn_id = self._txn_counter
        self._registry.register(txn_id, start_ts)
        return Transaction(txn_id=txn_id, start_ts=start_ts, store=self)

    def _find_visible_version(
        self,
        key: str,
        start_ts: int,
        own_start_ts: int,
    ) -> Optional[VersionNode]:
        """
        Walk the version chain for `key` and return the first (newest) version
        that is visible to a reader with snapshot timestamp `start_ts`.

        Visibility rules:
        - A committed version is visible if commit_ts <= start_ts.
        - An uncommitted version is visible only if start_ts == own_start_ts
          (i.e., it's our own write).
        - Among visible versions, pick the one with the highest commit_ts
          (which is the first match in the newest-to-oldest chain).
        """
        with self._data_lock:
            head = self._data.get(key)

        node = head
        while node is not None:
            if node.commit_ts is not None:
                if node.commit_ts <= start_ts:
                    return node
            else:
                if node.start_ts == own_start_ts:
                    return node
            node = node.next

        return None

    def _has_write_conflict(self, key: str, txn_start_ts: int) -> bool:
        """
        Check for write-write conflict on `key`.

        A conflict exists if any version on the chain has:
            commit_ts > txn_start_ts
        This means another transaction committed a write to this key
        after our transaction's snapshot was taken.
        """
        with self._data_lock:
            head = self._data.get(key)

        node = head
        while node is not None:
            if node.commit_ts is not None and node.commit_ts > txn_start_ts:
                return True
            node = node.next
        return False

    def _prepend_version(self, key: str, node: VersionNode):
        """Atomically prepend a version node to the head of the key's chain."""
        with self._data_lock:
            node.next = self._data.get(key)
            self._data[key] = node

    def gc(self):
        """
        Garbage collect old versions that are no longer visible to any active transaction.

        Algorithm:
        1. Compute low_watermark = min(start_ts of all active transactions).
           If no active transactions, low_watermark = +inf (all old versions are candidates).
        2. For each key's version chain:
           a. Collect all nodes into a list (oldest first).
           b. Partition nodes into two groups:
              - "old" nodes: commit_ts < low_watermark (and committed)
              - "recent" nodes: commit_ts >= low_watermark (or uncommitted)
           c. Among "old" nodes, only keep the LAST one (highest commit_ts
              below low_watermark). This is the version visible to the oldest
              active transaction. Discard all older ones.
           d. Keep ALL "recent" nodes.
           e. Rebuild the chain from newest to oldest.

        Safety argument:
        - Any version with commit_ts < low_watermark is not visible to any
          active transaction as their "latest visible" version, UNLESS there
          is no newer committed version with commit_ts < low_watermark.
          Specifically, the latest committed version with commit_ts < low_watermark
          IS visible to any active transaction whose start_ts falls in the gap
          between that version's commit_ts and the next newer committed version.
          Therefore, we must keep this one "boundary" version.
        - All versions with commit_ts >= low_watermark must be kept because
          some active transaction with start_ts in [commit_ts, +inf) might
          need to see them.
        - Uncommitted versions (commit_ts is None) are always kept.
        - The absolute latest committed version on each chain is always kept
          (it's either "recent" or the last "old" version).
        """
        with self._gc_lock:
            low_watermark = self._registry.low_watermark()

            keys_to_gc = []
            with self._data_lock:
                keys_to_gc = list(self._data.keys())

            for key in keys_to_gc:
                with self._data_lock:
                    head = self._data.get(key)
                    if head is None:
                        continue

                    all_nodes = []
                    node = head
                    while node is not None:
                        all_nodes.append(node)
                        node = node.next
                    all_nodes.reverse()

                    old_nodes = []
                    recent_nodes = []
                    for n in all_nodes:
                        if n.commit_ts is not None and n.commit_ts < low_watermark:
                            old_nodes.append(n)
                        else:
                            recent_nodes.append(n)

                    kept_old = old_nodes[-1:] if old_nodes else []

                    kept_all = kept_old + recent_nodes

                    new_head = None
                    for n in kept_all:
                        new_node = VersionNode(
                            start_ts=n.start_ts,
                            value=n.value,
                            is_deleted=n.is_deleted,
                        )
                        new_node.commit_ts = n.commit_ts
                        new_node.next = new_head
                        new_head = new_node

                    self._data[key] = new_head

    def gc_stats(self) -> dict:
        """Return statistics about version chains for debugging."""
        stats = {
            'keys': 0,
            'total_versions': 0,
            'low_watermark': self._registry.low_watermark(),
            'active_transactions': self._registry.active_count(),
        }
        with self._data_lock:
            for key, head in self._data.items():
                stats['keys'] += 1
                node = head
                while node is not None:
                    stats['total_versions'] += 1
                    node = node.next
        return stats

    def debug_chain(self, key: str) -> List[str]:
        """Return a string representation of the version chain for a key."""
        with self._data_lock:
            head = self._data.get(key)
        result = []
        node = head
        while node is not None:
            result.append(repr(node))
            node = node.next
        return result
