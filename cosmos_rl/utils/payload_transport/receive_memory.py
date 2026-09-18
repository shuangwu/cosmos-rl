# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte admission and explicit batch leases for bounded payload reception.

Reservations include the complete decoded batch and its transient workspace.
They are deliberately distinct from live tensor byte attribution.
"""

import math
import threading
import time


class ReceiveMemoryError(RuntimeError):
    """A receive cannot safely proceed under the configured memory contract."""


class ReceiveBudget:
    def __init__(self, limit, timeout):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("nccl_receive_budget_bytes must be a positive integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("nccl_receive_admission_timeout must be positive")
        self.limit = limit
        self.timeout = timeout
        self.condition = threading.Condition()
        self.closed = False
        self.used = self.peak = self.waits = 0
        self.raw = self.decoded = self.peak_live = 0

    def reserve(self, size):
        if size < 0:
            raise ValueError("negative receive reservation")
        if size > self.limit:
            raise ReceiveMemoryError(
                f"NCCL batch needs {size} bytes (decoded batch plus receive/decode "
                f"workspace), exceeding nccl_receive_budget_bytes={self.limit}; "
                "increase the budget or reduce batch/payload size. Disk spill is disabled."
            )
        deadline = time.monotonic() + self.timeout
        with self.condition:
            waited = False
            while self.used + size > self.limit and not self.closed:
                if not waited:
                    self.waits += 1
                    waited = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReceiveMemoryError(
                        "NCCL receive admission timed out; final readers must call "
                        "release_prefetch before waiting for the next batch, or "
                        "increase nccl_receive_budget_bytes."
                    )
                self.condition.wait(remaining)
            if self.closed:
                raise ReceiveMemoryError("NCCL receive admission cancelled by shutdown")
            self.used += size
            self.peak = max(self.peak, self.used)

    def release(self, size):
        with self.condition:
            self.used -= size
            self.condition.notify_all()

    def attribute(self, *, raw=0, decoded=0):
        with self.condition:
            self.raw += raw
            self.decoded += decoded
            self.peak_live = max(self.peak_live, self.raw + self.decoded)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(
                budget_bytes=self.limit,
                reserved_bytes=self.used,
                peak_reserved_bytes=self.peak,
                raw_receive_bytes=self.raw,
                decoded_leased_bytes=self.decoded,
                admission_waits=self.waits,
                current_tensor_bytes=self.raw + self.decoded,
                peak_tensor_bytes=self.peak_live,
            )


class ReceivedBatch(dict):
    """Dictionary owning a reservation until the final consumer releases it.

    Callers must finish every reader, including tensor views, before release.
    Completion events retain both storage and budget until all streams finish.
    No implicit destructor release: forgetting a lease must not undercount memory.
    """

    def __init__(self, values, budget, nbytes):
        super().__init__(values)
        self.budget = budget
        self.nbytes = nbytes
        self.released = False
        # The consumer can reshape, pop, or replace dictionary entries. Lease
        # ownership must survive those mutations until all readers complete.
        storages = {}
        for payload in self.values():
            for tensor in payload.values():
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage
        self._storages = tuple(storages.values())

    def release(self, *, streams=()):
        if self.released:
            return
        import torch

        # The current stream is always included. Additional reader streams must
        # be supplied by the consumer; record all events before waiting on any.
        devices = {
            storage.device
            for storage in self._storages
            if storage.device.type == "cuda"
        }
        readers = list(streams) + [torch.cuda.current_stream(d) for d in devices]
        events = []
        for stream in readers:
            event = torch.cuda.Event()
            event.record(stream)
            events.append(event)
        for event in events:
            event.synchronize()
        self.clear()
        self._storages = ()
        self.budget.attribute(decoded=-self.nbytes)
        self.budget.release(self.nbytes)
        self.released = True


def storage_bytes(payloads):
    """Count full unique storage, including backing storage of truncated views."""
    storages = {}
    for payload in payloads:
        for tensor in payload.values():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())
