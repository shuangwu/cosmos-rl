# Bounded NCCL payload reception

This opt-in mode limits **receiver-device payload storage**, not total device
memory. Configuration names and defaults are provisional pending maintainer
agreement. UCXX and Redis do not implement this budget.

```toml
[custom]
nccl_receive_budget_bytes = 8589934592  # example: 8 GiB, not a workload recommendation
nccl_receive_admission_timeout = 30.0  # seconds; finite and positive
```

Omitting `nccl_receive_budget_bytes`, or setting it to `0`, preserves the
existing concurrent NCCL receive path. A positive integer enables bounded
reception. No disk or host-payload spill is performed.

## Admission and accounting

The batch API must return the whole decoded batch. For each payload in input
order, let D be its decoded storage bytes and R its wire bytes (including the
32-byte device header). Before contacting any sender, reserve:

```
max(prefix decoded bytes including this payload + R)
```

Reservations for other batches, including unreleased consumer leases, also
count against the same per-strategy budget. This admission rule prevents a
partially received batch from occupying all capacity while waiting for space
that can only be released after handing the complete batch to its consumer.
A batch that cannot fit fails immediately with the required bytes and suggested
remedies. Waiting for an older consumer has a bounded, actionable timeout;
shutdown wakes admission waiters immediately.

Within an admitted batch, each payload is received, verified, and decoded before
receiving the next. The existing alignment clones remain. Receive completion
and decode completion are observed before advancing; all raw-buffer aliases in
the existing receive helper disappear when its per-payload call returns.
Reservations shrink to actual unique decoded storage at handoff. Truncated views
are charged for their **full backing storage**, and aliases of the same storage
are counted once. Peak reservations are conservative admission figures, distinct
from peak charged tensor bytes.

Compute on the prior batch and reception of the next still overlap when both
fit. This first implementation serializes payloads within the receiving batch,
even with extra headroom. It deliberately trades that receive concurrency for
predictable workspace. Measure throughput before selecting a production budget.

The cap excludes model/optimizer tensors, autograd state, consumer-created
copies/minibatch caches, CPU control-plane/header copies, NCCL internal memory,
and CUDA allocator reserved memory/rounding. Allocations can still OOM because
of those excluded users or fragmentation. Multiple strategies/devices have
independent budgets; this is not a process-wide cap.

## Consumer release contract

Bounded reception requires the prefetch API. A synchronous cache miss fails
explicitly instead of creating an untracked second receive. Repeated cache reads
are supported until final release. Call the following from the consumer thread:

```python
packer.start_prefetch(rollouts)
packer.wait_prefetch()
# Resolve/read the batch as many times as required.
# Training, reward and visualization readers may use additional CUDA streams.
# Drop every external payload/tensor/view alias when its final use is enqueued.
packer.release_prefetch(streams=(training_stream, reward_stream, visualization_stream))
```

`release_prefetch` records completion events on all supplied reader streams and
the current stream on each payload device, then waits before clearing the cache
and returning its budget. Event failure leaves the lease and its charge intact.
The lease independently retains its original backing storages even if a
consumer removes or replaces payload-dictionary entries. Those storages remain
charged until release. Consumer-created views do not extend the contract automatically: **every reader
must be finished or represented by a supplied stream, and no external alias may
be used or retained after release**. An external alias retained after release
violates the budget contract. Calls to release/collect must be serialized on the
consumer thread.

The next prefetch may be started before release to overlap communication and
compute. Release the old batch before `wait_prefetch`/`collect_prefetch` for the
next one; attempting to replace an unreleased cache raises an actionable error.
`release_prefetch` preserves the rollout double-buffer scheduling state.

The background worker drops its result reference immediately after queue
handoff. Shutdown closes admission and uses the existing communicator abort/join
lifecycle. It releases unconsumed queued batches after the worker stops. It does
**not** infer final use of a batch already handed to a consumer: the consumer
must still release that batch. Memory/prefetch errors are fatal to the attempted
collection and never silently turn into empty episodes. After a prefetch timeout,
shut down the packer before retrying to avoid consuming a late result as a new
batch. Existing per-transfer recovery/fallback behavior remains in the receive
helper. Failure to prove CUDA completion closes admission and retains the
reservation; restart that receiver rather than reusing its capacity.

NDAS must wire this boundary after its final training/reward/visualization readers
and manage its own caches. Enabling the setting without that caller integration
will deliberately fail at the next collection. No NDAS release boundary is
inferred or added by this Cosmos-only change. Consumer-specific integration is
a downstream opt-in task, not a prerequisite for validating this Cosmos PR.

## Telemetry

`NCCLTransportStrategy.receive_memory_stats()` returns:

- `budget_bytes`, `reserved_bytes`, `peak_reserved_bytes`;
- `raw_receive_bytes`, `decoded_leased_bytes`;
- `current_tensor_bytes`, `peak_tensor_bytes`;
- `admission_waits` (number of admissions that had to wait).

Tensor charges remain conservative until completion is observed. Each completed
bounded batch logs its snapshot. Allocation/decode failures log transfer ID,
schema, and budget attribution; receive-buffer allocation failures also preserve
the existing communicator resynchronization behavior.

## Standalone PR acceptance

Local CPU tests use real tensor allocation and unpacking with faked NCCL traffic.
They cover variable sizes, odd field alignment, backing-storage accounting,
repeated reads/batches, admission pressure/timeouts, shutdown, decode OOM cleanup,
worker reference release, stream-event ordering and lease retention on errors.
A CUDA test covers views and multiple reader streams when hardware is available.
These CPU tests alone are not evidence of a GPU remedy or workload-level
loss equivalence.

The PR can be validated without NDAS. Run the following on two GPUs, and
require zero skipped acceptance tests:

```sh
python -m pytest tests/test_receive_memory.py tests/test_receive_memory_cuda.py tests/test_nccl_e2e.py -v
python tests/receive_memory_cluster_probe.py --steps 20 --output results.json
```

`test_receive_memory_cuda.py` uses real Redis/NCCL, three producer endpoints with
variable and unaligned schemas, and a standalone consumer. It compares 20 SGD
steps against direct-input data, loss and gradients, both with the budget disabled
and enabled. It also asserts next-batch completion during a pending CUDA reader,
backpressure until a slow reader releases capacity, cancellation during admission,
oversized rejection before contacting a sender, real CUDA allocation cleanup
after an injected decode error, missing-payload recovery, and storage retention
when a consumer mutates its payload dictionary. The injected error is not a
physical device OOM. All batch leases and charges must be released at final use.

The repeated-transfer probe compares peak allocated memory and mean fetch time
at batch sizes 16/32/64 for 20 iterations per mode. It checks each process exit,
payload values, repeated reads, stream completion, budget usage, and lower
steady-state receiver peaks. It has no NDAS dependency.

## Downstream workload evaluation (outside the PR gate)

A deployment may separately evaluate its pinned model/data/image with budget
disabled and enabled. That evaluates the application's final-reader integration
and workload-specific memory savings; it does not block the standalone Cosmos
PR. For such an evaluation, record separately for **every trainer**:

- progress through all iterations and matching data/loss within existing tolerance;
- raw, decoded/leased, reservation and charged-tensor peaks;
- CUDA allocated/reserved peaks and device-wide memory, independently of payload accounting;
- OOM count and allocation context, admission waits, step time and throughput;
- recovery under slow readers, producer failure, cancellation and repeated restart.

Include all final NDAS readers in the release contract. Compare sustained receive
peaks and throughput with the motivating workload, without per-wave cache flushes.
Keep source/image hashes, job configuration, and per-trainer logs with results.
## Recorded validation results

Final-source H100 job **2256429** completed successfully (exit 0) on two H100
80GB GPUs. All **41 contract/acceptance/round-trip tests passed with zero skips**,
including standalone data/loss/gradient equivalence, real prefetch overlap,
backpressure, admission cancellation, injected decode-error cleanup, and dictionary
mutation while a CUDA reader remained pending. The lease now holds unique backing
storage independently of the mutable payload dictionaries.

The real Redis/NCCL synthetic producer/consumer completed 20 batches per
baseline/bounded case at 16/32/64. Steady-state receiver allocated peaks (MiB)
were 768.7/272.2, 1537.4/528.5, and 3074.7/1040.9 respectively. Mean bounded
fetch-time increases were 3.6%, 4.0%, and 3.9%. This measures the combined incremental
receive and explicit-release change with roughly 16 MiB payloads. Peak memory
is PyTorch receiver allocation, and fetch time is not application throughput.
Separate standalone SGD tests establish exact data/loss/gradient equivalence;
separate delayed-reader tests establish actual prefetch overlap and backpressure.

The original NDAS workload and its final-reader integration were not exercised;
they are outside the standalone PR gate, and no NDAS OOM-remedy claim is made.
The validation report accompanying the change contains source hashes, commands,
job accounting, per-iteration metrics, and test XML.
