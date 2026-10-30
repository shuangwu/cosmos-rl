# Completion selection and accounting

## Quality selection: identical in colocated and disaggregated modes

Applications return `RolloutResult.completion_trainable` and optional aligned
`completion_drop_reasons` from their rollout producer. This existing interface
is independent of deployment mode and uses the selected algorithm's
`minimum_trainable_completions`, not a hard-coded GRPO rule.

The shared worker/reward path performs these steps:

1. Compute reward telemetry (including excluded completions where supported).
2. Resolve the producer's quality mask for each original prompt group.
3. Exclude the entire group if too few eligible completions remain.
4. Compute advantages using **only eligible rewards**.
5. Select all completion-aligned fields consistently and report outcomes.

For example, rejecting reward `100` from `[0, 1, 100]` must normalize `[0, 1]`,
not retain the first two advantages computed from all three rewards. Rejected
members of one prompt must not be replaced with members of another prompt to
satisfy the minimum. Local and remote rewards honor this ordering; bypassed
rewards select the same members with zero rewards/advantages. Validation ignores
training masks and continues evaluating every completion.

There is deliberately **no controller quality callback**: received training
payloads already carry group-derived advantages. Controller extraction rejects
an unconsumed mask instead of silently training it or filtering too late.
Late staleness and terminal-job cleanup remain separate lifecycle decisions.

For step-bounded runs with validation, sampler exhaustion is not a prerequisite
for terminal cleanup. The controller closes admission and releases buffered tail
payloads after the final dispatch's sealed ACK set and final validation round
are complete, with no outstanding dispatch, active validation iterator or
payload lookahead. A missing training ACK cannot be replaced by live READY
status or a completed validation receipt. No extra synthetic training step is
issued to finish this path.

Colocated workers inherit the same producer and reward implementation. Their
controller queues only selected completions, aggregates numeric telemetry, and
does not sum routing IDs or originating weight versions. Both centralized and
uncentralized colocated ingestion follow this contract; existing queue-based
collection continues until enough accepted samples are available.

### Colocated terminal-sample accounting

Fetching a prompt charges its requested generations even in colocated mode.
Accepted payloads stay on the existing local queue path. Quality exclusions and
structurally failed/empty generations instead send an accounting-only report
through the registered producer's ordered HTTP receipt stream. No accepted
tensors, rewards or advantages cross that reporting path; healthy reports add
no HTTP request or training collective.

The controller validates the source incarnation, rank, sequence and current
preparing dispatch before retiring the count. It decrements both the global
in-flight charge and that dispatch's outstanding fetched balance. This makes
lost-ACK retries idempotent and prevents an exhaustion cancellation from
retiring the same dropped samples again. Training ACKs still settle only the
actual training batch; any accepted queue surplus remains charged. A fresh
discard cannot retire a batch whose preparation agreement is already sealed.

Each new dispatch records the charged local input carried from earlier updates.
Existing strict on-policy purges report `discarded_queued_samples` against that
carried balance, without changing which samples are trained or their ordering.
Dataset exhaustion cancels both remaining carried input and current fetched
input; acknowledged drops cannot be counted again. Once preparation is sealed,
a queue-only discard can retire only surplus outside the optimizer reservation.

On clean exit, each rank retires its own unused local queues before one final
CPU completion fence permits source unregistration. Empty ranks participate too:
rank zero must not unregister while a peer's queued-input receipt is pending.
Failure to confirm any purge prevents clean completion. This adds no per-update
training collective and leaves accepted completion reporting on its local path.

For example, six fetched samples with two exclusions leave four charged for
training, then zero after the training ACK. Six healthy samples with a batch of
four leave two queued samples charged until reused, purged or retired at clean
exit. Delivery failure stops the source rather than admitting healthy work after
uncertain accounting.

## Optional identified reporting

Set `[rollout].completion_admission = true` to enable the standard worker's
pre-generation reservations and the controller's replay-protected settlement.
Synchronous generation and the asynchronous task scheduler preserve identities
through failures, reward processing, selection, serialization and HTTP retries.
Exhausted report delivery raises rather than silently losing reservations.
Custom producers may instead enable `completion_admission=True` on controller
`main` / `Controller.setup` and implement the same contract.

This does not install quality policy or force a quality mask onto legacy
unmasked groups. The ordinary producer mask path does **not** require identified
reporting and retains report-level discard accounting. Identified settlement
supports disaggregated, non-DAPO RL; this accounting limitation does not restrict
quality selection in either mode. Unsupported configurations fail at startup.

`RolloutRequest.completion_identities` contains one `CompletionIdentity` per
accepted completion in payload/group order. Allocate monotonically increasing
sequences **before generation**, scoped to source replica incarnation and global
rank; retain the originating `weight_version`. A restart uses a fresh replica
name. Retries reuse both identity and payload reference.

Report generation failures, quality exclusions and all members of insufficient
groups through `completion_failures`. Their identities come from the same
allocation, not from renumbering the surviving group. An optional `payload`
carries the rejected completion's transport reference for controller cleanup;
it never enters the training buffer. Without it, resources remain producer-owned.
Do not attach an accepted consumer's reference to a rejected identity or share
one disposable reference across distinct completion identities.

The controller settles and refills each rejected identity once using existing
versioned accounting and cleanup. A failure followed by a late payload settles
once and releases the late payload once. Replays of accepted completions never
release buffers a consumer may still own. Legacy discard/admission metric reports
must not be mixed with identified settlement.

Duplicate identities within a report, mismatched/future versions, invalid source
ranks and reports from departed replicas fail before accounting mutation. Malformed
or expired reports do not transfer new ownership. Replay state retains at most
4096 sequences per live source rank; sequences below the window are rejected,
never considered new. Producers must bound reordering. This is not a durable
controller-restart journal.

Replacement prompts do not consume still-unissued normal prompt quota. This is
essential when a producer reports a failure before other producers finish
fetching their initial batch, especially with zero allowed staleness.

Admission and replica changes share the controller lifecycle lock. Closed
admission discards without reopening prompt capacity. An infrastructure error
after settlement starts terminates the controller process with exit code 86.
It cannot be converted into a retryable HTTP error or leave a poisoned controller
alive. Atomic validation errors remain ordinary errors before mutation. Process
restart and cross-node scheduler cancellation remain deployment responsibilities.

## Application reward callbacks

Custom `BaseDataPacker` implementations may override two optional methods:

```python
def annotate_reward_result(self, payload, result):
    # Original group order, including members excluded from training.
    return [
        {"sample": payload.prompt_idx, "slot": slot,
         "admitted": slot in result.selected_indices}
        for slot in range(len(result.rewards))
    ]

def on_reward_read_complete(self, payload, *, is_validation, error):
    # Finish any application asynchronous readers before returning.
    self.reward_views.pop(payload.prompt_idx, None)
```

Annotation runs once after native rewards, admission and advantages have been
computed. `RewardResult` contains tuples in original completion order: rewards,
filter rewards, advantages (`None` for rejected members), and borrowed report
metrics. `selected_indices` identifies native admitted members, or an empty tuple
when the group is insufficient. Selection precedes whole-group dynamic sampling;
this callback does not promise eventual policy consumption. Treat the payload
and native result as read-only. Annotations cannot change rewards, advantages,
admission, reservations, replacement capacity or training counters. Continue to
make application quality decisions in the producer's completion mask.

Return `None` for no annotations, or a list of dictionaries with one entry per
original slot. Each dictionary must contain only finite JSON primitives, with
string keys, at most 4096 encoded bytes, eight nesting levels and 1024 nodes.
The complete group's diagnostic envelope is limited to 65536 bytes. Values are
copied without invoking arbitrary serializers. Tensors, arrays, object handles
and transport leases are rejected; native producer-owned reports are unchanged.
Do not use annotations to retain borrowed payloads or native report objects.

The selected payload carries `completion_diagnostics`, whose `original_index`
survives selection. Its `rejected_completion_diagnostics` retains excluded slots
even when the group has no selected members. With identified reporting,
`RolloutRequest.completion_diagnostics` aligns with its accepted identities and
each `CompletionFailure.diagnostics` carries the corresponding rejected member.
These detached diagnostics survive removal of a rejected group from report
payloads, and identity/order remain aligned across serialization and HTTP retry.

Pass `hook_fns={"completion_settled": observer}` to the controller to observe
identified admission. The synchronous observer receives a tuple of
`CompletionSettlement` events with source replica/rank, stable identity, native
disposition and optional detached diagnostics. It receives no payload, controller
or settlement operation. Native accounting, cleanup and replay-window commit
complete before invocation. Duplicate identities and late cleanup-only payloads
produce no additional event, including reordered or all-rejected reports. An
accepted disposition means native admission, with final policy buffering and
outdated filtering still owned by the controller. Observer failure poisons
admission and follows the existing fatal controller path (exit 86); accounting
is already committed and must never be repeated to repair observer failure.
Applications needing durable delivery must persist events in their observer;
this is not an exactly-once journal across controller restart.

Reward annotation and final reward-reader cleanup work in colocated and
disaggregated modes, for local, remote and bypassed rewards. Identified settlement
observation retains the native disaggregated non-DAPO restriction above.
Colocated callbacks expose the native admission result and detached diagnostics;
they do not provide a per-identity controller settlement event. A colocated
consumer must capture rejected diagnostics in its annotation callback before
native reporting removes a fully rejected payload.

The final-reader callback runs in a `finally` boundary after evaluation and
annotations, including native reward/advantage, annotation and preprocessing
failures. Each submitted group's original payload is passed, including groups
not evaluated because an earlier group failed. Cleanup must finish synchronous
and asynchronous application reads, release only application reward-owned views,
and tolerate repeated calls. Trainable colocated completion views, policy leases
and transport ownership remain live until their respective final readers finish.
Hooks execute in the reward calculator's process/thread (or the calling process
for bypass), so application caches must belong to that execution context.
Tasks canceled before starting notify the submitting role's packer with
`CancelledError`; they have no calculator-process readers. Submission failures
also finish all groups that have not been submitted.

An annotation or cleanup failure fails its reward task. Cleanup attempts every
submitted group even if one cleanup callback fails; an original evaluation error
is preserved and secondary cleanup errors are logged. Remote transport errors,
server failures and rate limits retain readers for polling. Invalid response
content, mismatched completion counts, nonretryable HTTP failures and annotation
errors terminate the evaluation and clean pending readers with that original
error. Remote dispatcher shutdown waits for active readers and calls idempotent
`close`, notifying remaining pending groups with `CancelledError`. If that close
cannot complete within the shutdown budget, or canceled-reader cleanup fails,
the role retains its resources using the existing shutdown failure behavior.

Validation ignores training masks, exposes every original slot, uses the
validation data packer's annotation and cleanup callbacks, and carries accepted
diagnostics in validation payloads. Validation has no training identities,
replacement capacity or `completion_settled` events.

`tests/test_reward_callbacks.py` covers detached metadata, partial/insufficient/
all-rejected groups, original slots, native reward/advantage call counts,
custom-packer reward/policy ownership, repeated cleanup, local/remote/bypass and
error/close behavior, replay suppression and required-observer failure.
`tests/test_completion_admission_modes.py` exercises annotations and cleanup
through real disaggregated and both colocated reporting/ingestion paths. These
CPU regressions do not certify application-specific CUDA streams or distributed
transport implementations; those integrations must test their own final readers.

## Downstream adoption

Applications should decide quality while producing the completion group and
populate the mask before submitting it to rewards. They must not discard
individual members in controller ingest or trainer preprocessing after advantage
normalization. Existing whole-group pre-save gates can emit an all-false mask;
partial gates emit the appropriate member mask. Keep driving-specific policy
and diagnostic retention in the application. Quality-policy errors must not
silently admit an unchecked group.

## Validation scope

`tests/test_completion_admission_modes.py` drives real worker generation-result
processing, local rewards, reporting and ingestion for disaggregated and both
colocated queue modes. It checks partial/all rejection, insufficient groups,
algorithm-specific minima, aligned advantages, and rejection accounting.
Existing tests cover remote/bypassed rewards and validation behavior.

`tests/test_completion_reporting.py` checks pre-generation reservations, reward
pickling, stable identities, asynchronous failures, delivery exhaustion and real
subprocess exit after settlement/filter/buffering faults. These suites run in CI.

`tests/completion_admission_canary.py` requires the RL-Gym companion modules.
Run it as the launcher role entrypoint from the companion directory using
`tests/configs/completion_admission_disaggregated.toml` (two policy/two rollout
replicas) and `tests/configs/completion_admission_colocated.toml` (one policy
replica). Set `ADMISSION_CANARY_MODE` accordingly so the fixture selects the
appropriate packer. It uses standard identity reporting, injects a generation
failure and pre-reward quality rejection, then replays reports. Both configurations
run 20 steps. The test algorithm explicitly supports the companion's single-member
groups; GRPO's minimum is not weakened.

A separate disaggregated run with `ADMISSION_CANARY_SETTLEMENT_FAULT=1` must exit
nonzero after logging `Completion settlement failed; terminating controller`.
Use a finite outer timeout and distinguish a timeout from the expected failure.
