# Rollout-only execution

Set `[rollout].rollout_only = true` and
`[policy.parallelism].n_init_replicas = 0` to run generation and reward
calculation without a trainer. This is an explicit execution mode: an ordinary
training job whose policy replicas disappear keeps its existing end-of-rollouts
behavior. The controller rejects policy and reference registration in this mode.

A complete example is [`configs/rollout-only.toml`](../configs/rollout-only.toml):

```bash
cosmos-rl --config configs/rollout-only.toml
```

The example loads Qwen2.5-0.5B-Instruct and evaluates up to 32 GSM8K test
prompts on one GPU, with one completion per prompt. It needs the usual vLLM
installation, model/dataset access and Redis executable. Set
`policy.model_name_or_path` to a compatible local checkpoint to use local
weights. Custom entry scripts can register their own backend, dataset, packer
and reward functions through the normal `launch_worker` API; pass the script
as the launcher's positional script argument, for example
`cosmos-rl --config my-rollout.toml my_entry.py`.

## Initialization and versions

Each worker constructs its backend normally, including `post_init_hook()`.
Before starting its prompt producer or generation loop, it initializes the
engine with `load_format="auto"` and calls `post_init_engine_hook()` once through
the existing lazy initialization path. Re-entering that path after successful
initialization does not recreate the engine or repeat its hook. A custom backend
must set `_engine_initialized` after its successful `init_engine()`, as required
by the regular lazy initialization contract. Its `auto` load path must make its
checkpoint or application-owned model ready for inference.

No policy replica, initial weight transfer or rollout weight-sync communicator
is needed. Prompts and generated results use model version **0**, which denotes
the unchanged local model for this execution; there are no optimizer steps or
policy versions. Both synchronous generation and vLLM asynchronous generation
skip trainer version gates. Prompt preparation/prefetch continues to use the
existing bounded worker queues.

## Work, results and completion

`rollout.max_inflight_prompts` (default **128**) bounds outstanding prompts
across all workers, including generation, queued rewards and unacknowledged
reports. Each prompt reserves `n_generation` completion slots. Workers use the
normal controller-issued identities, ordered fetch/report receipts and failure
settlement: a retried HTTP report releases capacity once. A full inflight budget
returns an empty batch with `is_end=false` so producers retry after capacity is
released. Dataloader batches larger than the requested amount are kept as an
undispatched remainder and cannot overrun either admission limit.

Execution ends when the dataset completes `train.epoch` passes, or the optional
positive `rollout.max_prompts` limit is reached. The limit counts dispatched
prompts, including prompts whose generation or quality selection fails; it does
not request replacements. An empty dataset ends without any generation call.
`train.max_num_steps` does not define a rollout-only horizon.

Reward and result packing run normally on the worker. Reports are receipt-checked
and counted, then transport payloads are released after reward processing.
Results never enter policy training buffers and do not update an optimizer or
consume a training sample budget. The controller logs completion counts and
outstanding completion slots; the application owns any desired persistence of
scores, completions or simulator outputs through its backend/reward integration.
This mode does not introduce a result sink or evaluation metric definition.

After all initial rollout replicas have registered, a controller-side
`await controller.request_stop("reason")` closes new prompt admission, preserves the first stop reason, and lets admitted generation/reward
work drain. It does not request a training checkpoint. After every reporter of
every registered initial rollout replica has acknowledged its final report and
posted its end signal, the controller broadcasts the normal NCCL-free STOP.
Workers remain command participants until that broadcast, preserving multi-rank
shutdown ordering. The controller exits after workers unregister.

Initialization, generation, asynchronous scheduler and transport failures use the
existing worker failure and resource-ownership paths. Engine, reward workers,
packer transports and packers are closed on success and failure, including
partial initialization. Cleanup failures remain errors; dependent resources are
retained when a producer cannot safely drain. The launcher fails and contains
peer processes on any nonzero rollout-only worker exit, even if the controller
has already exited after that worker unregistered. Normal training retains its
existing controller-supervised recovery policy. See
[`rollout_worker_lifecycle.md`](rollout_worker_lifecycle.md) for ownership details.

## Initial supported contract

The first contract supports disaggregated execution with a non-DAPO GRPO rollout
configuration, one or more rollout replicas, synchronous or vLLM asynchronous
generation, and custom backends using `DisaggregatedRolloutControlWorker`.
Trainer-oriented validation rounds, distillation, checkpoint resume, completion
admission, rank-affine dispatch, asynchronous R2R weight sync and the separate
TRT-LLM worker are rejected. This avoids silently accepting options whose
lifecycle still depends on trainers. Multi-DP backends must support empty DP
batches to consume an uneven final batch, as in ordinary training.

All controller and worker processes must run the same build. Model changes
require a new execution. This mode does not add elastic replacement or checkpoint
recovery for an already admitted prompt.
