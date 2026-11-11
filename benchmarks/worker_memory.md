# Worker allocator memory diagnostics

Use the model-neutral extension in [worker_memory.py](worker_memory.py) to
collect allocator counters from an actual AR worker through its existing
collective-RPC interface. It is opt-in and adds no per-request callback.

## Start a diagnostic service

From a source checkout with a matching vLLM/Omni runtime:

~~~bash
PYTHONPATH="$PWD" VLLM_SERVER_DEV_MODE=1 \
  vllm serve /path/to/model --omni \
  --host 127.0.0.1 --port 18082 \
  --worker-extension-cls benchmarks.worker_memory.WorkerMemoryExtension
~~~

Add the deployment profile required by the model. The development RPC surface
is privileged; keep this diagnostic service on localhost. This setting is not
needed for ordinary serving.

The extension uses generic torch.accelerator APIs. The backend must report
allocated_bytes.all.current, reserved_bytes.all.current and their peak
counters. Missing counters raise an error instead of producing zero-valued
measurements. Diffusion workers and backends without these counters are not
implicitly supported by this AR-worker extension.

## Define the measurement interval

Wait for readiness, prewarm the intended shapes, and let all requests finish.
Reset counters explicitly before a diagnostic batch:

~~~bash
curl --fail-with-body http://127.0.0.1:18082/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"benchmark_memory_reset","args":["prewarmed-batch"],"timeout":30}'
~~~

Run the intended requests, wait for completion, then read the same interval:

~~~bash
curl --fail-with-body http://127.0.0.1:18082/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"benchmark_memory_read","timeout":30}'
~~~

Both calls synchronize the worker's current accelerator. Collect diagnostic
batches separately from latency comparisons. Do not reset counters while
another workload is using the worker.

The worker response includes PID, scope label, reset/observation timestamps,
current allocated/reserved bytes, and peak allocated/reserved bytes. A read
before an explicit reset raises an error. A new reset starts a new interval;
scope state belongs to each worker instance.

Omni may wrap responses with stage/replica metadata. HTTP 200 alone is
insufficient: reject unsupported/TODO/error results and verify the expected
number of workers, matching scope labels and reset timestamps. Keep each
worker's result, including failures.

## Interpretation

- Peaks cover the explicit interval. Initialization or other components may
  have reset allocator counters earlier, so a post-readiness response is not
  an exact cold-start peak.
- Reserved memory already includes allocated memory; do not add the two.
- These are allocator counters, not all device memory. Track whole-device and
  process-level usage separately, and do not sum unrelated stage peaks.
- Preserve multiple reads in a scope. A decreasing peak counter indicates
  another reset or an invalid scope; do not present that interval as complete.
  Re-check this across allocator lifecycle operations such as sleep/wake.
  Matching worker PID and extension reset timestamps alone do not establish
  an uninterrupted peak interval. Start a new reset after wake when measuring
  a new serving window; keep cross-cycle peaks missing if their scope was lost.
- Model parameters, KV pools, graph-private allocations and other buffers
  need their own attribution. The extension does not infer that breakdown.

## CPU contract tests

Prerequisites: the matching Torch runtime and repository test dependencies
(including pytest, pytest-asyncio and pytest-xdist). No model weights or GPU
context is needed; the tests replace accelerator operations.

~~~bash
python -m pytest tests/benchmarks/test_worker_memory.py \
  -m 'core_model and cpu' --run-level=core_model
~~~

The existing ready-pipeline Simple Other Test collects this path through its
core_model and cpu expression. No additional GPU CI job is required for these
unit checks. A serving experiment is still needed to validate a particular
worker/runtime integration.
