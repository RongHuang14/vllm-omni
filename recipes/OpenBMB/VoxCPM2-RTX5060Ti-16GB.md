# VoxCPM2 — RTX 5060 Ti 16GB

## Summary

- Vendor: OpenBMB
- Model: `openbmb/VoxCPM2`
- Task: Zero-shot speech synthesis and reference-audio voice cloning
- Mode: Single-stage online serving through `/v1/audio/speech`
- Hardware: One NVIDIA GeForce RTX 5060 Ti 16GB, native Linux
- Maintainer: Community

## When to use this recipe

Run VoxCPM2 on the measured 16GB card and compare a 6 GiB KV budget with an
optional 2 GiB profile. Both fit the recorded workloads; the smaller pool
leaves more device headroom but reduces token capacity. This is one baseline
for [consumer-device A1](https://github.com/vllm-project/vllm-omni/issues/10008).

## Supported model contract

| Property | Measured scope |
| --- | --- |
| Requests | English/Chinese short and long text; cloning from two fixed synthetic references |
| Output | 48 kHz, mono, PCM16 WAV; streaming interpreted at 48 kHz |
| Precision / topology | BF16, one stage on device 0; no quantization or CPU offload |
| Concurrency | Admission 8 for A/B, 4 for C/D; measured clients 1, 2 and 4 |
| Context / output limits | `max_model_len: 4096`, `max_tokens: 4096`; neither limit was saturated |
| Generation | 10 CFM steps, CFG 2.0, seed 42, VAE decode / audio emit cadence 2/1 |
| Graphs | Engine eager mode; model-owned unified graph on for A/B/C, off for D |

## References

- [Pinned model revision](https://huggingface.co/openbmb/VoxCPM2/tree/7936080b41163874f38610f46f438e2c0757f9fd)
- [Shared VoxCPM2 recipe](VoxCPM2.md) and [online TTS example](../../examples/online_serving/text_to_speech/README.md)
- [Speech API and cloning fields](../../docs/serving/speech_api.md#voxcpm2)
- [Supported models](../../docs/models/supported_models.md) and [feature compatibility](../../docs/user_guide/feature_compatibility.md)

## Hardware

- Accelerator: one NVIDIA GeForce RTX 5060 Ti, SM120, 16,311 MiB visible
- Topology: one device and one stage; no inter-device communication
- Host limits for the final measurements: 11 CPUs / 29 GiB RAM

## Software environment

| Component | Recorded value |
| --- | --- |
| OS / driver | Ubuntu 24.04.4 LTS / 580.142 |
| Python | 3.12.13 |
| vLLM-Omni commit | `7467d4b59f4290334aae8f84777e989e1cb1bbeb` |
| vLLM | 0.33.0, wheel from `4ba787ec9959e566d3b587a796dff07673324da5` |
| PyTorch / CUDA runtime | `2.14.0+cu130` / 13.0 |
| Loaded cuDNN | 9.25.1.1 |
| FlashInfer / JIT NVCC | 0.7.2.post1 / 12.8.93 |
| VoxCPM / Transformers | 2.0.3 / 5.19.0 |

Use the [installation guide](../../docs/getting_started/installation/README.md)
for a matching runtime and a local snapshot of the pinned model revision.
These results were measured at the commit above, not on current `main`.
The host loaded cuDNN separately; its installed package metadata still showed
9.24.0.43. To reconstruct that library directory:

~~~bash
uv pip install --python /path/to/venv/bin/python --no-deps \
  --target /data/voxcpm2-a1-16g/omni-cudnn-runtime \
  nvidia-cudnn-cu13==9.25.1.1
~~~

## Deployment profiles

| Profile | KV budget | Deploy config |
| --- | ---: | --- |
| A: reference | 6 GiB | [Reference YAML](VoxCPM2-RTX5060Ti-16GB-reference.yaml) |
| B: smaller pool | 2 GiB | [Candidate YAML](VoxCPM2-RTX5060Ti-16GB.yaml) |

Both inherit the bundled deploy config, explicitly set cadence 2/1 and disable
`enable_batched_vae_decode`. This avoids the legacy-VAE/unified-graph conflict
reported in [#9971](https://github.com/vllm-project/vllm-omni/pull/9971).
The two profiles differ only in KV budget; other runtime settings are shared.

## Command

Copy the two profile YAMLs and reference-results JSON from this recipe's
revision into the same paths in the pinned source checkout, then install
that checkout in the matching environment. Copy `benchmarks/worker_memory.py`
as well for allocator diagnostics. Run from that checkout and adapt the model and
cuDNN paths, preserving other library paths required by your environment.

~~~bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
export LD_PRELOAD=/data/voxcpm2-a1-16g/omni-cudnn-runtime/nvidia/cudnn/lib/libcudnn.so.9
export LD_LIBRARY_PATH=/data/voxcpm2-a1-16g/omni-cudnn-runtime/nvidia/cudnn/lib

vllm serve /data/voxcpm2-a1-16g/models/VoxCPM2 \
  --omni --served-model-name voxcpm2 \
  --host 127.0.0.1 --port 18080 \
  --deploy-config recipes/OpenBMB/VoxCPM2-RTX5060Ti-16GB.yaml
~~~

For the reference, use `VoxCPM2-RTX5060Ti-16GB-reference.yaml` in the same
directory. Start one comparison server at a time and wait for `/health`.

## Verification

~~~bash
curl --fail-with-body http://127.0.0.1:18080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxcpm2",
    "input": "Hello, this is a reproducible speech test on a single graphics card.",
    "voice": "default",
    "response_format": "wav",
    "seed": 42
  }' --output voxcpm2-short.wav
~~~

Check that the WAV is 48 kHz mono PCM16 and that it reads the requested text
without omissions, repetition or breaks. For cloning, use the reference fields
in the shared speech API above.

## Qualification evidence

The [reference results](../../benchmarks/tts/voxcpm2_rtx5060ti_16gb_reference.json)
retain unrounded per-cell measurements, corpus inputs, per-output ASR
transcriptions and edits, PCM hashes, evaluator pins and source-file hashes.

### Benchmark protocol

Use the recorded environment and frozen `seed_tts_smoke` dataset:

~~~bash
export VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000

python benchmarks/tts/bench_tts.py \
  --model openbmb/VoxCPM2 --served-model-name voxcpm2 \
  --task default_voice \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke --locale en \
  --host 127.0.0.1 --port 18080 \
  --concurrency 1 2 4 --num-prompts 30 --num-warmups 2 \
  --request-seed 42 --metric-percentiles 50,95,99 \
  --output-dir /data/voxcpm2-measured-r1-B1 \
  -- --tokenizer /data/voxcpm2-a1-16g/models/VoxCPM2 \
  --seed 0 --save-detailed --disable-tqdm
~~~

Before measurement, run that sweep with `--num-prompts 8` and a separate output
directory to prewarm its shapes, retaining `--num-warmups 2`. Run three rounds
in AB, BA, AB order (A: 6 GiB; B: 2 GiB), restarting the service for each
profile/round while retaining disk compilation caches. Each run also sent six
manual requests before the sweeps and three cloning requests afterward.

The final performance cohort has 540 measured requests, 144 explicit
shape-prewarm requests and 54 manual WAVs, with zero HTTP failures. CLI warmups
are additional. Latency runs did not load the worker extension or issue
diagnostic RPCs; first shape-exposure costs are outside the reported population.

### Memory and prewarmed latency

| KV budget | Full-run sampled device peak, each of three rounds | Device headroom at that peak |
| --- | ---: | ---: |
| 6 GiB | 12,465 MiB | 3,846 MiB |
| 2 GiB | 9,478 MiB | 6,833 MiB |

Device sampling covered startup, serving and shutdown at 200 ms intervals.
Samples can miss spikes; these are whole-device observations, not allocator
peaks or component sizes. The smaller pool reduced the sampled peak by
2,987 MiB (about 24%) in these runs.
In all three rounds, the 2 GiB maximum occurred before observed health, while
the 6 GiB maximum occurred afterward. The difference between full-run maxima
therefore does not measure the KV allocation difference directly.

| KV | Client concurrency | n per round | Per-round p95 first audio | Per-round p95 RTF | Audio seconds / second |
| --- | ---: | ---: | --- | --- | --- |
| 6 GiB | 1 | 30 | 124.1–128.0 ms | 0.196–0.204 | 5.182–5.278 |
| 6 GiB | 2 | 30 | 346.9–388.2 ms | 0.272–0.290 | 8.206–8.594 |
| 6 GiB | 4 | 30 | 477.4–490.7 ms | 0.334–0.338 | 13.063–13.137 |
| 2 GiB | 1 | 30 | 120.0–126.2 ms | 0.193–0.197 | 5.231–5.388 |
| 2 GiB | 2 | 30 | 374.7–381.1 ms | 0.284–0.285 | 8.262–8.385 |
| 2 GiB | 4 | 30 | 477.2–495.8 ms | 0.327–0.346 | 12.992–13.488 |

Ranges summarize three independent n=30 percentiles, not pooled n=90 results.
First audio is client-observed; RTF is request elapsed time / audio duration.
These uniform workloads met the study's p95 first-audio <=1 s and p95 RTF <1
targets. No statistically established speedup is claimed.

### Admission and graph profiles

A separate comparison holds KV at 2 GiB and uses these settings. The graph
toggle controls unified decode only; other model-owned graphs remain enabled.

| Profile | Admission | Unified graph | Graph batch ceiling |
| --- | ---: | --- | ---: |
| B | 8 | on | 8 |
| C | 4 | on | 4 |
| D | 4 | off | 4 |

To reproduce C, put this overlay beside the candidate YAML. For D, also set
`enable_unified_decode_graph: false` in the same runtime block.

~~~yaml
base_config: VoxCPM2-RTX5060Ti-16GB.yaml
stages:
  - stage_id: 0
    max_num_seqs: 4
    engine_extras:
      hf_overrides:
        voxcpm2_runtime_config:
          unified_decode_graph_max_batch_size: 4
~~~

Each profile starts a fresh service with retained disk caches. After n=8
short/long shape-prewarm at c=1/2/4, two measured sweeps use n=20 per cell:
short then long at c=1/2/4, followed by long then short at c=4/2/1. These are
two sweeps in one process, not independent process replications. The matrix
has 720 measured requests and 144 explicit prewarm requests; CLI readiness
checks and two warmup requests per cell are additional.

Build the controlled text fixtures from the reference results. Twenty IDs
repeat one fixed input per length; this is not a diverse traffic sample.

~~~bash
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("benchmarks/tts/voxcpm2_rtx5060ti_16gb_reference.json").read_text())
for label, text in data["length_inputs"].items():
    path = Path(f"/tmp/voxcpm2-{label}/en/meta.lst")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{label}-{i:03d}|unused.wav|unused reference|{text}\n" for i in range(20)))
PY
~~~

Use the benchmark command above with `--dataset-path /tmp/voxcpm2-short` or
`/tmp/voxcpm2-long`, n=8 for prewarm and n=20 for measurement. Use separate
output directories and reverse `--concurrency` to `4 2 1` in the second sweep.

All 720 measured requests completed without HTTP failures. At c=4:

| Profile | Input | Per-sweep p95 first audio (ms) | Per-sweep p95 RTF | Audio seconds / second |
| --- | --- | ---: | ---: | ---: |
| B | short | 201.4–208.3 | 0.273–0.280 | 14.645–14.915 |
| B | long | 204.5–205.7 | 0.245–0.249 | 16.216–16.538 |
| C | short | 202.9–204.4 | 0.275–0.276 | 15.003–15.144 |
| C | long | 208.2–212.8 | 0.247–0.253 | 15.966–16.331 |
| D | short | 223.2–227.6 | 0.277–0.280 | 14.512–14.550 |
| D | long | 217.7–219.0 | 0.254–0.258 | 15.749–15.864 |

Each range covers two n=20 percentiles, not a pooled n=40 statistic. Results
at c=1/2 are in the reference data; effects depend on concurrency and input
length. This study does not establish a general speedup from either setting.

Startup and the first external request were recorded separately (one fresh
process per profile, disk caches retained). Health includes the built-in
speech warmup; the external request is not the first model computation.

| Profile | Process start → observed health (s) | First external request first audio (s) | Full-run device peak (MiB) |
| --- | ---: | ---: | ---: |
| B | 69.1 | 2.362 | 9478 |
| C | 68.0 | 2.255 | 9478 |
| D | 66.1 | 2.320 | 9478 |

The first diagnostic service after the VM restart needed 439.3 s to reach
observed health while populating compilation caches. Cache eviction was not
controlled; this is one startup observation outside the latency comparison.

The full-run peak includes startup, latency sweeps, quality generation and
shutdown. The optional deploy profile keeps admission 8 and unified graphs
on; the smaller KV pool remains the only difference from its 6 GiB reference.

### Worker storage inventory

Separate diagnostic processes inspected actual worker tensors after two
uniform benchmark sweeps (n=8, c=1/2/4), outside the latency runs. Values below
are live storage/current allocator counters at the drained boundary, in MiB.
Shared tensor storage is counted once, including aliased KV views.

| Component / counter | B | C | D |
| --- | ---: | ---: | ---: |
| Model parameters | 4547.471 | 4547.471 | 4547.471 |
| Registered buffers, including fused weights | 316.003 | 316.003 | 316.003 |
| KV backing storage | 2047.500 | 2047.500 | 2047.500 |
| Selected retained audio-path tensors | 0.289 | 0.246 | 0.208 |
| Unassigned allocated | 425.574 | 424.449 | 424.451 |
| Allocator allocated | 7336.837 | 7335.669 | 7335.633 |
| Allocator reserved, including allocated | 7834 | 7802 | 7780 |

The audio row covers CFM inputs/work buffers, graph inputs/outputs and retained
request state. It does not cover all temporary CFM/VAE allocations or opaque
graph pools. Those, other unselected tensors and allocator rounding remain in
the unassigned row. These are current counters, not peaks or whole-device
residency. Lowering admission 8 to 4 reduced this observed allocated value by
only about 1.2 MiB; clients never exceeded 4, so this does not characterize
queuing above the admission limit. Scoped allocator peaks are retained in the
reference data; B covers ready onward, while C/D reset after shape prewarm.

### Bilingual quality

Each concurrency cohort generated 160 outputs: per profile, 30 default-voice
and 10 cloning requests in each language. Error rates below are fractions:
English WER and Chinese CER, summed edits / summed normalized reference units.
Scoring uses pinned Whisper large-v3, seaco Paraformer v2.0.4 and WavLM
base-plus; artifact hashes and per-item results are in the reference data above.

| Client concurrency | Locale / task | 6 GiB error | 2 GiB error | 6 / 2 GiB mean cloning proxy |
| --- | --- | ---: | ---: | --- |
| 1 | en / default_voice | 0.00625000 | 0.00625000 | — |
| 1 | en / voice_clone | 0.00471698 | 0.00471698 | 0.914671 / 0.914671 |
| 1 | zh / default_voice | 0.00740056 | 0.00740056 | — |
| 1 | zh / voice_clone | 0.00287356 | 0.00287356 | 0.908471 / 0.908471 |
| 4 | en / default_voice | 0.00468750 | 0.00625000 | — |
| 4 | en / voice_clone | 0.00000000 | 0.00000000 | 0.914974 / 0.915070 |
| 4 | zh / default_voice | 0.00740056 | 0.00832562 | — |
| 4 | zh / voice_clone | 0.00287356 | 0.00287356 | 0.908113 / 0.908613 |

Both cohorts passed the predeclared paired gates: content-error increase
<=1 percentage point and mean proxy drop <=0.02, with per-item drops above
0.05 investigated. Sequential PCM matched 80/80 pairs; concurrent c4 matched
29/80. Identical inputs reused ASR results within each cohort (80 / 131 unique
ASR inputs). Mean-pooled WavLM is a proxy, not a speaker verifier; the two
synthetic reference files do not establish speaker diversity.

The admission/graph comparison generated another 240 outputs at c=4 using
the same corpus and references (80 per profile, 213 unique ASR inputs).
Both B→C and C→D passed the same paired gates, with no per-item proxy drop
above 0.05. Cloning WER was zero in English and CER was 0.00287356 in Chinese
for all three profiles.

| Profile | EN default WER | ZH default CER | EN mean cloning proxy | ZH mean cloning proxy |
| --- | ---: | ---: | ---: | ---: |
| B | 0.00625000 | 0.00832562 | 0.915070 | 0.908441 |
| C | 0.00468750 | 0.00740056 | 0.916247 | 0.908756 |
| D | 0.00625000 | 0.00925069 | 0.916503 | 0.909359 |

These are automatic checks. The listening observations below cover the
earlier KV comparison, not changed PCM from the admission/graph study.

The author listened to six c4 pairs covering both languages and short, long
and cloning tasks. Both English-cloning files had some background noise and
are byte-identical across KV budgets; the source is unknown. No other issues
were reported. This was an informal non-blind check, not a MOS study.

### Mixed streaming and worker memory

A separate c4 run completed 991 measured requests after 24 prewarm over
904.81 s, using six English/Chinese short, long and cloning tasks.
No runtime errors or playback underruns were observed. All 54 saved clips
had zero ASR error; other outputs were not individually ASR-evaluated.

**Mixed latency missed the same interactive targets:** per-task p95 first
audio was 2.24–3.60 s; English/Chinese short-task p95 RTF was 1.038/1.056.
The uniform-workload table does not establish mixed-request latency bounds.

The separate [worker diagnostic](../../benchmarks/worker_memory.md) recorded
post-prewarm peaks of 7.434 GiB allocated / 8.008 GiB reserved in stage 0.
Sixteen snapshots retained the worker PID/reset scope; the last seven drained
boundaries stayed within 7336.837–7336.838 MiB allocated / 8198–8200 MiB
reserved. Reserved includes allocated. Full-startup allocator peaks and exact
opaque graph/transient-workspace attribution remain unmeasured.

### Idle sleep/wake

A separate [sleep-enabled](../../docs/features/sleep_mode.md) service completed three idle level-1 cycles through
`/v1/omni/sleep`, `/v1/omni/wakeup` and `/resume`. Output PCM matched prewarm in
all three. Device memory reclaimed was 7318–7328 MiB; wake-to-first-audio was
1.409–1.415 s. The service's sampled device peak was 9640 MiB.

The backend peak counter decreased across sleep despite the same extension
reset ID, so the cross-cycle allocator interval is invalid and its full-cycle
peak remains missing. Three idle cycles do not establish active-request
suspension, restore tail latency or an automatic idle policy.

## Notes

- Earlier cadence-3/1 profiles with legacy batched VAE enabled completed HTTP
  requests but omitted and repeated concurrent speech. Those timing results
  are excluded from the qualified profiles. Changing only that VAE switch
  produced 24 concurrent clips with zero ASR error at cadence 3/1, corroborating
  #9971; this recipe does not introduce another engine fix.
- Maximum input length and eight concurrent clients were not tested. Lowering
  the KV pool does not establish capacity at the configured limits.
- WAV/PCM responses do not expose server token counts or finish reasons.
  The tests do not establish production-duration stability or full A1 coverage.
- Other GPUs, 8/12GB devices, WSL2 and desktop co-tenancy were not qualified.
