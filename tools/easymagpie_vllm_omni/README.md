## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **NemotronTTS** (Nemotron-H backbone + architecture-selected
acoustic predictor over a neural audio codec) via [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

EasyMagpieTTS decomposes into EasyMagpie LM and SpectralCodec-BWE-22kHz:

| Stage | Role |
|-------|------|
| **0 — EasyMagpie LM** | Nemotron-H backbone + v1 local transformer or v2 acoustic predictor with global cache → stacked acoustic codes |
| **1 — SpectralCodec-BWE-22kHz** | Stateful native vLLM codec → 22.05 kHz waveform |

Model definition and pipeline registration live in
[`easymagpie_vllm_omni/`](easymagpie_vllm_omni/) and
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni/).
Deployment knobs are in [`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

### Convert a checkpoint

This step converts the EasyMagpie LM, precomputes the text-embedding lookup,
and saves the tokenizer and optional named-speaker embedding. The Stage-1 codec
decoder is always bundled. By default, the codec encoder and reference-speaker
Transformer are not copied, so the resulting artifact accepts only known
`speaker_id` values.

Pass `--bundle-audio-encoders` only when you intend to package the codec encoder
and reference-speaker Transformer together. This one opt-in enables request-time
raw audio and therefore zero-shot voice cloning; it does not control the decoder.
Run conversion in the **NeMo environment** from the repository root. Prepending
the repository root to `PYTHONPATH` is important when the environment has an
editable NeMo install pointing at a different checkout:

```bash
PYTHONPATH="$PWD" python tools/easymagpie_vllm_omni/scripts/convert_to_vllm.py \
  --nemo_file /path/to/emptts.nemo \
  --codec_model_path /path/to/25fps_spectral_codec.nemo \
  --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer.json \
  --outdir tools/easymagpie_vllm_omni/converted_model \
  --context_audio /path/to/reference_voice.wav \
  --speaker_name eng
```

### Setup the serving environment

Serving needs a GPU, matching **vLLM 0.24 / vLLM-Omni 0.24** versions, and this package.
It does not need NeMo after conversion:

```bash
cd tools/easymagpie_vllm_omni
conda create -n easymagpie-vllm python=3.12 -y
conda activate easymagpie-vllm
pip install -r requirements.txt
pip install -e .
# optionally for notebook
pip install ipykernel
python -m ipykernel install --user \
  --name easymagpie-vllm \
  --display-name "Python (easymagpie-vllm)"
```

Mamba's selective-state-update kernel requires shape- and GPU-specific tuning, so an untuned cache can give
suboptimal performance. Reuse the same Triton/vLLM cache directories across launches so repeated runs accumulate
better kernels; for an explicit sweep, run `python scripts/tune_mamba_ssu.py --model converted_model` and restart.

### Quick start — offline synthesis

See the [`offline_demo.ipynb`](../../tutorials/tts/easymagpie_vllm_omni/offline_demo.ipynb) tutorial to check how
`AsyncOmni` is initialized and used.

### V2 CFG cache-pairing diagnostic

The first CFG milestone provides an explicit diagnostic pipeline,
[`deploy/easymagpie_cfg_pairing.yaml`](deploy/easymagpie_cfg_pairing.yaml).
It allocates two physical Stage-0 sequences per logical request, each with its
own backbone attention, Mamba, and temporal acoustic-predictor cache state.
Both sequences advance together; only the public sequence's output reaches the
codec or client. Completion and cancellation free both histories.

The cache diagnostic uses the existing conditional inputs on both branches. It accepts the internal
`additional_information._cfg_pairing_probe: true` marker through direct `AsyncOmni`
requests. Select the synthesis profile below for `use_cfg: true`, including
HTTP/WebSocket requests as described below.

In the pinned GPU serving environment, run the completion/abort/reuse smoke test:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_pairing.py \
  --model /path/to/converted_v2 --speaker eng
```

The diagnostic supports one non-resumable known-speaker request at a time,
eager execution, synchronous scheduling, and disabled prefix caching and chunked
prefill. Stage-0 capacity counts physical sequences: one logical request needs
two slots and a token budget covering both prompts. Split scheduling or preemption
aborts the diagnostic before the worker executes the unsafe step. Use this profile
for development; it deliberately fails on unsupported scheduling rather than
recovering or serving other requests concurrently.

The scheduler tests exercise the real vLLM/Omni 0.24 classes and hybrid cache-block
allocator on CPU. The smoke test additionally requires GPU execution with a V2
checkpoint; CPU allocation bookkeeping alone does not establish numerical cache
correctness or audio parity.

### V2 CFG synthesis

[`deploy/easymagpie_cfg.yaml`](deploy/easymagpie_cfg.yaml) adds guided synthesis
for known-speaker requests with complete text or bounded incremental text. Its two physical
Stage-0 slots fit one active CFG request; additional requests queue. Guided and
conditional-only requests use the same explicit pipeline, with `use_cfg` selecting
whether the scheduler creates a private branch. The pairing-only diagnostic keeps
its separate single-request scheduler.

The corrected profile passed the GPU completion/abort/reuse probe at CFG 2.5,
acoustic temperature 0.7, top-k 80 before the mixed-admission extension. Mixed
concurrency also passed the mixed GPU cancellation/reuse probe;
see the profile and probe below. Raw reference/user audio,
retained turns, preemption, CUDA graphs, asynchronous scheduling, chunked prefill,
and prefix caching remain unsupported in this CFG path. Scale 1 did not match
conditional-only greedy inference on the tested prompt; numerical parity remains open.

Re-run checkpoint conversion before using CFG. V2 conversion now exports
`cfg_context_embedding` using the exact one-token context CAS setting from NeMo,
the `cfg_unk_token_id`, and versioned `cfg_capability` metadata. Older artifacts
remain usable for conditional synthesis; CFG rejects them with a reconversion error.
An advertised capability with missing or malformed embedding weights fails loading.

For direct `AsyncOmni` calls, load plugins before constructing the engine so the
custom pipeline is registered when Omni resolves the deployment:

```python
from vllm.plugins import load_general_plugins
from vllm_omni import AsyncOmni

load_general_plugins()
omni = AsyncOmni(model="/path/to/reconverted_v2", deploy_config="tools/easymagpie_vllm_omni/deploy/easymagpie_cfg.yaml")
# Add these to an ordinary complete-text known-speaker prompt's additional_information:
prompt["additional_information"].update(use_cfg=True, cfg_scale=2.5, temperature=0.7, top_k=80)
```

`use_cfg` is a strict boolean. `cfg_scale` must be finite; zero and one are valid.
The serving default is 2.5, matching NeMo's high-level inference default (the
low-level streaming API defaults to 1.0). Acoustic temperature is finite, with
values at or below zero selecting actual greedy argmax. `top_k` is a positive
integer and retains all logits tied at the kth cutoff. V2 acoustic sampling settings
belong to each request and do not affect the engine's dummy continue/stop sampler.

Both branches advance their own temporal caches. The unconditional branch uses
CFG_UNK across the full task/speaker/context span, zeros during target warmup,
and shared audio inputs during decode. Phonemes come from the conditional branch.
Each predictor block combines logits before sanitization and sampling, then feeds
the one sampled code into both branches. Only one logical code stream reaches the codec.
The CFG profiles reserve 2048 positions for the codec's cumulative prompt history,
matching the Stage-0 generation budget. Keep those capacities aligned when raising
`max_tokens`; a small codec chunk size does not bound accumulated prompt history.

Run the same completion/abort/reuse check at the default acoustic settings:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_pairing.py \
  --model /path/to/reconverted_v2 --speaker eng --use-cfg --cfg-scale 2.5 --temperature 0.7
```

CPU tests cover export semantics, metadata/weight validation, branch inputs,
logit guidance, cutoff/greedy sampling, shared feedback, and engine bookkeeping.
They do not establish waveform quality, numerical cache parity, or measured CFG overhead.

### HTTP and WebSocket CFG selection

Launch the server with a CFG deployment and a reconverted V2 checkpoint containing
registered speaker embeddings:

```bash
EASYMAGPIE_DEPLOY_CONFIG=tools/easymagpie_vllm_omni/deploy/easymagpie_cfg.yaml \
  bash tools/easymagpie_vllm_omni/scripts/run_server.sh /path/to/reconverted_v2 8091
```

For complete text, send acoustic settings in `extra_params`:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from the speech server.","voice":"eng","response_format":"wav",
       "extra_params":{"use_cfg":true,"cfg_scale":2.5,"temperature":0.7,"top_k":80}}' \
  --output speech.wav
```

For incremental text, connect to `ws://localhost:8091/v1/audio/speech/stream`
and send these JSON messages in order. Receive binary PCM audio concurrently
with sending text; completion is reported by `audio.done` and `session.done`.

```json
{"type":"session.config","voice":"eng","stream_audio":true,"response_format":"pcm","max_new_tokens":512,"extra_params":{"use_cfg":true,"cfg_scale":2.5,"temperature":0.7,"top_k":80}}
{"type":"input.text","text":"Hello from "}
{"type":"input.text","text":"the speech server."}
{"type":"input.done"}
```

CFG is opt-in: omitting `use_cfg` or setting it to `false` selects conditional-only
inference. Guidance defaults to 2.5, acoustic temperature to 0.7, and top-k to 80.
Settings use the same strict validation as direct backend requests: `use_cfg` must
be a boolean, scale and temperature must be finite numbers, and top-k must be a
positive integer. Nonpositive temperature selects greedy acoustic decoding; scales
zero and one are supported. These are acoustic settings, separate from the engine's
dummy-token sampling settings.

Guided requests require the `easymagpie_cfg` pipeline (either CFG deployment profile),
valid V2 capability metadata, and a registered speaker. Unsupported deployments,
older artifacts, and raw-reference CFG requests are rejected before generation.
Incremental sessions require registered speakers for conditional-only inference too.
On the CFG pipeline, `max_new_tokens` bounds the whole stream, including text warmup,
EOS, and acoustic tail, for guided and conditional requests. The bound must fit the
deployment's cache and codec capacities. Settings are fixed for the session.

Live GPU acceptance passed complete-text HTTP and incremental WebSocket synthesis
for guided and conditional requests, invalid-setting and unsupported-profile rejection,
and disconnect after audio delivery followed by successful reuse. These smoke checks
do not establish speech quality, numerical parity, throughput, or long-run leak freedom.

### Incremental text with paired V2 CFG

Use the direct `AsyncOmni.generate()` streaming-input API with the explicit
`easymagpie_cfg` pipeline and a registered speaker. Speaker embeddings must be
prepared offline with the served checkpoint. The pairing-only diagnostic still
rejects streaming. HTTP/WebSocket clients can select CFG through `extra_params` as described above.

[`EasyMagpieInputStream`](easymagpie_vllm_omni/serving_stream.py) builds the input
iterator, buffers the initial text warmup, assigns absolute `text_token_start`
offsets, appends text EOS, and reserves the remaining budget for the acoustic tail.
Pass `use_cfg=True`, `cfg_scale=2.5`, `temperature=0.7`, `top_k=80`, and `speaker_id`
in its prefill prompt's `additional_information`. Size `prompt_token_ids` for the
registered speaker context **plus** `arch.text_prefill_num`, and pass that same
`text_prefill_num` to the input stream. Feed tokenizer IDs through `put_tokens()`
and call `finish()` when text is complete. For every engine output call
`stream.observe_output(output)` so the next text segment can resume. Run the input
producer and output consumer concurrently; abort the request on disconnect or error.

The stream's `max_new_tokens` becomes the immutable `max_stream_tokens` request
bound. It covers the sum of all segment generation limits, including warmup, text
EOS, and tail; it is not a per-chunk limit. The initial prompt plus that bound must
fit Stage 0's `max_model_len`, and the codec history must cover the bound (2048 in
the supplied profiles). Text that leaves no room for EOS/tail is rejected. Use
the checkpoint's phoneme-BOS warmup; a too-short initial text input cannot start
generation. Requests keep their original CFG mode, scale, speaker, and acoustic
sampling settings throughout the stream.

Both physical histories pause and resume together, using separate engine-managed
cache blocks and one shared acoustic sample. A suspended stream retains its full
reservation and resumes before fresh admissions that may be waiting for its
blocks. Exact duplicate text chunks are ignored; gaps, conflicts, partial overlaps,
and text after the acoustic tail are rejected. Only the public branch reaches the
codec and frontend. Text chunks belong to one synthesis request; retained dialogue
turns and request-time audio conditioning are outside this contract.

Run a delayed-chunk completion, cancellation-while-waiting, and reuse probe:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_streaming.py \
  --model /path/to/reconverted_v2 --speaker eng --output /tmp/cfg-streaming.json
```

Add `--mixed --deploy-config tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_concurrent.yaml`
to test streaming CFG/conditional requests alongside complete-text traffic under
sequence pressure. The probe uses CFG 2.5 / temperature 0.7 / top-k 80 and eager,
synchronous scheduling. Successful completion checks output routing and lifecycle;
it does not establish waveform/text quality, numerical parity, or throughput.

### Experimental mixed CFG/conditional admission

[`deploy/easymagpie_cfg_concurrent.yaml`](deploy/easymagpie_cfg_concurrent.yaml)
selects the same `easymagpie_cfg` pipeline with six physical Stage-0 slots and four
public Stage-1 slots. CFG consumes two physical slots, conditional-only consumes
one, and private branches never transfer codec payloads. Set `use_cfg=False` for
conditional-only requests in this pipeline, including HTTP/WebSocket requests.

`EasyMagpieCFGScheduler` admits logical units in FCFS order. Before admission it
reserves engine-managed cache for the entire bounded prompt plus generation
history of every branch. A pair waits together if sequence slots, the per-step
token budget, or cache capacity are unavailable. A unit that cannot fit the
total cache at all is rejected. Both branches run at aligned positions, without
preemption or Mamba replay. Request-wide reservation trades memory utilization
for this scheduling guarantee; lowering the output bound can increase admission
capacity. No throughput or memory-efficiency claims have been measured.

The mixed scheduler requires FCFS, Mamba cache mode `none`, and a token budget
covering every active physical decode sequence. Keep eager synchronous execution
and disable prefix caching/chunked prefill. CPU tests use the real hybrid cache
allocator to check pair isolation, independent stopping, queueing under pressure,
abort/late outputs, failure cleanup, and slot reuse. They also check multiple
guided pairs mixed with a conditional predictor row and private connector filtering.

The GPU probe submits six logical requests needing nine physical slots against
the six-slot profile, cancels one guided request, then checks another mixed wave
using the same engine. It requires public request IDs, no reported engine errors,
and completed codec output for every non-cancelled request:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_concurrency.py \
  --model /path/to/reconverted_v2 --speaker eng --cfg-scale 2.5 --temperature 0.7 --top-k 80 \
  --output mixed_probe_results.json
```

GPU execution of this mixed profile remains pending because CUDA/NVML became
unavailable again. Do not interpret the earlier single-request lifecycle pass
as validation of this extension or of exact waveform parity across batch sizes.

### Request-time reference and user audio

Every source EasyMagpie NeMo checkpoint contains the codec encoder and
reference-speaker Transformer. They are omitted from converted artifacts by
default and bundled together only with `--bundle-audio-encoders`; the Stage-1
codec decoder is always bundled. Without this explicit opt-in, requests must
select a precomputed embedding by `speaker_id`.

`arch.audio_input_token_id` in `prompt_token_ids` is an audio marker. Each
marker is paired, in order, with one item from `multi_modal_data["audio"]`;
the processor rejects a marker/item count mismatch. A non-final first marker
selects reference conditioning, while a final marker selects user history.
Each item is encoded once into both representations, and its marker selects
which representation is inserted.

Pass each item as `(waveform, sample_rate)`. It must already be mono and match
`arch.codec_input_sample_rate`; the serving path deliberately does not downmix
or resample it.

Each raw audio item is limited to `arch.max_audio_seconds` (30 seconds by
default). Longer inputs are rejected rather than truncated or chunked; set
`--max-audio-seconds` during conversion to change the startup profiling limit.

For a first turn with raw reference audio and user speech, place the reference
marker before context rows and the user marker last:

```python
prompt = {
    "prompt_token_ids": (
        [0] * task_rows
        + [arch.audio_input_token_id]
        + [0] * len(context_token_ids)
        + [arch.audio_input_token_id]
    ),
    "multi_modal_data": {
        "audio": [
            (reference_waveform, reference_sample_rate),
            (user_waveform, user_sample_rate),
        ],
    },
    "additional_information": {
        "context_text": "[EN]",
        "text": response_text,
        "text_prefill_num": arch.text_prefill_num,
        "temperature": 0.7,
        "top_k": 80,
        "reset_codec_on_segment": True,
    },
}
```

A non-final marker is reference conditioning; a final marker is user history.
For reference-only synthesis, replace the final user marker with
`arch.text_prefill_num` zero rows and provide `prefill_text_tokens`. For a
known voice, set `speaker_id`; then a sole final marker is user history. On
later turns of the same Stage-0 request, submit only that final marker and the
new user waveform—the model retains the earlier raw reference conditioning.

User-history prefill additionally requires a checkpoint configured with
`use_multiturn_dataset`, `condition_on_user_speech`, and the user-speaking
tokens. Reference-only raw audio does not require those multiturn features.

Yielding turns as `StreamingInput` items from one async input generator keeps
the Stage-0 causal/Mamba state. A separate `omni.generate(...)` call starts a
new request and must provide `speaker_id` or raw reference conditioning again.
Set `reset_codec_on_segment=True` for each reply so Stage 1 flushes and resets
its response-local codec state.

The OpenAI-compatible speech endpoint accepts request-time reference audio for
zero-shot TTS. The incremental WebSocket endpoint currently accepts text input
with a known `voice` only; use the direct `AsyncOmni` path for multi-turn raw
user-audio history.

### Serve over HTTP and WebSocket

```bash
bash ./scripts/run_server.sh ./converted_model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091. Two serving
APIs are available:

- `POST /v1/audio/speech` with a complete text input and either a known `voice`
  or one request-time `ref_audio`.
- `WS /v1/audio/speech/stream` with incremental text/token updates and
  asynchronous PCM audio output.

Converted checkpoints with `enable_phoneme_text_input=true` accept inline IPA
spans such as `Turn <bop>lɛft<eop> here`. The markers are syntax only: ordinary
segments use the exported text tokenizer, while span contents use the bundled
IPA tokenizer and the checkpoint's reserved text-token range.

For delayed-stream checkpoints, the adapter folds the known text-led positions
into the causal prefill. The current `phoneme_delay=3`, `speech_delay=5` model
therefore prefills four target positions: text-only positions 0–2 and position
3 with the known phoneme BOS input. Whole-text HTTP requests satisfy this
automatically. Incremental WebSocket input buffers initial updates until at
least `phoneme_delay + 1` tokens are available. Marker strings and IPA spans may
cross `input.text` messages. An unclosed IPA span is rejected at `input.done`;
`input.tokens` remains an exact tokenization bypass and is accepted only when
there is no incomplete text marker or IPA span.

Query the HTTP endpoint from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a TTS service test.","voice":"eng","response_format":"wav","stream":true,"stream_format":"audio"}' \
  --output out.wav
```

See the [`server_request.ipynb`](../../tutorials/tts/easymagpie_vllm_omni/server_request.ipynb) tutorial for examples
of both serving APIs.

For zero-shot TTS, omit `voice` and send `ref_audio` as an HTTP(S) URL, a base64
data URL, or a `file://` URI. Local files must be under the server's
`--allowed-local-media-path`. EasyMagpie does not require `ref_text`. The audio
is downmixed to mono by the HTTP media loader, but it is not resampled: its
sample rate must equal `codec_input_sample_rate` in the converted model.
The model must have been converted with `--bundle-audio-encoders`.

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input":"This voice is conditioned from request-time reference audio.",
    "ref_audio":"file:///absolute/path/to/reference.wav",
    "response_format":"wav",
    "stream":true,
    "stream_format":"audio"
  }' \
  --output out_zero_shot.wav
```

### Benchmarks

```bash
# Benchmark acoustic token prediction only (no codec).
python scripts/benchmark_model.py --model ./converted_model -n 128 -c 1 32 \
    [--streaming --tokens-per-chunk 5]

# Benchmark the service's HTTP API with a known speaker.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32 \
    --speaker-id eng

# Benchmark zero-shot synthesis with one request-time reference.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32 \
    --reference-audio /path/to/reference.wav

# Perturb the reference for every request to bypass audio caches.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32 \
    --reference-audio /path/to/reference.wav \
    --randomize-reference-audio

# Benchmark the service's incremental synthesis via its WebSocket API.
python scripts/benchmark_incremental_server.py --model ./converted_model \
    --text-file vctk_subset.txt --tokens-per-chunk 5 -n 128 -c 1 32
```
