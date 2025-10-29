# Copilot Project Instructions: flux-fast

Purpose: This repo is a hackable reference showing concrete, composable optimization techniques that yield ~2.5-3.3x speedups for Flux.1 (Schnell, Dev, Kontext) image generation pipelines on high-end GPUs (H100, MI300X, L20). It is NOT a packaged library—scripts are meant to be read, modified, and benchmarked.

## Core Entry Points
- `gen_image.py`: Single image generation with all optimizations enabled by default; can optionally reuse cached export binaries (`--use-cached-model`).
- `run_benchmark.py`: Benchmarks (10 timed runs + warmup) and optional PyTorch profiler trace export; exposes fine‑grained flags to disable individual optimizations.
- `utils/pipeline_utils.py`: Implements optimization stack (QKV fusion, Flash Attention v3 / AITER, channels_last, float8 quant, inductor flags, torch.compile vs torch.export+AOTI+CUDAGraphs, cache‑dit integration). This is where new performance ideas should be wired.
- `utils/benchmark_utils.py`: CLI arg parser + profiler annotation helper.
- `cache_config.yaml`: Example configuration for cache‑dit (DBCache) when using Dev / Kontext 28‑step workflows.

## Optimization Toggle Model
All optimizations are ON by default; each has a corresponding --disable_* flag (see parser in `benchmark_utils.py`). New optimizations should follow the same pattern: add flag in `create_parser()`, implement in `optimize()` (short-circuit if disabled), keep ordering: structural graph changes (fusions) -> attention processor swap -> memory format -> cache‑dit -> quantization -> inductor flags -> compile/export path.

## Compile / Export Modes
`--compile_export_mode` values:
- `compile`: Uses `torch.compile` (mode="max-autotune" or "max-autotune-no-cudagraphs" if cache‑dit present; AMD forces `dynamic=True`).
- `export_aoti`: Uses `torch.export` + Ahead-of-Time Inductor + manual CUDAGraph wrapping (`cudagraph` helper). Serialized artifacts stored/loaded from `--cache-dir`. Hardware / environment specific; do not reuse across heterogeneous GPUs or OS.
- `disabled`: Runs eager (still with other enabled optimizations).
Behavioral nuance: `export_aoti` path is skipped (prints incompatibility message) when cache‑dit is active because dynamic cache logic breaks export stability.

## Flash Attention v3 / AITER Integration
Custom op registered as `flash::flash_attn_func` plus processor class `FlashFusedFluxAttnProcessor3_0`. It converts query/key/value to float8 (NVIDIA) or lets AITER handle fp8 conversion (AMD). Any changes should preserve custom op schema and `.register_fake` for compile tracing. When replacing attention, call `pipeline.transformer.set_attn_processor(...)` before compilation/export.

## Quantization
Float8 dynamic activation + float8 weights via `torchao.quantization.float8_dynamic_activation_float8_weight()`. Applied only to `pipeline.transformer`. If adding other quant schemes, gate behind a new flag; keep ordering BEFORE inductor flag tweaks and compile/export so that the compiled graph sees the quantized modules.

## cache-dit (DBCache)
Enabled via `--cache_dit_config <yaml>` (not for Schnell; enforced). Loads YAML via `load_cache_options_from_yaml` then `apply_cache_on_pipe`. Presence marks transformer with `_is_cached` (checked to decide compile mode and graph breaks). Export path disallowed—document this clearly in help text if modified.

## Inductor Tuning Flags
Set only if not disabled: `conv_1x1_as_mm=True`, `epilogue_fusion=False`, `coordinate_descent_tuning=True`, `coordinate_descent_check_all_directions=True`. Place additional experimental flags here (keep grouped). Avoid side effects after compile/export.

## Shape / Example Constraints (export)
Export uses hardcoded example tensors (resolution 1024x1024, specific sequence lengths). Changing resolution, guidance, or sequence length requires updating shapes inside `use_export_aoti` (both transformer and decoder example kwargs) to regenerate binaries. Missing update => silent mismatches or runtime errors. Add new args by extending `transformer_kwargs` and mirroring warmup logic.

## Kontext Differences
Kontext adds image input and doubled latent spatial tokens (`4096 * 2` for some tensors). Logic branches on `"Kontext" in pipeline.__class__.__name__`—retain this heuristic if adding subclass-based behavior. Infer `is_timestep_distilled` from `pipeline.transformer.config.guidance_embeds` (guidance None -> distilled Schnell).

## Profiling Workflow
To produce a Chrome trace: run `run_benchmark.py --trace-file trace.json.gz ...`; function wrappers from `annotate()` label regions: denoising_step, decoding, prompt_encoding, postprocessing, pil_conversion. Add new labeled regions by wrapping additional pipeline methods *after* warmup but before invoking profiler.

## Randomness & Repro
`set_rand_seeds()` seeds `random` + `torch`. Inference calls pass a fixed `generator=torch.manual_seed(seed)`—maintain this pattern when adding new sampling logic.

## Adding a New Optimization (Example Pattern)
1. Add flag: `--disable_my_feature` (default False -> enabled).
2. Implement in `optimize()` right before quantization if it alters module structure; after quantization if purely runtime scheduling.
3. Guard with `if not args.disable_my_feature:`.
4. Ensure interaction rules (e.g., works with compile but not export) and print a clear message if incompatible.

## External Dependencies & Version Sensitivities
Relies on PyTorch nightly (>=2.8 dev), `torchao` nightly, `diffusers` with specific upstream PRs, Flash Attention v3 (NVIDIA) or AITER (AMD), optional `cache-dit`. When scripting automation, surface informative errors if imports fail (see ImportError patterns already present). Avoid swallowing import errors silently.

## Safe Edits
- Avoid changing default ON behavior unless performance regressions are proven.
- Keep flag names stable; scripts and blog post may reference them.
- When modifying export shapes or filenames, mirror hosted artifact naming if expecting remote download (`download_hosted_file`).

## Quick Commands
Generate image (NVIDIA compile/export path):
`python gen_image.py --prompt "An astronaut standing next to a giant lemon" --use-cached-model`
Benchmark with trace:
`python run_benchmark.py --trace-file trace.json.gz --ckpt black-forest-labs/FLUX.1-dev --num_inference_steps 28`
Use cache-dit:
`python run_benchmark.py --ckpt black-forest-labs/FLUX.1-dev --num_inference_steps 28 --cache_dit_config cache_config.yaml --compile_export_mode compile`

## When Things Break
- Black images on AMD: ensure `dynamic=True` compile path retained.
- Export binary mismatch: delete cache dir (`~/.cache/flux-fast`) and rerun without `--use-cached-model`.
- FA3 import error: install Flash Attention v3 (NVIDIA) or switch to AMD with AITER installed.
- Quantization quality concerns: re-run with `--disable_quant`.

Feedback welcome—let us know if any implicit workflow isn't documented here so we can refine these instructions.
