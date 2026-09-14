# Local patches and model-file edits (not part of any upstream)

Reapply these after reinstalling SGLang or re-downloading a model. `.orig`
copies of every edited file sit next to the live file in `~/qwen35`.

## `sglang-0.5.19-gemma4_unified.patch`
`sglang/srt/models/gemma4_unified.py` in `~/qwen35/.venv`:
1. `self.lm_head_is_tied` was never set -> AttributeError during decode CUDA graph capture.
2. The audio embedder used `output_proj_dims` (1536) instead of `audio_embed_dim` (640)
   -> weight-shape assert on real load. Invisible to `--load-format dummy`.

Apply: `cd ~/qwen35/.venv/lib/python3.12/site-packages && patch -p0 < ~/bench/patches/sglang-0.5.19-gemma4_unified.patch`

## `model-config-edits.patch`
- `mellum2-12b`, `Qwen3.5-9B-AWQ-4bit`, `mellum2-12b-awq` generation_config.json: vendor
  sampling added. SGLang fills unset chat sampling from this file, then OpenAI defaults (no
  top-k), and clients that omit top_k (possibly omp) got corrupted tool-call JSON.
- `gemma4-12b-nvfp4`, `gemma4-12b-awq` config.json: `vision_config.model_patch_size: 48`
  (transformers derives it as patch_size x pooling_kernel_size; SGLang needs it explicit).
