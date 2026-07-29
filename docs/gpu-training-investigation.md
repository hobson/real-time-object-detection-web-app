# GPU (ROCm) training on taco — investigation notes

taco's iGPU is an AMD Radeon 8060S (gfx1151, "Strix Halo" / Ryzen AI Max+
395). `training/train_from_db.py` and `training/compute_label_confidence.py`
both default to `device="cpu"` for all PyTorch/ultralytics calls. This doc
tracks why, and what's been tried, so that detail doesn't have to live in
code comments (which rot silently as the actual GPU/driver/torch state
changes) or CLI `--help` text.

## 2026-07-28: ruled out GPU contention; found the real cause

Initial hypothesis was that taco's `llama-multi-models.service` (llama-swap,
which loads a vision LLM onto the same iGPU on demand) was blocking other
processes from using the GPU. Ruled out directly:

- A brand-new, completely idle Python process running nothing but
  `torch.randn(64,64,device='cuda') @ ...` fails with `HIP error: invalid
  device function` even while llama-swap's model sits idle in VRAM (0% GPU
  utilization at the time). Zero contention, same crash.
- `torch.cuda.get_arch_list()` on `training/.venv`'s torch (2.9.1+rocm6.4)
  does not include `gfx1151` at all: `['gfx900', 'gfx906', 'gfx908',
  'gfx90a', 'gfx942', 'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200',
  'gfx1201']`. The wheel simply has no compiled kernels for this GPU.
- `HSA_OVERRIDE_GFX_VERSION` spoofing (tried 11.0.0/11.0.1/11.0.2, targeting
  the closest already-compiled gfx11xx archs) changes the error to `HIP
  error: no kernel image is available for execution on the device` - still
  fails.
- Corroborating evidence: `llama.cpp`'s own ROCm build (which *does* run on
  this GPU - it's actively serving requests) was compiled with
  `GPU_TARGETS=gfx1151` explicitly (`~/code/hobs/llama.cpp/build/CMakeCache.txt`
  on taco). It works because it was built for this exact hardware; pip's
  official PyTorch ROCm wheels weren't.
- A sibling project on this same host (`~/code/corethink/retrosynformer`)
  pins `torch==2.5.1+rocm6.2` in its `pyproject.toml` with a comment
  claiming it was "verified working" on taco. Re-tested directly: **it now
  fails identically.** Likely cause: taco has since been upgraded to Ubuntu
  26.04 / kernel 7.0.0 with **ROCm 7.1 userspace libraries installed
  system-wide** (`hsa-rocr` 7.1.0, `rocblas` 7.1.0, `miopen` 7.1.1) - a
  pip-vendored rocm6.2 build's bundled libraries are no longer ABI-compatible
  with the current system driver stack. This is a second, separate problem
  layered on top of the missing-gfx1151-kernel issue.

## Candidate fix, in progress

PyTorch publishes `rocm7.1`/`rocm7.2` wheels (up to torch 2.13.0, `cp314` -
matching taco's system Python 3.14 and current ROCm 7.1 userspace exactly).
Testing `torch==2.13.0+rocm7.1` in an isolated venv (via `uv venv` /
`uv pip install` - use `uv` for all Python package/venv management on this
project per project convention, not raw `pip`/`venv`) at
`/tmp/rocm7_test_venv` on taco, touching no project venv. Once installed,
re-running the same matmul/conv2d/backward smoke test to check both (a) ABI
compatibility with the current ROCm 7.1 system libraries and (b) whether
gfx1151 is actually in this build's compiled kernel list.

**Do not modify `training/.venv` (or any other in-use project venv) while
Phase G's background training job is running** - it's actively using that
exact venv. Any real fix (installing a working torch+rocm build into
`training/.venv`) waits until that job is confirmed finished.
