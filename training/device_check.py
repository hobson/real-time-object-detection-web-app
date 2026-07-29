"""Verifies a requested torch device (e.g. "cuda", "rocm", a GPU index) is
actually usable before handing it to ultralytics - falls back to "cpu" with
a warning otherwise.

See docs/gpu-training-investigation.md for why this exists: taco's
currently-installed torch/ROCm build silently accepts a GPU device
selection (torch.cuda.is_available() returns True, model.train()/predict()
start normally) but then crashes with a HIP error or segfaults partway
through real work, because the installed build is missing compiled kernels
for taco's GPU (gfx1151) or is ABI-mismatched against the system's ROCm
runtime. Losing hours of training to that crash after the fact is far worse
than a cheap up-front check - so every script that accepts a --device
override resolves it through here before use, not passing the raw CLI
value straight to ultralytics/torch.

Every location in this codebase that sets a torch device to something other
than a hardcoded "cpu" goes through resolve_device() first:
  - train_from_db.py (model.train)
  - compute_class_fn_fp_rates.py, review_confusion_matrix.py,
    compute_label_confidence.py (model.predict)
"""
import logging

logger = logging.getLogger("device_check")


def resolve_device(requested: str) -> str:
    """Returns `requested` unchanged only if it passes every check below;
    otherwise logs why and returns "cpu". Each check is independently
    try/excepted so a failure in one (e.g. an attribute that doesn't exist
    on a particular torch build) degrades gracefully to the next check
    rather than raising - the real matmul/conv2d/backward smoke test at the
    end is the authoritative determinant regardless of what the earlier,
    cheaper checks find, since even a version/arch match doesn't guarantee
    the installed build actually works on this exact machine (see
    docs/gpu-training-investigation.md - a "verified working" pin on this
    same host stopped working after a system ROCm upgrade)."""
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except Exception as e:
        logger.warning("Could not import torch (%s) - falling back to cpu.", e)
        return "cpu"

    try:
        if not torch.cuda.is_available():
            logger.warning("torch.cuda.is_available() is False for device %r - falling back to cpu.", requested)
            return "cpu"
    except Exception as e:
        logger.warning("torch.cuda.is_available() raised %s - falling back to cpu.", e)
        return "cpu"

    try:
        rocm_version = getattr(getattr(torch, "version", None), "hip", None)
        if rocm_version is None:
            logger.warning(
                "Requested device %r but this torch build has no ROCm/HIP support "
                "(torch.version.hip is None) - falling back to cpu.", requested,
            )
            return "cpu"
        logger.info("torch %s built against ROCm/HIP %s", torch.__version__, rocm_version)
    except Exception as e:
        logger.warning("Could not read torch.version.hip (%s) - continuing to the next check anyway.", e)

    try:
        props = torch.cuda.get_device_properties(0)
        gpu_arch = getattr(props, "gcnArchName", None)
        compiled_archs = torch.cuda.get_arch_list()
        if gpu_arch and compiled_archs and gpu_arch not in compiled_archs:
            logger.warning(
                "GPU architecture %r is not in this torch build's compiled kernel list %r "
                "- falling back to cpu. See docs/gpu-training-investigation.md.",
                gpu_arch, compiled_archs,
            )
            return "cpu"
        logger.info("GPU %r (arch %r) is in this torch build's compiled kernel list.", getattr(props, "name", "?"), gpu_arch)
    except Exception as e:
        logger.warning("Could not verify GPU architecture against compiled kernels (%s) - continuing to the smoke test anyway.", e)

    try:
        import torch.nn as nn

        a = torch.randn(64, 64, device=requested)
        b = torch.randn(64, 64, device=requested)
        _ = a @ b

        conv = nn.Conv2d(3, 8, 3).to(requested)
        x = torch.randn(1, 3, 32, 32, device=requested, requires_grad=True)
        y = conv(x)
        y.sum().backward()

        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(
            "Requested device %r failed a matmul/conv2d/backward smoke test (%s) - "
            "falling back to cpu. See docs/gpu-training-investigation.md.",
            requested, e,
        )
        return "cpu"

    logger.info("Device %r passed all checks - using it.", requested)
    return requested
