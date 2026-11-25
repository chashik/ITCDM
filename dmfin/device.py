"""Choice of the device for the PyTorch work of the labs.

A graphics processor pays off only when one training or inference step is large enough to
hide the cost of launching its kernels (measured by ``python -m dmfin.bench_devices``, table
in the introduction to the practical part):

* GPU, if there is one: the global networks of Lab 6 (batches of 1 024 windows of 62 series),
  the pretrained forecasters of Lab 6 (Chronos), the text classifiers of Lab 3 (above all the
  fine-tuning of a pretrained transformer) and the recurrent networks of Lab 8 with at least
  ``GPU_HIDDEN`` hidden units;
* CPU: the small recurrent networks and autoencoders of Lab 6, the multilayer perceptrons of
  Lab 3 and of the searches of Lab 8 and the Q-networks of Lab 7.  Their steps are so small
  that the GPU is slower than one CPU core, and the labs train several of them at a time in
  parallel processes.

``DMFIN_DEVICE=cpu`` disables the GPU everywhere.

CPU and GPU work are not run at the same time unless ``DMFIN_OVERLAP=1``: all CPU cores under full load
together with the load steps of the GPU exceed what some power supplies deliver (the author's computer
switched off), and the results do not depend on the order of the jobs, only the running time does.
"""
from __future__ import annotations

import os

GPU_HIDDEN = 64          # recurrent networks at least this wide are trained on the GPU (Lab 8)


def overlap_allowed() -> bool:
    """May CPU-heavy work run while the GPU works?  Off unless DMFIN_OVERLAP=1."""
    return os.environ.get("DMFIN_OVERLAP", "0") == "1"


def gpu_available() -> bool:
    """True if a CUDA or ROCm device is present and not disabled by DMFIN_DEVICE=cpu."""
    if os.environ.get("DMFIN_DEVICE", "auto") == "cpu":
        return False
    import torch
    return torch.cuda.is_available()


def large_work_device() -> str:
    """Device for the large workloads: global networks and pretrained forecasters."""
    return "cuda" if gpu_available() else "cpu"


def rnn_device(hidden: int, gpu: bool | None = None) -> str:
    """Device for a recurrent network of the given width (Lab 8)."""
    gpu = gpu_available() if gpu is None else gpu
    return "cuda" if gpu and hidden >= GPU_HIDDEN else "cpu"
