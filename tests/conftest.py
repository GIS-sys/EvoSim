import os
import sys

import pytest
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
wp.config.quiet = True
wp.init()


@pytest.fixture(scope="session")
def device():
    """CUDA if available, else CPU. Override with EVOSIM_DEVICE=cpu|cuda:0."""
    d = os.environ.get("EVOSIM_DEVICE")
    if d:
        return d
    return "cuda:0" if wp.is_cuda_available() else "cpu"
