"""Device selection for torch-backed ML components.

Resolves the "auto" | "cpu" | "cuda" setting into a concrete device string,
detecting CUDA availability only when "auto" is requested so CPU-only boxes
without a torch CUDA build never trigger the lazy CUDA initializer.
"""

from typing import Literal


def resolve_device(setting: str) -> Literal["cpu", "cuda"]:
    s = (setting or "auto").strip().lower()
    if s == "cpu":
        return "cpu"
    if s == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
