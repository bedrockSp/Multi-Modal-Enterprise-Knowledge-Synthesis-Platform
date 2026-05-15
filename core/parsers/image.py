import asyncio
import os
import sys
import time
from typing import Optional

import easyocr
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract

from core.constants import EASYOCR_GPU, EASYOCR_WORKERS, TESSERACT_WORKERS

# Max pixel dimension for images fed to EasyOCR GPU.
# A 4096×4096 RGB image ≈ 48 MB in RAM; EasyOCR's CRAFT net expands this ~6×
# during inference (~300 MB VRAM) which is safe.  Without this cap, a 20000×15000
# GIF (like slide2_img35.gif) would need ~29 GiB of VRAM and OOM the GPU.
_MAX_IMAGE_DIM = 4096

# Formats PIL cannot load — skip these early to avoid loader errors
_UNSUPPORTED_EXTENSIONS = {".wmf", ".emf"}

# Enable cuDNN benchmark for consistent image sizes (auto-tunes GPU kernels)
if EASYOCR_GPU:
    try:
        import torch

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# On Windows, pytesseract auto-detects the binary only if it's on PATH. The
# UB-Mannheim installer does not add itself to PATH by default, so we point
# pytesseract at the standard install path when present.
if sys.platform == "win32" and not pytesseract.pytesseract.tesseract_cmd.endswith(".exe"):
    _default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _default_tesseract

_EASYOCR_SEMAPHORE = None
_EASYOCR_SEMAPHORE_LOCK = asyncio.Lock()
_TESSERACT_SEMAPHORE = None
_TESSERACT_SEMAPHORE_LOCK = asyncio.Lock()
_EASYOCR_READER = None
_EASYOCR_READER_LOCK = asyncio.Lock()


async def _get_easyocr_reader():
    """Return a cached EasyOCR Reader instance (avoids reloading ~200MB model)."""
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    async with _EASYOCR_READER_LOCK:
        if _EASYOCR_READER is None:
            _EASYOCR_READER = await asyncio.to_thread(
                lambda: easyocr.Reader(["en"], gpu=EASYOCR_GPU)
            )
    return _EASYOCR_READER


async def get_easyocr_semaphore() -> asyncio.Semaphore:
    global _EASYOCR_SEMAPHORE
    if _EASYOCR_SEMAPHORE is not None:
        return _EASYOCR_SEMAPHORE
    async with _EASYOCR_SEMAPHORE_LOCK:
        if _EASYOCR_SEMAPHORE is None:
            _EASYOCR_SEMAPHORE = asyncio.Semaphore(EASYOCR_WORKERS)
    return _EASYOCR_SEMAPHORE


async def get_tesseract_semaphore() -> asyncio.Semaphore:
    global _TESSERACT_SEMAPHORE
    if _TESSERACT_SEMAPHORE is not None:
        return _TESSERACT_SEMAPHORE
    async with _TESSERACT_SEMAPHORE_LOCK:
        if _TESSERACT_SEMAPHORE is None:
            _TESSERACT_SEMAPHORE = asyncio.Semaphore(TESSERACT_WORKERS)
    return _TESSERACT_SEMAPHORE


def _prepare_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load, normalize, and downscale an image for OCR.

    Handles:
    - Unsupported formats (WMF/EMF) → returns None
    - Palette images with transparency → converts to RGBA then RGB
    - Oversized images → downscales to _MAX_IMAGE_DIM to prevent GPU OOM
    - Animated GIFs → uses first frame only

    Returns a BGR numpy array (what EasyOCR expects) or None if unsupported.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in _UNSUPPORTED_EXTENSIONS:
        print(f"[ImagePrep] Skipping unsupported format: {ext} ({os.path.basename(image_path)})")
        return None

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"[ImagePrep] Cannot open {os.path.basename(image_path)}: {e}")
        return None

    # Animated GIF: use first frame
    if getattr(img, "is_animated", False):
        img.seek(0)

    # Convert palette/transparency to RGBA then RGB
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        # Composite onto white background to drop alpha channel
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downscale if any dimension exceeds the cap
    w, h = img.size
    if max(w, h) > _MAX_IMAGE_DIM:
        scale = _MAX_IMAGE_DIM / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"[ImagePrep] Downscaling {os.path.basename(image_path)} from {w}×{h} to {new_w}×{new_h}")
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # EasyOCR expects BGR numpy array (or file path, but array avoids re-read)
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # RGB → BGR
    return arr


async def image_parser(image_path: str) -> str:
    """
    OCR pipeline (no VLM):
    1. Primary: EasyOCR with spatial sorting (bounding-box aware)
    2. Fallback: Tesseract with image preprocessing
    """

    # Pre-load and normalize image once (handles palette, transparency, oversized, WMF)
    prepared = await asyncio.to_thread(_prepare_image, image_path)
    if prepared is None:
        print(f"[OCR] Skipping unsupported image: {os.path.basename(image_path)}")
        return ""

    async def easyocr_parse() -> str:
        """OCR using EasyOCR with spatial sorting for tables/flowcharts."""
        try:
            semaphore = await get_easyocr_semaphore()
            async with semaphore:
                reader = await _get_easyocr_reader()
                # batch_size=8 for GPU (processes multiple text regions in parallel)
                batch_size = 8 if EASYOCR_GPU else 1
                result = await asyncio.to_thread(
                    lambda: reader.readtext(prepared, batch_size=batch_size)
                )

                if not result:
                    return ""

                # Sort by Y-position (top→bottom), then X (left→right)
                # This preserves table row order and flowchart structure
                sorted_results = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))

                text_lines = [item[1] for item in sorted_results]
                return "\n".join(text_lines)
        except Exception as e:
            print(f"[EasyOCR] Exception: {e}")
            return ""

    async def tesseract_parse() -> str:
        """Fallback OCR with Tesseract + image preprocessing."""
        try:
            semaphore = await get_tesseract_semaphore()
            async with semaphore:

                def _preprocess_and_ocr():
                    # Convert BGR numpy array back to PIL for Tesseract preprocessing
                    arr = prepared
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        arr = arr[:, :, ::-1]  # BGR → RGB
                    img = Image.fromarray(arr)
                    # Convert to grayscale
                    img = img.convert("L")
                    # Boost contrast
                    img = ImageEnhance.Contrast(img).enhance(2.0)
                    # Binary threshold for cleaner text edges
                    img = img.point(lambda x: 0 if x < 128 else 255)
                    return pytesseract.image_to_string(img)

                return await asyncio.to_thread(_preprocess_and_ocr)
        except Exception as e:
            print(f"[Tesseract] Exception: {e}")
            return ""

    # ---- Primary: EasyOCR ----
    try:
        start_time = time.time()
        print(f"Processing image: {os.path.basename(image_path)} with EasyOCR")
        easyocr_result = await easyocr_parse()
        if easyocr_result and easyocr_result.strip():
            elapsed = time.time() - start_time
            print(
                f"[EasyOCR] Succeeded in {elapsed:.2f}s for {os.path.basename(image_path)}"
            )
            return easyocr_result.strip()
    except Exception as e:
        print(f"[EasyOCR] Exception: {e}")

    # ---- Fallback: Tesseract ----
    try:
        print(
            f"EasyOCR failed or returned empty, falling back to Tesseract for {os.path.basename(image_path)}"
        )
        start_time = time.time()
        result = (await tesseract_parse()).strip()
        elapsed = time.time() - start_time
        print(
            f"[Tesseract] Completed in {elapsed:.2f}s for {os.path.basename(image_path)}"
        )
        return result
    except Exception as e:
        print(f"[Tesseract] Fatal exception: {e}")
        return ""

