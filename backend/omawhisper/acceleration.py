"""Resolve engine devices without confusing the microphone with compute hardware."""

import ctypes
import os

from .config import UserError


def cuda_available() -> tuple[bool, str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False, "CUDA devices are hidden by CUDA_VISIBLE_DEVICES."
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False, "NVIDIA CUDA driver is unavailable. Install the driver for this machine or choose CPU."
    result = driver.cuInit(0)
    if result:
        return False, f"NVIDIA CUDA driver initialization failed (code {result}). Check the driver or choose CPU."
    version = ctypes.c_int()
    result = driver.cuDriverGetVersion(ctypes.byref(version))
    if result or version.value < 12000:
        return False, "The installed speech runtimes require a CUDA 12-capable NVIDIA driver. Update the driver or choose CPU."
    count = ctypes.c_int()
    result = driver.cuDeviceGetCount(ctypes.byref(count))
    if result or count.value == 0:
        return False, f"No usable NVIDIA CUDA device (code {result}). Check the driver or choose CPU."
    return True, "NVIDIA CUDA device available."


def onnx_gpu_compatible(quantization: str | None) -> bool:
    return not quantization or quantization.lower() in {"fp16", "float16", "fp32", "float32", "bf16", "bfloat16"}


def whisper_runtime(ctranslate2, config: dict) -> dict:
    requested = config["acceleration"]
    device = "cpu"
    detail = "CPU selected."
    if requested != "cpu":
        try:
            available = ctranslate2.get_cuda_device_count() > 0
        except RuntimeError as exc:
            raise UserError(f"Cannot detect CUDA devices: {exc}. Check the NVIDIA driver or select CPU.") from exc
        if available:
            device = "cuda"
            detail = "NVIDIA CUDA selected."
        elif requested == "gpu":
            raise UserError("No supported CUDA GPU detected. Whisper GPU mode requires an NVIDIA GPU, "
                            "a working driver and a CUDA-enabled CTranslate2 build. Select Auto or CPU otherwise.")
        else:
            # CTranslate2 also returns zero for driver/runtime enumeration failures.
            detail = "Auto selected CPU: CUDA unavailable to CTranslate2 (GPU, driver or runtime)."
    try:
        supported = ctranslate2.get_supported_compute_types(device)
    except RuntimeError as exc:
        raise UserError(f"Cannot initialize {device.upper()}: {exc}. Check the runtime requirements or select CPU.") from exc
    compute = config["compute_type"]
    if compute == "auto":
        preferences = ("float16", "float32") if device == "cuda" else ("int8", "int8_float32", "float32")
        compute = next((value for value in preferences if value in supported), None)
    if compute not in supported:
        raise UserError(f"Whisper precision {config['compute_type']} is not supported on {device.upper()}. "
                        f"Choose Auto or one of: {', '.join(sorted(supported))}.")
    return {"device": device, "provider": "CTranslate2", "compute_type": compute, "detail": detail}
