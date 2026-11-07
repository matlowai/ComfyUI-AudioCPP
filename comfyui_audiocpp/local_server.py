"""An audiocpp_server that ComfyUI starts on this computer for Nemotron Transcribe, bound to 127.0.0.1.

The executable comes from the AUDIOCPP_SERVER environment variable or PATH, never from a workflow, so a shared workflow
cannot make ComfyUI run a program. The model is a GGUF that ComfyUI's model folders list. One process per port; a node that
asks for the same settings again gets the running one.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from . import client

MODEL_ID = "nemotron-transcribe"
# Session options per memory mode (audio.cpp docs/models/nemotron_transcribe.md, "CPU weight offload").
MEMORY = {
    "all on GPU": {},
    "experts in pinned RAM": {"nemotron_transcribe.disk_experts": True, "nemotron_transcribe.pin_experts": True,
                              "nemotron_transcribe.prefill_batch_size": 1024},
    "experts in RAM": {"nemotron_transcribe.disk_experts": True, "nemotron_transcribe.prefill_batch_size": 1024},
}
GPU = re.compile(r"^(\d+|GPU-[0-9a-fA-F-]{36})$")

_running: dict[int, tuple[subprocess.Popen, dict, str]] = {}


def executable() -> str:
    path = os.environ.get("AUDIOCPP_SERVER") or shutil.which("audiocpp_server")
    if not path or not os.path.isfile(path):
        raise client.AudioCPPError("audiocpp_server was not found: set the AUDIOCPP_SERVER environment variable to its path (for example "
                                   "<audio.cpp>/build/release/bin/audiocpp_server, built with scripts/build_audiocpp.sh) or put it on PATH, "
                                   "then restart ComfyUI")
    return path


def config(model_path: str, port: int, memory: str, idle_unload_ms: int) -> dict:
    return {"host": "127.0.0.1", "port": port, "backend": "cuda", "device": 0, "lazy_load": True, "max_loaded_models": 1,
            "idle_unload_ms": idle_unload_ms,
            "models": [{"id": MODEL_ID, "family": "nemotron_transcribe", "path": model_path, "task": "asr", "mode": "offline",
                        "session_options": MEMORY[memory]}]}


def start(model_path: str, port: int, gpu: str, memory: str, idle_unload_ms: int, workdir: str, wait: float = 120.0) -> None:
    """Start (or keep) the server on `port` with these settings. `gpu`: a CUDA index or a GPU-<uuid>, as nvidia-smi -L shows."""
    if not GPU.match(gpu):
        raise ValueError(f"gpu must be a CUDA device index or a GPU-<uuid> from nvidia-smi -L, not {gpu!r}")
    cfg = config(model_path, port, memory, idle_unload_ms)
    current = _running.get(port)
    if current and current[0].poll() is None and current[1] == cfg and current[2] == gpu:
        return
    stop(port)
    if _port_in_use(port):
        raise client.AudioCPPError(f"port {port} is already used by a program this ComfyUI did not start (another ComfyUI's server, or one "
                                   f"left running): stop it, choose another port, or connect to it with the audio.cpp Server node")
    os.makedirs(workdir, exist_ok=True)
    cfg_path = os.path.join(workdir, f"server_{port}.json")
    log_path = os.path.join(workdir, f"server_{port}.log")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([executable(), "--config", cfg_path, "--no-ui", "--log"], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu),  # indices as nvidia-smi numbers them
                                preexec_fn=_die_with_comfyui if sys.platform.startswith("linux") else None)
    _running[port] = (proc, cfg, gpu)
    server = client.Server(port, 5.0)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            del _running[port]
            with open(log_path, encoding="utf-8", errors="replace") as f:
                raise client.AudioCPPError(f"audiocpp_server exited with code {proc.returncode} while starting. Its log ({log_path}):\n{f.read()}")
        try:
            client.health(server)
            return
        except client.AudioCPPError:
            time.sleep(0.5)
    stop(port)
    raise client.AudioCPPError(f"audiocpp_server did not answer on port {port} within {wait:g} s; see {log_path}")


def _port_in_use(port: int) -> bool:
    try:
        client.health(client.Server(port, 2.0))
    except client.AudioCPPError as error:
        return not isinstance(error.__cause__, ConnectionRefusedError)
    return True


def _die_with_comfyui():
    """Linux: the server gets SIGTERM when the thread that started it ends (ComfyUI's prompt worker, alive as long as ComfyUI), also
    when ComfyUI is killed and atexit never runs."""
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def stop(port: int) -> bool:
    """Stop the server this ComfyUI started on `port`; False when it started none there."""
    current = _running.pop(port, None)
    if current is None:
        return False
    proc = current[0]
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return True


@atexit.register
def _stop_all():
    for port in list(_running):
        stop(port)
