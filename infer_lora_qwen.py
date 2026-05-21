from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_MIN_PAGEFILE_GB = 16


def _check_memory_preflight() -> None:
    if platform.system() != "Windows":
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        total_page_gb = mem.ullTotalPageFile / (1024**3)
        avail_page_gb = mem.ullAvailPageFile / (1024**3)
        phys_gb = mem.ullTotalPhys / (1024**3)
        print(
            f"[mem] physical={phys_gb:.1f}GB  pagefile_total={total_page_gb:.1f}GB  pagefile_avail={avail_page_gb:.1f}GB",
            flush=True,
        )
        if total_page_gb < _MIN_PAGEFILE_GB:
            print(
                f"[ERROR] page file {total_page_gb:.1f}GB < {_MIN_PAGEFILE_GB}GB, model load may fail (os error 1455).",
                flush=True,
            )
            sys.exit(1)
    except Exception:
        pass


def _project_root_from_config(cfg_path: Path) -> Path:
    p = cfg_path.resolve().parent
    while p != p.parent:
        if (p / "configs").is_dir():
            return p
        p = p.parent
    return cfg_path.resolve().parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen base + LoRA inference")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--base", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--no-adapter", action="store_true", default=False)
    args = p.parse_args()

    cfg: dict[str, Any] = {}
    cfg_path: Path | None = None
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg = load_yaml(cfg_path)

    root = _project_root_from_config(cfg_path) if cfg_path else Path.cwd().resolve()

    m = cfg.get("model") or {}
    base = args.base if args.base is not None else m.get("base")
    if base is None:
        base = "Qwen/Qwen2.5-7B-Instruct"

    use_adapter = not args.no_adapter and bool(cfg.get("load_adapter", True))
    adapter_path: Path | None = None
    if use_adapter:
        adapter = args.adapter if args.adapter is not None else cfg.get("adapter")
        if not adapter:
            raise SystemExit("missing adapter: use --adapter or set adapter in config yaml")
        adapter_path = Path(adapter)
        if not adapter_path.is_absolute():
            adapter_path = (root / adapter_path).resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            ckpts = sorted(adapter_path.glob("checkpoint-*"), key=os.path.getmtime)
            if ckpts and (ckpts[-1] / "adapter_config.json").is_file():
                adapter_path = ckpts[-1]
                print(f"[infer] auto-selected latest checkpoint: {adapter_path}", flush=True)
            else:
                raise SystemExit(
                    f"adapter_config.json not found in {adapter_path}\n"
                    "point to a directory containing adapter_config.json (usually checkpoint-*)."
                )

    pr = cfg.get("prompt") or {}
    user_msg = args.user if args.user is not None else pr.get("user", "用一句话介绍大语言模型。")

    gen = cfg.get("generation") or {}
    max_new = args.max_new_tokens if args.max_new_tokens is not None else int(gen.get("max_new_tokens", 128))
    temperature = float(gen.get("temperature", 0.7))
    top_p = float(gen.get("top_p", 0.9))
    do_sample = bool(gen.get("do_sample", True))

    bnb_cfg = cfg.get("bnb") or {}
    compute_dtype = getattr(torch, bnb_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    device_mode = str(m.get("device", "gpu")).strip().lower()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bnb_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    trust_remote_code = bool(m.get("trust_remote_code", True))

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw: dict[str, Any] = dict(trust_remote_code=trust_remote_code, low_cpu_mem_usage=True)

    if device_mode == "gpu":
        load_kw["device_map"] = {"": 0}
        load_kw["quantization_config"] = bnb_config
        print("[infer] device mode: GPU (forced gpu:0, 4bit)", flush=True)
    elif device_mode == "auto":
        load_kw["device_map"] = "auto"
        load_kw["quantization_config"] = bnb_config
        print("[infer] device mode: AUTO (accelerate auto dispatch, 4bit)", flush=True)
    elif device_mode == "cpu":
        load_kw["device_map"] = {"": "cpu"}
        load_kw["torch_dtype"] = torch.float32
        print("[infer] device mode: CPU (no 4bit quantization)", flush=True)
    else:
        raise SystemExit(f"unknown model.device: '{device_mode}', expected gpu/cpu/auto")

    _check_memory_preflight()
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    except OSError as e:
        if "1455" in str(e):
            print(
                "\n[ERROR] insufficient page file / swap for model load.\n"
                f"original error: {e}",
                flush=True,
            )
            sys.exit(1)
        raise
    if use_adapter:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"[infer] adapter loaded: {adapter_path}", flush=True)
    else:
        print("[infer] base-only mode (no adapter)", flush=True)
    model.eval()

    messages = [{"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
