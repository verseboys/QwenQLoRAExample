from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


class TrainingCompletionProgressCallback(TrainerCallback):
    def __init__(self) -> None:
        self._pbar: tqdm | None = None

    def _ensure_pbar(self, state: TrainerState) -> None:
        if self._pbar is not None:
            return
        total = int(getattr(state, "max_steps", 0) or 0)
        if total <= 0:
            return
        initial = min(int(getattr(state, "global_step", 0) or 0), total)
        self._pbar = tqdm(
            total=total,
            initial=initial,
            desc="train",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._ensure_pbar(state)
        if self._pbar is not None and self._pbar.total:
            step = min(int(getattr(state, "global_step", 0) or 0), int(self._pbar.total))
            if step > self._pbar.n:
                self._pbar.update(step - self._pbar.n)
            elif step < self._pbar.n:
                self._pbar.n = step
                self._pbar.refresh()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._pbar is not None:
            if self._pbar.total and self._pbar.n < self._pbar.total:
                self._pbar.n = self._pbar.total
                self._pbar.refresh()
            self._pbar.close()
            self._pbar = None


_DEFAULT_MIN_PAGEFILE_GB = 16


def _check_memory_preflight(min_pagefile_gb: int = _DEFAULT_MIN_PAGEFILE_GB) -> None:
    if sys.platform != "win32":
        return
    try:
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

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

        phys_gb = stat.ullTotalPhys / (1024 ** 3)
        pagefile_total_gb = stat.ullTotalPageFile / (1024 ** 3)
        pagefile_only_gb = pagefile_total_gb - phys_gb

        print(
            f"[preflight] physical RAM: {phys_gb:.1f} GB | "
            f"page file (swap): {max(pagefile_only_gb, 0):.1f} GB | "
            f"total virtual: {pagefile_total_gb:.1f} GB | "
            f"required pagefile >= {min_pagefile_gb} GB",
            flush=True,
        )

        if pagefile_only_gb < min_pagefile_gb:
            print(
                f"\n{'='*72}\n"
                f"[ERROR] Windows page file is too small ({pagefile_only_gb:.1f} GB).\n"
                f"Loading the base model requires at least {min_pagefile_gb} GB of swap.\n\n"
                f"  Fix: increase virtual memory to at least 32 GB (recommended 64 GB):\n"
                f"    1. Win+R -> sysdm.cpl -> Advanced -> Performance Settings\n"
                f"    2. Advanced -> Virtual Memory -> Change\n"
                f"    3. Uncheck 'Automatically manage', select a drive with enough free space\n"
                f"    4. Custom size: Initial=32768 / Maximum=65536 (MB)\n"
                f"    5. Click Set -> OK -> Reboot\n"
                f"{'='*72}\n",
                flush=True,
            )
            sys.exit(1)
    except Exception:
        pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _project_root_from_config(cfg_path: Path) -> Path:
    p = cfg_path.resolve().parent
    while p != p.parent:
        if (p / "configs").is_dir():
            return p
        p = p.parent
    return cfg_path.resolve().parent.parent


def _resolve_model_path(name_or_path: str) -> str:
    candidate = Path(name_or_path)
    if not candidate.exists():
        if any(sep in name_or_path for sep in ("\\", "/")) and ":" in name_or_path:
            raise FileNotFoundError(
                f"Model path does not exist on disk: {name_or_path}\n"
                "Either fix the path in YAML, or download the model first via huggingface-cli."
            )
        return name_or_path

    if (candidate / "config.json").is_file():
        return str(candidate)

    snap_dir = candidate / "snapshots"
    if snap_dir.is_dir():
        snapshots = [
            s for s in snap_dir.iterdir()
            if s.is_dir() and (s / "config.json").is_file()
        ]
        if not snapshots:
            raise FileNotFoundError(
                f"No usable snapshot under {snap_dir}.\n"
                "The model cache looks incomplete (no snapshot has config.json). "
                "Re-download via huggingface-cli or delete the broken cache."
            )
        snapshots.sort(key=lambda s: s.stat().st_mtime, reverse=True)
        resolved = str(snapshots[0])
        print(
            f"[status] resolved HF cache root -> snapshot: {resolved}",
            flush=True,
        )
        return resolved

    raise FileNotFoundError(
        f"'{name_or_path}' exists but is neither a snapshot dir (no config.json) "
        "nor a HF cache root (no snapshots/). Point name_or_path at a directory "
        "that contains config.json, or at 'models--<org>--<name>/'."
    )


def _cell_to_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return "\n".join(_cell_to_str(x) for x in val).strip()
    return str(val).strip()


def build_chat_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    prompt_field: str,
    input_field: str,
    output_field: str,
    system_field: str | None,
) -> str:
    instruction = _cell_to_str(row.get(prompt_field))
    inp = _cell_to_str(row.get(input_field))
    output = _cell_to_str(row.get(output_field))
    if inp:
        user_content = f"{instruction}\n{inp}".strip()
    else:
        user_content = instruction
    system = ""
    if system_field:
        system = _cell_to_str(row.get(system_field))
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Tokenizer has no chat_template; use a chat-capable Instruct model."
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


class GracefulStopCallback(TrainerCallback):
    def __init__(self) -> None:
        self._stop_requested = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        if self._stop_requested:
            print("\n[stop] forced exit (second Ctrl+C)", flush=True)
            sys.exit(1)
        self._stop_requested = True
        print(
            "\n[stop] graceful stop requested — will save checkpoint after current step...",
            flush=True,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._stop_requested:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        signal.signal(signal.SIGINT, self._original_sigint)


def _find_latest_checkpoint(output_dir: str) -> str | None:
    out = Path(output_dir)
    if not out.is_dir():
        return None
    ckpts = sorted(out.glob("checkpoint-*"), key=os.path.getmtime)
    return str(ckpts[-1]) if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/qwen2.5_7b_instruct.yaml",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    print(f"[status] loading config: {cfg_path}", flush=True)
    cfg = load_yaml(cfg_path)

    runtime_cfg = cfg.get("runtime") or {}
    use_gpu = bool(runtime_cfg.get("use_gpu", True))
    min_pagefile_gb = int(runtime_cfg.get("min_pagefile_gb", _DEFAULT_MIN_PAGEFILE_GB))
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(
            "[runtime] use_gpu=false -> forcing CPU mode "
            "(4bit quant / bf16 / paged optim auto-disabled; needs >=30 GB pagefile)",
            flush=True,
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"[status] CUDA_VISIBLE_DEVICES={visible_devices if visible_devices is not None else '<not set>'}",
        flush=True,
    )
    if use_gpu and torch.cuda.is_available():
        gpu_list = [f"{i}:{torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
        print(f"[status] visible GPUs: {', '.join(gpu_list)}", flush=True)
    elif use_gpu:
        print(
            "[runtime] use_gpu=true but CUDA is unavailable -> auto fallback to CPU mode "
            "(4bit quant / bf16 / paged optim auto-disabled)",
            flush=True,
        )
        use_gpu = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print("[status] running in CPU-only mode by config", flush=True)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    bnb_cfg = cfg["bnb"]
    train_cfg = cfg["training"]

    resolved_model_path = _resolve_model_path(str(model_cfg["name_or_path"]))

    compute_dtype = getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"])

    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bnb_cfg["load_in_4bit"],
            bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        bnb_config = None

    _check_memory_preflight(min_pagefile_gb=min_pagefile_gb)

    print(f"[status] loading base model from: {resolved_model_path}", flush=True)
    print("[status] (this can take several minutes)", flush=True)
    try:
        if use_gpu:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                quantization_config=bnb_config,
                torch_dtype=compute_dtype,
                device_map={"": 0},
                trust_remote_code=model_cfg.get("trust_remote_code", False),
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=model_cfg.get("trust_remote_code", False),
                low_cpu_mem_usage=True,
            )
    except OSError as exc:
        if "1455" in str(exc):
            raise RuntimeError(
                "Model load failed: Windows page file is too small (os error 1455). "
                "Increase virtual memory (recommended >= 32768 MB, better 65536 MB), reboot, "
                "then retry training."
            ) from exc
        raise
    print("[status] base model loaded", flush=True)

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        devices_used = set(str(v) for v in device_map.values())
        on_cpu = sum(1 for v in device_map.values() if str(v) == "cpu")
        on_gpu = sum(1 for v in device_map.values() if str(v) != "cpu")
        print(
            f"[status] device map: {on_gpu} layers on GPU, {on_cpu} layers on CPU "
            f"(devices: {', '.join(sorted(devices_used))})",
            flush=True,
        )
        if on_cpu > 0 and on_gpu > 0:
            print(
                "[warn] model is split between GPU and CPU — training will be slow. "
                "Consider reducing max_seq_length or using a smaller model.",
                flush=True,
            )
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(
            f"[status] GPU memory after load: allocated={alloc_gb:.1f} GB, reserved={reserved_gb:.1f} GB",
            flush=True,
        )

    model_device = getattr(model, "device", None)
    if model_device is not None:
        if getattr(model_device, "type", "") == "cuda":
            dev_idx = model_device.index if model_device.index is not None else torch.cuda.current_device()
            print(
                f"[status] model primary device: cuda:{dev_idx} ({torch.cuda.get_device_name(dev_idx)})",
                flush=True,
            )
        else:
            print(f"[status] model primary device: {model_device}", flush=True)
    if train_cfg.get("gradient_checkpointing", True):
        if use_gpu:
            print("[status] enabling gradient checkpointing for k-bit training", flush=True)
            model = prepare_model_for_kbit_training(model)
        else:
            print("[status] enabling gradient checkpointing (CPU mode, no kbit prep)", flush=True)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    print("[status] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    pf = data_cfg["prompt_field"]
    inf = data_cfg["input_field"]
    of = data_cfg["output_field"]
    system_field = data_cfg.get("system_field") or None

    def formatting_func(batch: dict[str, Any]) -> str | list[str]:
        anchor = batch.get(pf)
        if anchor is None:
            anchor = batch.get(inf) or batch.get(of)
        if anchor is None:
            return []
        if not isinstance(anchor, list):
            return build_chat_prompt(tokenizer, batch, pf, inf, of, system_field)
        n = len(anchor)
        texts: list[str] = []
        for i in range(n):
            row: dict[str, Any] = {}
            for k, v in batch.items():
                if str(k).startswith("__"):
                    continue
                if isinstance(v, list):
                    row[k] = v[i] if i < len(v) else None
                else:
                    row[k] = v
            texts.append(build_chat_prompt(tokenizer, row, pf, inf, of, system_field))
        return texts

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    project_root = _project_root_from_config(cfg_path)
    train_file = Path(data_cfg["train_file"])
    if not train_file.is_absolute():
        train_file = (project_root / train_file).resolve()
    if not train_file.is_file():
        raise FileNotFoundError(f"train_file not found: {train_file}")

    print(f"[status] loading dataset: {train_file}", flush=True)
    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    valid_file = data_cfg.get("valid_file")
    if valid_file:
        valid_path = Path(valid_file)
        if not valid_path.is_absolute():
            valid_path = (project_root / valid_path).resolve()
        if not valid_path.is_file():
            raise FileNotFoundError(f"valid_file not found: {valid_path}")
        print(f"[status] loading eval dataset: {valid_path}", flush=True)
        eval_ds = load_dataset("json", data_files=str(valid_path), split="train")
    else:
        val_split = float(data_cfg.get("validation_split", 0))
        if val_split > 0:
            split = train_ds.train_test_split(test_size=val_split, seed=int(train_cfg.get("seed", 42)))
            train_ds, eval_ds = split["train"], split["test"]
        else:
            eval_ds = None
    print(
        f"[status] dataset ready | train={len(train_ds)} | eval={0 if eval_ds is None else len(eval_ds)}",
        flush=True,
    )

    max_len = int(data_cfg["max_seq_length"])

    if use_gpu:
        fp16_flag = bool(train_cfg.get("fp16", False))
        bf16_flag = bool(train_cfg.get("bf16", True))
        optim_name = train_cfg.get("optim", "paged_adamw_8bit")
    else:
        fp16_flag = False
        bf16_flag = False
        optim_name = "adamw_torch"
        print(
            "[runtime] CPU mode -> overriding fp16=False, bf16=False, optim=adamw_torch",
            flush=True,
        )
    train_kw: dict[str, Any] = dict(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        fp16=fp16_flag,
        bf16=bf16_flag,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=optim_name,
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        seed=int(train_cfg.get("seed", 42)),
        report_to=train_cfg.get("report_to", "none"),
        disable_tqdm=True,
        logging_strategy="steps",
        logging_first_step=True,
    )
    if not use_gpu:
        train_kw["use_cpu"] = True
    if eval_ds is not None:
        train_kw["eval_strategy"] = "steps"
        train_kw["eval_steps"] = int(train_cfg.get("eval_steps", train_cfg["save_steps"]))
        train_kw["load_best_model_at_end"] = True
        train_kw["metric_for_best_model"] = "eval_loss"
        train_kw["greater_is_better"] = False
    else:
        train_kw["eval_strategy"] = "no"

    if args.stop_after is not None and args.stop_after > 0:
        train_kw["max_steps"] = args.stop_after

    import inspect as _inspect
    _sft_params = set(_inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" in _sft_params:
        train_kw["max_seq_length"] = max_len
    elif "max_length" in _sft_params:
        train_kw["max_length"] = max_len
    if "packing" in _sft_params:
        train_kw["packing"] = False

    training_args = SFTConfig(**train_kw)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=[TrainingCompletionProgressCallback(), GracefulStopCallback()],
    )

    resume_ckpt: str | None = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_ckpt = _find_latest_checkpoint(train_cfg["output_dir"])
            if resume_ckpt:
                print(f"[status] resuming from latest checkpoint: {resume_ckpt}", flush=True)
            else:
                print("[status] --resume=auto but no checkpoint found, training from scratch", flush=True)
        else:
            resume_ckpt = args.resume
            print(f"[status] resuming from checkpoint: {resume_ckpt}", flush=True)

    print("[status] training started (Ctrl+C to save checkpoint and stop)...", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    out = Path(train_cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[status] saved adapter and tokenizer to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
