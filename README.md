# QwenQLoRAExample

基于 Hugging Face 的 Qwen2.5 Instruct 模型 QLoRA 微调与 LoRA 推理示例。默认在 Linux + NVIDIA GPU 环境下运行；也支持 CPU 调试（需在配置中关闭 GPU 与 4bit 量化）。

## 环境要求

- Python 3.10+
- Linux（推荐 Ubuntu 22.04+）+ CUDA 12.x + 单卡显存 ≥ 16GB（7B + 4bit QLoRA）
- 可选：已安装 `huggingface-cli` 并登录，用于下载私有或 gated 模型

## 安装

```bash
cd /path/to/QwenQLoRAExample
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

首次使用建议下载基座权重（二选一）：

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

或在训练/推理配置里把 `name_or_path` / `base` 改成本地缓存目录，例如：

```text
/home/<user>/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
```

脚本会自动解析 `snapshots/<hash>/` 下的最新快照。

## 数据格式

训练数据为 JSONL，每行一条样本，字段与 `configs/train/qwen2.5_7b_instruct.yaml` 中 `data` 段一致：

| 字段 | 说明 |
|------|------|
| `instruction` | 用户任务描述 |
| `input` | 可选补充输入 |
| `output` | 期望的助手回复 |

示例见 `data/example_train.jsonl`。将 `train_file` / `valid_file` 指向你的文件即可。

## 训练（QLoRA）

编辑 `configs/train/qwen2.5_7b_instruct.yaml`：

- `model.name_or_path`：Hub ID（如 `Qwen/Qwen2.5-7B-Instruct`）或 Linux 本地路径
- `data.train_file` / `valid_file`：训练与验证 JSONL
- `training.output_dir`：LoRA 输出目录（默认 `outputs/qwen2.5-7b-qlora-run`）
- `runtime.use_gpu`：`false` 时强制 CPU（关闭 4bit / bf16 / paged 优化器）

启动训练：

```bash
python train_qlora_qwen.py --config configs/train/qwen2.5_7b_instruct.yaml
```

常用参数：

```bash
python train_qlora_qwen.py --config configs/train/qwen2.5_7b_instruct.yaml --resume auto
python train_qlora_qwen.py --config configs/train/qwen2.5_7b_instruct.yaml --stop-after 10
```

训练过程中按一次 `Ctrl+C` 会在当前步结束后保存 checkpoint 并退出；再按一次强制退出。

产物目录结构示例：

```text
outputs/qwen2.5-7b-qlora-run/
  adapter_config.json
  adapter_model.safetensors
  checkpoint-200/
  ...
```

## 推理（基座 + LoRA）

编辑 `configs/infer/qwen2.5_7b_lora.yaml`：

- `model.base`：须与训练时的基座一致
- `adapter`：含 `adapter_config.json` 的目录，或 `output_dir` 根目录（脚本会自动选最新 `checkpoint-*`）
- `prompt.user`：测试提示词

```bash
python infer_lora_qwen.py --config configs/infer/qwen2.5_7b_lora.yaml
```

命令行覆盖示例：

```bash
python infer_lora_qwen.py \
  --config configs/infer/qwen2.5_7b_lora.yaml \
  --adapter outputs/qwen2.5-7b-qlora-run/checkpoint-200 \
  --user "用一句话介绍大语言模型。"
```

仅测基座（不加载 LoRA）：

```bash
python infer_lora_qwen.py --config configs/infer/qwen2.5_7b_lora.yaml --no-adapter
```

## 配置目录

| 路径 | 用途 |
|------|------|
| `configs/train/` | 训练 YAML |
| `configs/infer/` | 推理 YAML |
| `configs/eval/` | 评估脚本配置（如 `outputs_dir`） |

## 项目结构

```text
QwenQLoRAExample/
  train_qlora_qwen.py      # QLoRA 训练入口
  infer_lora_qwen.py       # LoRA 推理入口
  configs/                 # 训练 / 推理 / 评估配置
  data/                    # JSONL 样例与自定义数据
  outputs/                 # 训练输出（运行后生成）
  requirements.txt
```

## 常见问题

**CUDA OOM**  
减小 `per_device_train_batch_size`、`max_seq_length`，或增大 `gradient_accumulation_steps`。

**本地路径找不到**  
确认路径使用 Linux 正斜杠，例如 `/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`，且目录下存在 `config.json` 或完整的 `snapshots/` 结构。

**推理找不到 adapter**  
将 `adapter` 指向具体 `checkpoint-*` 子目录，或保证该目录下存在 `adapter_config.json`。
