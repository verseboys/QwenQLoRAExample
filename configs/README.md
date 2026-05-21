配置目录说明

| 子目录 | 用途 |
|--------|------|
| `train/` | 训练 YAML：`model` / `data` / `lora` / `bnb` / `training` |
| `infer/` | 推理 YAML：`model.base`、`adapter`、生成参数与提示词 |
| `eval/` | 评估 YAML：`outputs_dir` / `top` / `save` |

训练：`python train_qlora_qwen.py --config configs/train/qwen2.5_7b_instruct.yaml`  
推理：`python infer_lora_qwen.py --config configs/infer/qwen2.5_7b_lora.yaml`
