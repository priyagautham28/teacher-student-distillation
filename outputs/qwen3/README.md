# Qwen3-1.7B Student — QLoRA Distillation on GSM8K

This folder contains the Qwen3-1.7B student model training setup and artifacts for the teacher–student distillation project.

## Model

- **Base model:** `Qwen/Qwen3-1.7B`
- **Task:** GSM8K mathematical reasoning
- **Training method:** Supervised Fine-Tuning (SFT) with QLoRA
- **Quantization:** 4-bit NF4 with double quantization
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Target modules:** all linear layers
- **Seed:** 42

## Training Data

The student was trained on teacher-generated GSM8K reasoning data.

- **Training examples:** 1,922
- **Validation examples:** 485

The training script expects JSONL files in conversational format ending with an assistant response.

## Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max sequence length | 1536 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Quantization | 4-bit NF4, double quantization |
| Compute dtype | bfloat16 |

## Training Results

The final training run completed 3 epochs.

| Metric | Value |
|---|---:|
| Train loss | 0.1009 |
| Eval loss | 0.1364 |
| Eval mean token accuracy | 0.9584 |
| Train runtime | 5648.34 s |
| Eval runtime | 112.37 s |

## GSM8K Evaluation

The model was evaluated on the full GSM8K test set of 1,319 examples before and after SFT.

| Stage | Exact Match Accuracy |
|---|---:|
| Before SFT | 74.68% |
| After SFT | 79.61% |

**Absolute improvement:** +4.93 percentage points.

The after-SFT run used the trained QLoRA adapter and achieved a valid output-format rate of approximately 98.94%.

## Repository Structure

```text
student/qwen3/
├── qwen3_pipeline.png
├── train_qwen3_1_7b_qlora.py
├── requirements.txt
└── README.md

outputs/qwen3/
├── before_sft/
│   ├── *_metrics.json
│   ├── *_predictions.jsonl
│   └── *_summary.csv
├── after_sft/
│   ├── *_metrics.json
│   ├── *_predictions.jsonl
│   └── *_summary.csv
└── qwen3_1_7b_gsm8k_qlora_v4/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── all_results.json
    ├── eval_results.json
    ├── train_results.json
    ├── trainer_state.json
    └── training_manifest.json
```

## Installation

From the repository root:

```bash
pip install -r student/qwen3/requirements.txt
```

## Training

Run the QLoRA training script with the teacher-generated train and validation files:

```bash
python student/qwen3/train_qwen3_1_7b_qlora.py \
  --train-file <path-to-train.jsonl> \
  --val-file <path-to-val.jsonl> \
  --output-dir outputs/qwen3_1_7b_gsm8k_qlora_v4
```

The default training configuration uses:

- 3 epochs
- learning rate `2e-4`
- max sequence length `1536`
- batch size `1`
- gradient accumulation `8`

## Loading the Trained Adapter

The saved model is a PEFT/LoRA adapter on top of `Qwen/Qwen3-1.7B`.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen3-1.7B"
adapter_path = "outputs/qwen3/qwen3_1_7b_gsm8k_qlora_v4"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)
```

## Notes

- `adapter_model.safetensors` contains the learned LoRA adapter weights; it is not a full copy of the base Qwen3-1.7B model.
- The base model must still be available when loading the adapter.
- Detailed evaluation predictions and metrics are stored under `outputs/qwen3/before_sft/` and `outputs/qwen3/after_sft/`.
