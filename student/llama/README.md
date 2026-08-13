# Llama-3.2-1B Reasoning Distillation (GSM8K)

Student track for distilling teacher GSM8K chain-of-thought into
**meta-llama/Llama-3.2-1B-Instruct** with QLoRA.

Shared across the team: teacher data, `evaluate.py` protocol, GSM8K test split.
Flexible per student: training script, hyperparameters, LoRA config.

---

## Final result (official)

| Setting | Value |
|--------|------:|
| Adapter | `outputs/llama3_1b_v4_promptmatch_r16_lr2e4/final_adapter` |
| Metrics | `eval_outputs/meta-llama_Llama-3.2-1B-Instruct_after_sft_10a84857_metrics.json` |
| **Exact-match accuracy** | **51.3%** (CI 48.5–54.0%) |
| Correct-and-valid | 51.3% |
| Valid format | 92.8% |
| Truncation | 7.1% |
| Eval | 0-shot, tagged CoT, greedy (`temperature=0`), `max_new_tokens=1024` |

---

## Pipeline overview

```text
Qwen3-14B-AWQ teacher  →  verified GSM8K SFT JSONL  →  Llama-3.2-1B QLoRA SFT  →  shared evaluate.py

## Step-by-step what we did

### 1. `train.py` (baseline script, not used for reported runs)

- First QLoRA SFT skeleton (MLflow, answer-only vs reasoning variants).
- Kept as a starting point; **no official adapter** from this file.

### 2. `train_v2.py` + teacher **v3** data

**Data:** `cot_faithfulness/data_final/`  
`teacher_gsm8k_*_gsm8k_teacher_v3_9cbf703286_full_sft.jsonl`

**Teacher / student style:** “concise” step-by-step CoT (2–8 steps encouraged).

**Training recipe (main changes vs bare SFT):**
- Loss only on assistant tokens (`DataCollatorForCompletionOnlyLM`)
- LoRA on attention + MLP (`q/k/v/o` + `gate/up/down_proj`)
- Cosine LR + warmup, gradient checkpointing, NEFTune
- Early stopping + `load_best_model_at_end` on **eval_loss**
- Small generation probe each epoch (logged to MLflow only; not used for selection)

**Hyperparameters used:**

| Param | Value |
|-------|------:|
| variant | reasoning |
| lr | 2e-4 |
| LoRA r / alpha | 16 / 32 |
| dropout | 0.05 |
| epochs | 3 |
| batch × grad_accum | 4 × 4 (eff. 16) |
| max_seq_length | 1024 |
| neftune_alpha | 5.0 |

**Output:** `llama-1b-reasoning/` (best checkpoint often `checkpoint-232`)

**Full greedy eval (old concise prompt):** ~**49.0%** EM  
(`…57b83258_checkpoint232_metrics.json`)

### 3. Teacher dataset revision → **v4**

**Why:** Concise CoTs were too compressed for hard problems; 1B student needed finer steps.

**Changes in `generate__teacher_gsm8k.py` (`gsm8k_teacher_v3` → `v4`):**
- Dropped “concise” / “2–8 steps” bias
- Added **at most one arithmetic operation per step**
- Require symbolic `+ - * / =` equations
- Write for a small student that slips on arithmetic
- New student system prompt (one-op-per-step)

**Data:** `cot_faithfulness/data_final_v2/`  
`teacher_gsm8k_*_gsm8k_teacher_v4_434a9551e7_full_sft.jsonl`  
(~1922 train / ~485 val accepted)

### 4. `train_v3.py` on v4

**New vs v2:**
- Checkpoint / early stop on **`eval_gen_exact_match`** (n=100), not eval_loss
- Generation callback registered before early stopping
- Score gen-eval against `gold_answer`
- Save `final_adapter` only to MLflow

**Hyperparameter runs on v4:**

| Run | lr | r / α | Notes | Train gen-EM peak | Full test EM |
|-----|---:|-------|-------|------------------|-------------:|
| v4 r16 | 2e-4 | 16 / 32 | prompts not fully matched to eval | 0.59 | **48.9%** |
| v4 r32 | 2e-4 | 32 / 64 | more capacity | 0.60 | **48.7%** |
| v4 r16 | 1e-4 | 16 / 32 | half-aligned prompts | 0.53 | (skipped full) |
| **v4 prompt-match r16** | **2e-4** | **16 / 32** | train = eval system + `Problem:\n{q}` | 0.57 | **51.3%** |

**Prompt-match fix (what moved 48.9% → 51.3%):**
- Stop using JSONL `messages` system/user as-is
- Rebuild every row with shared `evaluate.py` `SYSTEM_PROMPT` + `USER_TEMPLATE = "Problem:\n{question}"`
- Same prompts in train-time gen-eval

**Best official adapter:**  
`outputs/llama3_1b_v4_promptmatch_r16_lr2e4/final_adapter`

---

## Training version comparison

| Version | Data | Eval prompt | Test EM | Format | Trunc |
|---------|------|-------------|--------:|-------:|------:|
| train_v2 | v3 concise | concise | 49.0% | 98.5% | 1.5% |
| train_v3 r16 | v4 | new (mismatched train) | 48.9% | 91.3% | 8.7% |
| train_v3 r32 | v4 | new | 48.7% | 90.0% | 9.9% |
| **train_v3 prompt-match r16** | **v4** | **new (matched)** | **51.3%** | **92.8%** | **7.1%** |

Failure mix (best run): mostly **wrong_answer with valid tags** (~642); ~94 loop/trunc format failures.

---

## Comparison to Meta Llama-3.2-1B-Instruct

| | Meta (model card) | Ours (best) |
|--|-------------------|-------------|
| GSM8K score | **44.4%** | **51.3%** |
| Training on GSM8K | No (general chat model; GSM8K is a benchmark) | **Yes** — SFT on teacher GSM8K CoTs |
| Shots | **8-shot** CoT | **0-shot** |
| Prompt | Meta CoT few-shot | Tagged `<reasoning>` / `<final_answer>` |
| Metric | `em_maj1@1` (1 sample, exact match) | Same idea: 1 greedy decode, exact match |

### Why ours can exceed Meta’s number (fair wording)

1. **Different setup** — not a strict +7pp on Meta’s metric; say “higher under our protocol / competitive with Meta’s 1B GSM8K band.”
2. **We fine-tuned on GSM8K** — Meta’s 44.4% is the **base Instruct** model **tested** on GSM8K, not SFT’d on it.
3. **Task-specific distillation** — verified teacher traces teach the required format and solution style for this benchmark.
4. **Prompt alignment** — matching train to eval removed a distribution shift and gained ~2 pp over our own mismatched v4 run.

**Do not claim:** “We beat Meta’s training.”  
**Do claim:** “After GSM8K CoT distillation with eval-matched prompts, our 1B student reaches 51.3% under the team’s 0-shot tagged eval; Meta reports 44.4% for the untuned Instruct model under 8-shot CoT.”

---

## How to reproduce

```bash
cd ~/reasoning-distillation && source .venv/bin/activate

# Train (prompt-matched train_v3)
python train_v3.py --variant reasoning \
  --train_file ../cot_faithfulness/data_final_v2/teacher_gsm8k_train_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full_sft.jsonl \
  --val_file ../cot_faithfulness/data_final_v2/teacher_gsm8k_val_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full_sft.jsonl \
  --lr 2e-4 --r 16 --alpha 32 \
  --gen_eval_max_new_tokens 1024 \
  --output_dir outputs/llama3_1b_v4_promptmatch_r16_lr2e4

# Eval (shared protocol)
python evaluate.py \
  --backend transformers \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --adapter-path outputs/llama3_1b_v4_promptmatch_r16_lr2e4/final_adapter \
  --stage after_sft \
  --max-new-tokens 1024

# Curves
python plot_v3.py