# Knowledge Distillation for Efficient Mathematical Reasoning in Compact Language Models

*(Repo: teacher-student-distillation)*

**Team:** Pick and Parse
**Members:** Priyadarshini Rajmohan · Poojitha Alam · Mounika Akkenapragada

This repository is the shared team repo for the full distillation pipeline: one teacher model and three independent student tracks. **This README covers the shared/root-level pieces and focuses mainly on the teacher model**, since that's owned here. Each student track will document setup and training in its own subfolder README when available.

## Team structure

### Contributions on this component
| Piece | Who |
|---|---|
| Generation pipeline code (`generate_teacher_gsm8k.py`) | Poojitha Alam (majority) & Priyadarshini Rajmohan |
| Train/val dataset generation — actually run | Priyadarshini Rajmohan |
| Shared evaluator (`evaluation/evaluate_gsm8k.py`) | Poojitha Alam & Mounika Akkenapragada |
| Official teacher test-set evaluation (Qwen3-14B-AWQ's accuracy on the full 1,319-question test set) | Initial run: Mounika Akkenapragada. Final official run (after the `gsm8k_teacher_v4` prompt update): Priyadarshini Rajmohan |

| Role | Model | Owner | Details |
|------|--------|--------|---------|
| Teacher + dataset | Qwen3-14B-AWQ | Priyadarshini Rajmohan | See contribution breakdown above. |
| Student — `student/llama/` | Llama-3.2-1B-Instruct | Priyadarshini Rajmohan | `student/llama/` (track README TBD) |
| Student — `student/gemma/` | Gemma 3 1B | Poojitha Alam | `student/gemma/` (track README TBD) |
| Student — `student/qwen/` | Qwen3-1.7B | Mounika Akkenapragada | `student/qwen/` (track README TBD) |

Shared responsibilities: teacher prompts, dataset quality, hyperparameter protocol, audit of results, final report/presentation.

## Pipeline overview

![Project pipeline](teacher/project_pipeline.png)

*Note: the diagram shows an earlier planning snapshot (2,000 train + 200 val). The current `gsm8k_teacher_v4` run samples **2,000 train + 500 val** and keeps **1,922 / 485** accepted SFT examples after validation — see [Data](#data).*

## Repository structure (current)

```text
teacher-student-distillation/
├── README.md                     # this file — shared overview, mainly teacher model
├── .gitignore
├── requirements-teacher.txt      # teacher / vLLM stack
├── requirements-llama.txt        # this track's student deps
├── audits/                       # teacher generation accepted/rejected audit records
├── data/                         # shared train/val SFT JSONL used by all student tracks
├── evaluation/
│   ├── evaluate_gsm8k.py         # shared teacher + student evaluator
│   └── poster_analysis.py        # McNemar + team summary merge
├── outputs/                      # run manifests, metrics, predictions (adapters may be gitignored)
├── teacher/
│   ├── generate_teacher_gsm8k.py # teacher dataset generation + validation pipeline
│   └── project_pipeline.png      # high-level pipeline diagram
└── student/
    ├── llama/                    # Priyadarshini — track README
    ├── gemma/                    # Poojitha — track README
    └── qwen/                     # Mounika — track README
```

Each student subfolder owner will maintain their own README with that track's environment setup, training commands, and hyperparameter choices — this file doesn't duplicate that detail.


## Research question

How effectively can knowledge distillation transfer mathematical reasoning capability from a large language model to compact language models while maintaining computational efficiency?

- **Minimal goal:** Generate a teacher dataset from a GSM8K subset; fine-tune three compact students (Qwen3-1.7B, Gemma 3 1B, Llama 3.2 1B) with QLoRA; evaluate each against its own pretrained base on the official GSM8K test split.
- **Ambitious goal:** Compare student architectures under identical training conditions; measure efficiency gains from distillation; analyze how much of the teacher's performance each student retains; investigate whether architecture choice affects distillation effectiveness.
- **Success criterion:** Reproducible adapters + metrics under a fixed protocol so the three student tracks are fairly comparable. A null or small gain is a valid scientific result.

## Executive Summary

This project investigates knowledge distillation techniques to transfer complex mathematical reasoning capabilities from large teacher language models to smaller, compact student models. By distilling reasoning pathways and intermediate rationales, we aim to build resource-efficient models capable of solving multi-step mathematical problems locally without incurring the high latency and computational overhead of large-scale models.

## Key Findings & Results

**Teacher (shared reference).** Qwen3-14B-AWQ reaches **92.27%** exact-match on the official GSM8K test set under the shared tagged-CoT protocol — the ceiling every student is measured against.

**Llama-3.2-1B-Instruct (Priyadarshini track) — filled.**
- Base (before SFT, bf16): **43.4%** EM; valid format only **~50%**; correct-and-valid **~25%**.
- Best after QLoRA (prompt-matched `train_v3` on `gsm8k_teacher_v4`): **51.3%** EM (95% CI 48.5–54.0%); valid format **92.8%**; correct-and-valid **51.3%**.
- **Distillation gain:** **+7.9 pp** absolute EM over the same student’s base under the shared evaluator.
- **Gap to teacher:** **41.0 pp** remaining (92.27% − 51.3%) — most headroom is still open.
- **What improved most:** format adherence (tagged `<reasoning>` / `<final_answer>`). Residual failures are mostly **valid-but-wrong math** (~642), not tag errors (~94 trunc/loop).
- **What did *not* move the needle much:** longer one-op-per-step teacher traces alone (v4 without prompt match ≈ **48.9%**, similar to v3 concise **~49.0%**); LoRA **r=32** slightly worse than **r=16** (**48.7%**).
- **Largest in-protocol win for Llama:** aligning train prompts to `evaluate_gsm8k.py` (`SYSTEM_PROMPT` + `Problem:\n{question}`) — **48.9% → 51.3%**.

**Vs Meta’s published Llama-3.2-1B-Instruct GSM8K (44.4%, 8-shot CoT).** Our **51.3%** is higher but **not a strict apples-to-apples beat**: Meta’s number is the untuned Instruct model under 8-shot CoT; ours is **0-shot tagged CoT after GSM8K teacher distillation**. Fair claim: distillation + eval-matched prompts lifts this 1B student above both its own base and Meta’s reported 1B band under *our* protocol — not that we beat Meta’s training recipe.

**Gemma / Qwen student tracks:** TBD (fill when teammates finish).

**Takeaway so far:** distillation clearly teaches the *protocol* (format + answer extraction) and recovers a meaningful but bounded accuracy gain on a ~1B Llama; architecture/scale still dominate the remaining gap to the 14B teacher.


## Architectural differences across student models, and our expectations

The three students are meaningfully different in architecture and training history, not just parameter count:

| | Qwen3-1.7B | Gemma 3 1B | Llama 3.2 1B |
|---|---|---|---|
| Layers | 28 | 26 | 16 |
| Hidden dim | 2048 | 1152 | 2048 |
| Attention | GQA, 16 query heads / 8 key-value heads | Interleaved: 5 local sliding-window layers (1024-token window) per 1 global layer | Standard GQA every layer, 8 KV heads |
| Vocab / tokenizer | BBPE, 151,669 tokens, 119 languages | Gemini 2.0 SentencePiece, 262,144 tokens | BPE, 128,256 tokens |
| Native context | 32,768 tokens | 32,768 tokens (1B variant) | 131,072 tokens |
| Pretraining scale | ~36 trillion tokens | ~2 trillion tokens (1B variant) | Derived from Llama 3.1's pretraining, not trained fresh at comparable scale |
| How it was actually built | Standard large-scale dense pretraining, with an explicit built-in "thinking / non-thinking" reasoning mode | Pretrained, then post-trained with knowledge distillation from a larger instruct model | Built by *pruning* Llama 3.1 8B, then using knowledge distillation (logits from the 8B and 70B models as token-level targets) to recover performance lost during pruning |

**The key observation this surfaces:** two of our three students (Llama 3.2 1B and Gemma 3 1B) are themselves already distilled models, produced from a larger teacher before our project even begins. Qwen3-1.7B, by contrast, is a standard densely-pretrained model with no such distillation lineage, and is also the largest of the three students.

**Our ranked expectation:**

1. **Qwen3-1.7B is expected to reach the highest absolute accuracy after distillation** — it has the most parameters, by far the largest pretraining scale, and a reasoning-oriented ("thinking mode") architecture already aligned with the kind of step-by-step supervision our teacher dataset provides.
2. **Llama 3.2 1B may show the largest *relative* improvement from our distillation step**, even if it doesn't win on final accuracy, its own training history already depends on learning from a larger teacher's logits, which may make it comparatively receptive to a second round of teacher-based fine-tuning.
3. **Gemma 3 1B has documented, targeted training for math reasoning, which sets it apart from the other two students.** Its post-training pipeline explicitly includes reinforcement learning with ground-truth rewards for solving math problems, a dedicated math-reasoning training step neither Qwen3-1.7B's general pretraining nor Llama 3.2 1B's distillation lineage specifically includes. Given its targeted reasoning training, we expect Gemma to outperform Llama at baseline and likely retain some of that edge after distillation.


## The teacher model — Qwen3-14B-AWQ

This is the core piece owned in this repo's root, since every student track depends on it.

**Why Qwen3-14B-AWQ specifically:** the team initially discussed Qwen3-32B or DeepSeek-R1 as the teacher, but both were ruled out on hardware grounds — Qwen3-32B needs ~64GB VRAM at full precision (not possible on a single 24GB GPU), its FP8 form is unreliable on Ampere-generation cards, and AWQ with tensor parallelism would need two confirmed 24GB GPUs, which wasn't available. Qwen3-14B-AWQ was already smoke-tested and running locally via vLLM, making it the lower-risk, immediately workable choice while still being a meaningfully stronger reasoner than any of the ~1-2B students.

**What the teacher pipeline (`teacher/generate_teacher_gsm8k.py`) does:**
- Samples a fixed, reproducible subset of GSM8K (**2,000 train + 500 validation** examples), with the split cached and fingerprinted against the source dataset so it can't silently drift across reruns.
- Prompts the teacher with **`gsm8k_teacher_v4`**: tagged `<reasoning>...</reasoning><final_answer>...</final_answer>` output, with at most one arithmetic operation per step and symbolic equations (digits/`+ - * / =`) so a small student can follow each step.
- Validates every generation against a strict quality bar: correct final answer, well-formed tags, a minimum/maximum reasoning length, genuine calculation content (not just a restated total), no excessive repetition, no significant text outside the required tags.
- Retries failed generations up to a fixed attempt limit, with deterministic per-attempt seeding so any regeneration is reproducible.
- Logs every attempt to an append-only event log (`audits/`), so a killed run can resume exactly where it left off without losing or duplicating work, and rejected examples remain available for audit or later recovery.
- Produces a clean, minimal SFT-ready JSONL per split, used identically by all three student tracks.

**Known edge case already handled:** a small number of otherwise-correct generations were being rejected because dollar-sign formatting in a calculation (e.g. `12 * $0.50 = $6.00`) broke the strict calculation-detail regex check. `teacher/rescur_dollar_rejects.py` recovers these specific false-rejects under a deliberately looser (but still validated) pattern, with a full backup taken before any file is modified.

**Teacher evaluation:** run via `evaluation/evaluate_gsm8k.py` against the official, untouched GSM8K test split, exactly like every student. Mounika ran the initial official evaluation; after the teacher prompt was updated to `gsm8k_teacher_v4` (one arithmetic operation per step), Priyadarshini reran the official test-set evaluation to confirm the teacher's ceiling under the current protocol. Current official test exact-match for Qwen3-14B-AWQ: **92.27%** (95% CI 90.9–93.7%; metrics file: `outputs/teacher_testset/Qwen_Qwen3-14B-AWQ_teacher_3cb9a5c9_metrics.json`).

## Data

Primary source: **GSM8K** (grade-school math word problems).

Current shared dataset: prompt version **`gsm8k_teacher_v4`**, config hash `434a9551e7` (run slug `qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full`).

| Split | Sampled | Accepted (SFT) | Acceptance rate |
|-------|---------|----------------|-----------------|
| Train | 2,000 | 1,922 | 96.1% |
| Validation | 500 | 485 | 97.0% |
| Test | — | official GSM8K test (1,319) — **untouched** throughout training | — |

Shared files consumed by every student track:
- `data/teacher_gsm8k_train_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full_sft.jsonl`
- `data/teacher_gsm8k_val_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full_sft.jsonl`

Audits / run metadata:
- `audits/teacher_gsm8k_{train,val}_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full_*.jsonl`
- `outputs/teacher_train_metrics/run_manifest_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full.json`
- `outputs/teacher_train_metrics/run_metrics_qwen3_14b_awq_gsm8k_teacher_v4_434a9551e7_full.json`
- `outputs/teacher_train_metrics/gsm8k_subset_indices_bf6906f85e.json`

## Evaluation

Shared across the teacher and all three students — one evaluator, not duplicated per model, so comparisons stay fair:

- `evaluation/evaluate_gsm8k.py` — model-family-agnostic evaluator (works with Qwen, Llama, or Gemma via `--model`). Supports the teacher (`--backend openai`, pointed at a running vLLM server) and any student (`--backend transformers`, optionally with `--adapter-path` and `--compare-base` for an automatic before/after comparison in one run).
- `evaluation/poster_analysis.py` — run once predictions exist: McNemar's significance test (was an accuracy difference real, or could it be noise?) and a merge utility to combine all three teammates' `summary.csv` into one team-wide comparison table.

### Metrics reported

| Metric | Purpose |
|--------|---------|
| GSM8K exact-match accuracy | Primary quality metric |
| Improvement over base model | Distillation gain |
| Gap to teacher performance | Remaining headroom |
| Training time | Cost of fine-tuning per student |
| Peak GPU memory usage | Local deployment feasibility |
| Inference latency | Practical speed, per example |
| Generation throughput (tokens/sec) | Practical speed, aggregate |
| Model size | Deployment footprint |
| Trainable parameter count | QLoRA efficiency |
| Output-format success rate | How often a scoreable final answer could be extracted at all |

All three student tracks use the same teacher-generated dataset, identical train/val/test splits, identical prompts and decoding parameters, and the same evaluation script and scoring methodology.

### Compute / GPU usage

Teacher generation (train + val) and teacher test eval were run with different batching settings, so peak VRAM is reported separately. Generation config `max_tokens` is **2048** (see run manifest); train/val peak ~23 GB was observed at **batch size 4** on the vLLM teacher server.

| Model | Stage | Batch size | Max tokens | Peak GPU memory |
|-------|--------|------------|------------|-----------------|
| Teacher (Qwen3-14B-AWQ) | Train + val generation | 4 | 2048 | ~23 GB |
| Teacher (Qwen3-14B-AWQ) | Official test eval (Mounika) | 1 | 2048 | ~9.7 GB |
| Llama-3.2-1B (QLoRA train) | Fine-tuning (`train_v3`, r=16, lr=2e-4, eff. batch 16, max_seq 1024) | 4 × accum 4 | 1024 (seq) / gen-eval 1024 | ~4-bit QLoRA on 24GB class GPU; wall-clock ~30 min / run |
| Llama-3.2-1B (base eval) | Official test (before SFT, bf16) | 1 | 768 | ~2.3 GB |
| Llama-3.2-1B (after QLoRA eval) | Official test (best adapter) | 1 | 1024 | ~2.6 GB |
| Gemma-3-1B (QLoRA train) | Fine-tuning | TBD | TBD | TBD |
| Gemma-3-1B (eval) | Test | TBD | TBD | TBD |
| Qwen3-1.7B (QLoRA train) | Fine-tuning | TBD | TBD | TBD |
| Qwen3-1.7B (eval) | Test | TBD | TBD | TBD |

## Results

Teacher official test and Llama base (before SFT) are filled from existing metrics; remaining student cells stay TBD until each track finishes.

| Model | Exact-match accuracy | Improvement over base | Gap to teacher | Peak GPU memory | Inference latency | Model size |
|---|---|---|---|---|---|---|
| Teacher (Qwen3-14B-AWQ) | 92.27% | — | — | ~9.7 GB (test) / ~23 GB (gen) | ~2.27 s / ex (test) | 14B AWQ |
| Llama-3.2-1B (base) | 43.4% | — | 48.9 pp | ~2.3 GB | ~1.9 s / ex | ~1.2B |
| Llama-3.2-1B (after QLoRA) | **51.3%** | **+7.9 pp** | **41.0 pp** | ~2.6 GB | ~5.6 s / ex | ~1.2B + ~58 MB adapter |
| Gemma-3-1B (base) | TBD | — | TBD | TBD | TBD | TBD |
| Gemma-3-1B (after QLoRA) | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-1.7B (base) | TBD | — | TBD | TBD | TBD | TBD |
| Qwen3-1.7B (after QLoRA) | TBD | TBD | TBD | TBD | TBD | TBD |

Teacher metrics: `outputs/teacher_testset/Qwen_Qwen3-14B-AWQ_teacher_3cb9a5c9_metrics.json`. Llama base: `outputs/llama/before_sft/meta-llama_Llama-3.2-1B-Instruct_before_sft_5b7bd0c3_before_sft_bf16_metrics.json`.

### Llama before vs after SFT (detail)
| Metric | Before SFT | After SFT (best) | Change |
|--------|----------:|-----------------:|--------|
| Exact-match accuracy | 43.4% | **51.3%** | **+7.9 pp** |
| Correct-and-valid rate | 25.2% | **51.3%** | **+26.1 pp** |
| Valid format rate | 50.4% | **92.8%** | **+42.4 pp** |
| Truncation rate | 0.8% | 7.1% | +6.3 pp (longer CoTs) |
| Avg inference latency | ~1.9 s / ex | ~5.6 s / ex | longer generations |
| Peak GPU memory (eval) | ~2.3 GB | ~2.6 GB | similar |
| `max_new_tokens` | 768 | 1024 | — |
| Adapter | none | `…/llama3_1b_v4_promptmatch_r16_lr2e4/final_adapter` | QLoRA r=16, lr=2e-4 |
**What changed:** distillation mainly taught the required tagged format (format ~50% → ~93%) and raised answer accuracy by **+7.9 pp**. Remaining errors after SFT are mostly **wrong math with valid tags**, not missing tags.
**Metrics files:**
- Before: 
- After: 

## Proof: Llama-3.2-1B vs Meta’s model-card GSM8K (what actually exceeded 44.4%)

### Fair comparison first

| | Meta Llama-3.2-1B-Instruct (model card) | Our best Llama student |
|--|----------------------------------------|-------------------------|
| GSM8K score | **44.4%** | **51.3%** |
| Training on GSM8K | No (base Instruct; GSM8K is a benchmark) | **Yes** — QLoRA SFT on verified teacher CoTs |
| Shots | **8-shot** CoT | **0-shot** |
| Prompt | Meta few-shot CoT | Tagged `<reasoning>` / `<final_answer>` (shared team prompt) |
| Decoding | `em_maj1@1` (1 sample) | Greedy (`temperature=0`), 1 decode |
| Metric idea | Exact match on final answer | Exact match on final answer |

**Claim we make:** under our shared 0-shot tagged protocol, after distillation, Llama reaches **51.3%**, which is **above Meta’s published 44.4%** and **+7.9 pp** over our own base (**43.4%**).  
**Claim we do *not* make:** that we beat Meta’s training recipe under Meta’s 8-shot setup.

---

### Ablation evidence — which methods moved the score

Same student (`meta-llama/Llama-3.2-1B-Instruct`), same shared GSM8K test (1,319), greedy eval unless noted.

| Step | Method / change | Data | Key params | Test EM | vs Meta 44.4% |
|------|-----------------|------|------------|--------:|:-------------:|
| A | Base Instruct (no SFT) | — | bf16, `max_new_tokens=768` | **43.4%** | below |
| B | QLoRA SFT (`train_v2`) | teacher **v3** (concise CoT) | lr `2e-4`, LoRA r/α `16/32`, select on **eval_loss** | **~49.0%** | above |
| C | QLoRA SFT (`train_v3`) | teacher **v4** (one-op-per-step) | lr `2e-4`, r/α `16/32`, select on **`eval_gen_exact_match`** | **48.9%** | above |
| D | Same as C + larger LoRA | v4 | lr `2e-4`, r/α **`32/64`** | **48.7%** | above |
| E | Same as C + lower LR (partial prompt align) | v4 | lr **`1e-4`**, r/α `16/32` | train gen-EM peak 0.53 (full test skipped) | — |
| **F (best)** | **C + train↔eval prompt match** | **v4** | **lr `2e-4`, r/α `16/32`, rebuild messages to match `evaluate.py`** | **51.3%** | **above (+6.9 pp vs Meta card)** |

**What the table proves**
1. **SFT alone** (B/C) already clears Meta’s **44.4%** (~49%), mainly by teaching the tagged format + GSM8K-style solutions.
2. **v4 data alone** did **not** beat v3 (~48.9% ≈ 49.0%); longer CoTs without prompt match mainly hurt format/truncation.
3. **More LoRA rank (r=32)** did **not** help (D ≤ C).
4. The **extra lift to 51.3%** is from **prompt alignment** (F): train system + user templates identical to shared eval (`SYSTEM_PROMPT` + `Problem:\n{question}`), not from a bigger adapter.

---

### Exact recipe of the run that scored 51.3%

| Knob | Value used |
|------|------------|
| Script | `train_v3.py` |
| Base model | `meta-llama/Llama-3.2-1B-Instruct` |
| Method | QLoRA SFT (assistant-token loss only) |
| Teacher data | `gsm8k_teacher_v4` / hash `434a9551e7` (~1922 train / 485 val) |
| Variant | `reasoning` |
| Learning rate | `2e-4` (cosine + warmup) |
| LoRA r / α / dropout | `16` / `32` / `0.05` |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Batch × grad accum | `4 × 4` (effective 16) |
| `max_seq_length` | `1024` |
| NEFTune α | `5.0` |
| Checkpoint selection | `eval_gen_exact_match` (n=100), early stopping |
| Prompt match | Rebuild every SFT row with shared eval `SYSTEM_PROMPT` + `USER_TEMPLATE = "Problem:\n{question}"` |
| Adapter path | `outputs/llama3_1b_v4_promptmatch_r16_lr2e4/final_adapter` |
| Eval | shared `evaluate.py` / `evaluate_gsm8k.py`, `--stage after_sft`, greedy, `max_new_tokens=1024` |
| Metrics file | `…_after_sft_10a84857_metrics.json` |

**Best-run secondary metrics (proof it’s not just format hacking):**
- Exact-match: **51.3%** (CI 48.5–54.0%)
- Correct-and-valid: **51.3%**
- Valid format: **92.8%**
- Truncation: **7.1%**
- Dominant failure: **wrong_answer with valid tags** (~642) — reasoning errors, not missing tags

*Add once available:*
- McNemar significance results for each student's before-vs-after comparison
- McNemar significance results for each student's after-vs-teacher gap
- Cross-model comparison chart (accuracy, latency, memory) across all three students
- A few concrete example outputs (teacher trace vs. student trace) for the report/poster
- Whether the ranked expectation above (Qwen3-1.7B highest accuracy, Llama largest relative gain, Gemma as the open question) held up against the actual results

## Conclusion

*Placeholder — write once Results above is filled in. A few prompts to structure it:*

- **Headline finding:** in one or two sentences, did distillation meaningfully recover reasoning performance across the students, and how consistent was that across architectures?
- **Does distillation effectiveness differ by architecture?** Revisit the ranked expectation above directly — which prediction held, which didn't, and why?
- **Efficiency trade-off:** how do the accuracy gains weigh against the peak memory / latency / model size numbers — is the smallest student "good enough" for the privacy-preserving local-deployment motivation, or does it fall short in practice?
- **Limitations:** training subset size (2,000 examples), reliance on a single teacher model's generations as ground truth (errors in the teacher dataset propagate to all students), and any student where distillation gains were small or null.
- **Why this matters going forward:** tie back to the motivation — privacy-preserving assistants, edge deployment, offline educational tools — what does the result actually tell someone deciding whether a distilled compact model is viable for their use case?

## Lessons learned

1. **Pilot small, then scale.** Next time: smoke the full train→gen-eval→full-test loop on **~500** accepted teacher examples (or a hard subset), lock prompts/hparams, **then** scale to the full 2,000-train run. We paid full-data cost several times for issues that a 500-ex pilot would have caught earlier (prompt mismatch, truncation under longer CoTs, r=32 not helping).

2. **Train must match eval prompts exactly.** Teacher JSONL `messages` are not automatically the eval prompt. Rebuilding every SFT row with the shared `evaluate_gsm8k.py` system prompt + `Problem:\n{question}` was worth ~**+2 pp** on Llama (48.9% → 51.3%). Treat prompt drift as a first-class bug.

3. **Teacher style trades accuracy for format risk on tiny students.** One-op-per-step v4 traces are clearer but longer; without enough `max_new_tokens` / loop control, 1B models truncate and lose `</final_answer>`. Measure format/truncation alongside EM.

4. **More LoRA rank ≠ better.** On this task, **r=32** did not beat **r=16** (48.7% vs 48.9% before prompt-match). Prefer a small controlled grid over assuming capacity helps.

5. **Select checkpoints on generation EM, not only eval_loss.** `train_v3` early-stops on `eval_gen_exact_match` (n=100). Loss can improve while greedy answers do not.

6. **Error mix tells you what to optimize next.** After prompt-match, Llama’s residual is mostly **wrong math with valid tags** — a capacity/reasoning ceiling — not more tag engineering. Further gains likely need better data mixture, rejection sampling, or a stronger student, not another format tweak.

7. **Save adapters + config hashes + metrics together.** Resume-friendly eval and config hashes (`10a84857`, etc.) made full 1,319-run debugging tractable, keep that discipline for the team merge.

## Risks and mitigations (summary — see full proposal for details)

- **Teacher prompt/data iteration cost:** revising the teacher prompt for hard problems required a full dataset regeneration and retraining cycle before the real issues surfaced. **Mitigation:** pilot new prompts on a small subset (~500 examples) through the full loop before scaling to all 2,000.
- **Limited training data:** fixed random seed for reproducibility; subset size may be expanded if time/compute allow.
- **Architecture differences across students:** shared dataset/splits/protocol across all three; only QLoRA target modules are adjusted per architecture where necessary.
- **GPU/compute constraints:** 4-bit QLoRA with gradient accumulation and checkpointing across all students.
- **Uneven distillation effectiveness across architectures:** evaluated under identical conditions regardless of outcome; a null or uneven result is reported as a valid finding, not treated as project failure.

## Privacy perspective

Privacy is the **application motivation**, not a new privacy algorithm. A compact local student keeps prompts on user-controlled hardware. The project discusses that advantage in the report; it does not claim a novel privacy method.

## Ethical considerations

- Uses public benchmarks (GSM8K) and openly published model weights under their respective licenses.
- No human subjects or private user data in the core experiments.
- Llama access requires accepting Meta's community license and acceptable-use policy.
- Teacher-generated demonstrations are verified against GSM8K ground truth before being used for training.
- Results are reported honestly, including negative or small distillation gains.

## Licenses & attribution

- **Qwen3-14B-AWQ / Qwen3-1.7B:** Qwen license / model card on Hugging Face
- **Llama 3.2:** Meta Llama 3.2 Community License — [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Gemma 3:** Gemma license / model card on Hugging Face
- **GSM8K:** OpenAI GSM8K dataset (see Hugging Face `openai/gsm8k`)
- **Libraries:** PyTorch, Hugging Face Transformers / PEFT / TRL / datasets, bitsandbytes, MLflow, vLLM (teacher) — under their respective open-source licenses

## References

1. Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K). https://arxiv.org/abs/2110.14168
2. Qwen Team (2025). *Qwen3 Technical Report.* Alibaba Group.
3. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* https://arxiv.org/abs/2106.09685
4. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. https://arxiv.org/abs/2305.14314
5. Hinton, Vinyals, & Dean (2015). *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop.
6. Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020.
7. Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).* JMLR.
