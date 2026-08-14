# Qwen3-1.7B Knowledge Distillation on GSM8K

**Teacher:** Qwen3-14B-AWQ
**Student:** Qwen3-1.7B
**Task:** GSM8K mathematical reasoning
**Method:** Offline sequence-level knowledge distillation using automatically validated teacher-generated reasoning traces and 4-bit QLoRA SFT

## Key Result

Knowledge distillation improved **Qwen3-1.7B** from **74.68% to 79.61% exact-match accuracy** on the untouched 1,319-question official GSM8K test split.

- **+4.93 percentage points** absolute accuracy
- **985 -> 1,050** correct answers (**+65 net**)
- **334 -> 269** errors (**19.5% error reduction**)
- **157 fixes vs. 92 regressions** in paired question-level analysis
- **6.22% -> 98.94%** valid structured outputs
- Teacher reference: **92.27%** exact match
- Approximately **28% of the original teacher-student accuracy gap** was closed

---

## What We Did

We selected **2,000 training** and **500 validation** candidate questions from the GSM8K training split and used **Qwen3-14B-AWQ** to generate worked mathematical solutions.

The teacher prompt was designed for a smaller student model. It requested:

- logically complete step-by-step reasoning;
- explicit numerical calculations;
- at most one arithmetic operation per step;
- all necessary intermediate calculations; and
- a normalized numerical final answer.

Teacher-generated solutions were **automatically validated before SFT**. Each output was checked against the GSM8K ground-truth final answer and screened for required reasoning detail, tag/format structure, truncation, repetition, and related quality failures.

| Split | Candidates | Accepted | Rejected | Acceptance rate |
|---|---:|---:|---:|---:|
| Train | 2,000 | **1,922** | 78 | **96.1%** |
| Validation | 500 | **485** | 15 | **97.0%** |

The accepted teacher traces were then used as fixed supervised targets for **Qwen3-1.7B**.

---

## Distillation Pipeline

```text
GSM8K training split
        |
        v
2,000 train + 500 validation candidates
        |
        v
Qwen3-14B-AWQ teacher
        |
        v
Granular worked solutions
        |
        v
Automatic validation & filtering
  - ground-truth final-answer match
  - reasoning / format checks
  - truncation / repetition checks
        |
        v
1,922 train + 485 validation demonstrations
        |
        v
Qwen3-1.7B + 4-bit QLoRA SFT
        |
        v
Untouched GSM8K test split
1,319 examples
        |
        v
Before-SFT vs. After-SFT evaluation
```

---

## Student Training Setup

| Training setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B` |
| Training objective | SFT on teacher-generated reasoning traces |
| Accepted train / validation | 1,922 / 485 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Maximum sequence length | 1,536 tokens |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Quantization | 4-bit NF4, double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | all-linear |

### Why QLoRA?

QLoRA was used to fine-tune Qwen3-1.7B efficiently by keeping the base model quantized to **4-bit** and training only small low-rank adapter weights. This reduces GPU-memory and compute requirements while preserving the pretrained model, making it a practical choice for testing knowledge distillation without full-parameter fine-tuning.

### Final Training Summary

| Metric | Result |
|---|---:|
| Training loss | **0.1009** |
| Validation loss | **0.1364** |
| Validation token accuracy | **95.84%** |
| Training time | **~94 min** |

Validation token accuracy measures how well the model fits the teacher-generated target sequences. It should not be interpreted as GSM8K reasoning accuracy. The main evidence of task improvement is the separate held-out GSM8K evaluation.

---

## Evaluation Methodology

The base and distilled Qwen3-1.7B models were evaluated on the same untouched **1,319-example official GSM8K test split**.

The before- and after-SFT student runs used:

- the same fixed dataset revision;
- the same evaluation prompt;
- seed 42;
- temperature 0;
- top-p 1;
- Qwen thinking mode disabled;
- maximum input length of 1,536 tokens;
- maximum generation length of 768 tokens; and
- identical answer extraction, numerical normalization, and scoring.

The primary metric is **GSM8K exact-match numerical accuracy**.

---

## Primary Results

| Condition | Correct | Incorrect | Exact match | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Qwen3-1.7B before SFT | 985 / 1,319 | 334 | **74.68%** | 72.40% - 77.10% |
| Qwen3-1.7B after SFT | 1,050 / 1,319 | 269 | **79.61%** | 77.33% - 81.65% |
| Qwen3-14B-AWQ teacher | 1,217 / 1,319 | 102 | **92.27%** | 90.90% - 93.71% |

The distilled student gained **+4.93 percentage points** of exact-match accuracy.

This corresponds to:

- **6.6% relative accuracy improvement**
- **65 additional correct answers net**
- **19.5% fewer errors**

---

## Question-by-Question Analysis

Aggregate accuracy alone does not show whether SFT genuinely corrected errors or simply changed which questions were missed. Because the same 1,319 test questions were evaluated before and after SFT, every example can be tracked as a matched outcome.

| Transition | Problems |
|---|---:|
| Correct before -> Correct after | **893** |
| Wrong before -> Wrong after | **177** |
| **Wrong before -> Correct after** | **157** |
| Correct before -> Wrong after | **92** |

Among the **249 questions whose correctness changed** after SFT:

- **157** changed from wrong to correct;
- **92** changed from correct to wrong;
- **63.1%** of changed outcomes moved in the favorable direction;
- the model fixed about **1.71 problems for every regression**.

The net effect is:

**157 fixes - 92 regressions = +65 correct answers**

This exactly matches the change from **985 to 1,050** correct answers.

An exact paired McNemar/binomial test on the 249 discordant pairs gives:

**p = 4.56e-05**

This provides strong evidence that the favorable imbalance between fixes and regressions is unlikely to be explained by random answer switching alone.

---

## What Changed After Distillation?

### Mathematical task performance improved

Exact-match accuracy increased from **74.68% to 79.61%**. Because exact match is based on the extracted numerical answer rather than perfect output formatting, the improvement cannot be explained simply by learning the requested response template.

### Generated solutions became more consistent with the granular training style

The teacher demonstrations used small, explicit arithmetic transitions. After SFT, average completion length increased from **171.5 to 202.3 tokens**, an increase of about **18%**, consistent with more explicit generated solution traces.

### Structured-output behavior transferred strongly

Valid-format generation increased from:

**6.22% -> 98.94%**

After SFT, Qwen3-1.7B became much more reliable at producing the requested `<reasoning>` and `<final_answer>` structure with a normalized numerical answer.

---

## Comparison With the Teacher

| Model / condition | Exact match | Correct / 1,319 |
|---|---:|---:|
| Qwen3-1.7B base | **74.68%** | 985 |
| Qwen3-1.7B distilled | **79.61%** | 1,050 |
| Qwen3-14B-AWQ teacher | **92.27%** | 1,217 |

Before distillation, the student was **17.59 percentage points** behind the teacher.

After distillation, the remaining gap was **12.66 percentage points**.

The student therefore closed approximately **28% of the original teacher-student accuracy gap**.

---

## Efficiency Trade-Off

| Metric | Before SFT | After SFT |
|---|---:|---:|
| Average completion length | 171.5 tokens | 202.3 tokens |
| Mean inference time | 14.51 s/example | 32.12 s/example |
| Completion throughput | 11.82 tok/s | 6.30 tok/s |
| Peak CUDA memory | 3.34 GiB | 3.41 GiB |
| Adapter storage | - | 0.369 GB |

Distillation improved mathematical accuracy and structured-output behavior, while the evaluated after-SFT configuration generated longer responses and had higher latency. Peak CUDA-memory usage changed only modestly.

---

## Main Scientific Interpretation

The experiment provides evidence that a compact language model can acquire measurable additional mathematical problem-solving capability from a larger teacher through carefully structured teacher-generated supervision.

Three changes occurred simultaneously:

1. **Final-answer accuracy improved:** 74.68% -> 79.61%.
2. **More failures were corrected than successes were lost:** 157 fixes vs. 92 regressions.
3. **Structured-output behavior transferred strongly:** 6.22% -> 98.94% valid format.

Taken together, the results support measurable transfer of both **mathematical task competence** and **structured output behavior** from Qwen3-14B-AWQ to Qwen3-1.7B.

---

## Scope and Evaluation Notes

- Final student results use the untouched official GSM8K test split; test questions were not used as student supervision.
- Distillation data and final evaluation are both from the GSM8K domain, so the reported result demonstrates held-out generalization within this benchmark.
- The reported student result comes from the final QLoRA training run rather than an average across multiple training seeds.
- The teacher reference used a 2,048-token maximum generation budget versus 768 for the student runs; the before-vs.-after student comparison itself is fully matched.

---

## Conclusion

> **Knowledge distillation helped Qwen3-1.7B: GSM8K exact-match accuracy increased from 74.68% to 79.61% (+4.93 pp), yielding 65 additional correct answers net while valid structured outputs increased from 6.22% to 98.94%.**

The paired **157-fix vs. 92-regression** analysis further supports a real improvement in held-out GSM8K problem-solving behavior rather than random answer churn.
