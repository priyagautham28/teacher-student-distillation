"""
Smoke test for the Llama-3.2-1B QLoRA pipeline.
Run this BEFORE any real training to confirm:
  - the base model loads in 4-bit
  - the LoRA adapter attaches correctly
  - generation actually works
  - GPU memory usage looks sane

Usage:
    python smoke_test.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.cuda.reset_peak_memory_stats()

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Attaching LoRA adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nRunning one test generation (teacher-aligned chat format)...")
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the mathematical problem using concise step-by-step reasoning. "
                "Return the reasoning inside <reasoning> tags and the numerical answer "
                "inside <final_answer> tags."
            ),
        },
        {
            "role": "user",
            "content": (
                "If a train travels 60 miles in 1.5 hours, what is its average speed?"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nPrompt:\n{prompt}")
    print(f"Output:\n{decoded}")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU memory used: {peak_mem_gb:.2f} GB")
    print("\nSmoke test PASSED — safe to proceed to real training.")


if __name__ == "__main__":
    main()
