import os
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# =========================
# CONFIG
# =========================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "mcq_train.jsonl"
OUT_DIR = "tinyllama_lora_mcq"

# Treino (ajuste se quiser)
SEED = 42
TRAIN_SPLIT = 0.9          # 90% treino, 10% validação
EPOCHS = 4                 # com ~300 linhas, 3-5 é ok
LR = 2e-4
BATCH = 2
GRAD_ACCUM = 8
MAX_SEQ_LEN = 768          # MCQ em JSON costuma caber bem aqui
LOG_STEPS = 10
SAVE_STEPS = 200

# =========================
# Helpers
# =========================
def format_row(ex):
    # ex tem: instruction, input, output (output é string JSON)
    return (
        f"### Instruction:\n{ex['instruction']}\n"
        f"### Input:\n{ex['input']}\n"
        f"### Output:\n{ex['output']}"
    )

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Device: {device}")
    print(f"📌 Base model: {BASE_MODEL}")
    print(f"📌 Data: {DATA_PATH}")
    print(f"📌 Out:  {OUT_DIR}")

    # -------------------------
    # Load dataset + split
    # -------------------------
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.shuffle(seed=SEED)
    split = ds.train_test_split(test_size=(1 - TRAIN_SPLIT), seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"✅ Dataset loaded: train={len(train_ds)} | eval={len(eval_ds)}")

    # -------------------------
    # Load tokenizer + model
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype: float16 na GPU, float32 na CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype
    )
    model.to(device)

    # -------------------------
    # LoRA config (estável)
    # -------------------------
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -------------------------
    # Training args
    # -------------------------
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        seed=SEED,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    # -------------------------
    # Trainer (SFT)
    # -------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
        formatting_func=format_row,
        max_seq_length=MAX_SEQ_LEN,
        args=args,
    )

    print("🚀 Training started...")
    trainer.train()

    print("💾 Saving LoRA adapter...")
    trainer.save_model(OUT_DIR)
    print(f"✅ Saved to: {OUT_DIR}")

    # -------------------------
    # Quick sanity test (1 sample)
    # -------------------------
    sample_topic = "phishing"
    prompt = (
        "### Instruction:\nGenerate one multiple-choice question (4 options) for 9th-grade cybersecurity education. Output ONLY JSON.\n"
        f"### Input:\nTopic bucket: {sample_topic}\n"
        "### Output:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = trainer.model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.12,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n🧪 Sample generation:\n", gen)

if __name__ == "__main__":
    main()
