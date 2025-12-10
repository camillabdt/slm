# trainLora2.py — versão estável sem TRL
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH  = os.environ.get("DATA_PATH",  "re_mcq_50.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/re-lora")
SEQ_LEN    = int(os.environ.get("SEQ_LEN", "1024"))

# Quantização moderna (substitui load_in_8bit/load_in_4bit)
bnb = BitsAndBytesConfig(load_in_8bit=True)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = SEQ_LEN

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)

# Config LoRA
lora = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)

# Dataset: cada linha deve ter {"text": "..."}
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset não encontrado em: {DATA_PATH}")
ds = load_dataset("json", data_files={"train": DATA_PATH})

def tokenize_function(example):
    return tok(
        example["text"],
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
    )

tok_ds = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,   # efetivo 64
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,                        # ou bf16=True se sua GPU suportar
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    data_collator=collator,
    tokenizer=tok,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("[OK] Saved LoRA to", OUTPUT_DIR)
