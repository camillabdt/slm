import os, json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ==========================================
# 1) Configurações
# ==========================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_FILE = "datasetoficial.jsonl"

LORA_DIR = "tinyllama_lora_9th_cybersecurity_2phases"

MAX_LENGTH = 384

# Passos por fase (ajuste conforme quantidade de dados)
MAX_STEPS_PHASE1 = 400   # content
MAX_STEPS_PHASE2 = 250   # mcq

LR_PHASE1 = 1e-4
LR_PHASE2 = 8e-5

GRAD_ACCUM = 8
BATCH_SIZE = 1

# ==========================================
# 2) Utilitários
# ==========================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def messages_to_chatml(messages):
    parts = []
    for m in messages:
        role = m["role"].strip()
        content = m["content"].strip()
        parts.append(f"<|{role}|>\n{content}<|end|>\n")
    return "".join(parts)

def build_dataset(items, tokenizer, max_length):
    formatted = [{"text": messages_to_chatml(x["messages"])} for x in items]
    ds = Dataset.from_list(formatted).map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        ),
        batched=True,
        remove_columns=["text"]
    )
    return ds

def make_trainer(model, tokenizer, ds, output_dir, max_steps, lr, logging_steps=20):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=max_steps,
        learning_rate=lr,
        use_cpu=True,
        logging_steps=logging_steps,
        save_strategy="no",
        report_to=[]  # evita logs externos
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    return trainer

# ==========================================
# 3) Carregar Tokenizer
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==========================================
# 4) Treino em 2 fases
# ==========================================
def train_two_phases():
    print(f"Loading dataset from: {DATASET_FILE}")
    raw = load_jsonl(DATASET_FILE)

    content_items = [x for x in raw if x.get("task") == "content"]
    mcq_items = [x for x in raw if x.get("task") == "mcq"]

    print(f"Items -> content: {len(content_items)}, mcq: {len(mcq_items)}")

    ds_content = build_dataset(content_items, tokenizer, MAX_LENGTH)
    ds_mcq = build_dataset(mcq_items, tokenizer, MAX_LENGTH)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(base_model, peft_config)

    # ---------
    # FASE 1: Content
    # ---------
    print(f"\n=== PHASE 1: CONTENT (max_steps={MAX_STEPS_PHASE1}, lr={LR_PHASE1}) ===")
    trainer1 = make_trainer(
        model=model,
        tokenizer=tokenizer,
        ds=ds_content,
        output_dir="./results_phase1",
        max_steps=MAX_STEPS_PHASE1,
        lr=LR_PHASE1
    )
    trainer1.train()

    # (Opcional) salvar checkpoint intermediário
    phase1_dir = LORA_DIR + "_phase1"
    model.save_pretrained(phase1_dir)
    tokenizer.save_pretrained(phase1_dir)
    print(f"Saved Phase 1 LoRA to: {phase1_dir}")

    # ---------
    # FASE 2: MCQ (continua com o MESMO LoRA)
    # ---------
    print(f"\n=== PHASE 2: MCQ (max_steps={MAX_STEPS_PHASE2}, lr={LR_PHASE2}) ===")
    trainer2 = make_trainer(
        model=model,
        tokenizer=tokenizer,
        ds=ds_mcq,
        output_dir="./results_phase2",
        max_steps=MAX_STEPS_PHASE2,
        lr=LR_PHASE2
    )
    trainer2.train()

    # Salvar final
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"\nTraining complete. Final LoRA saved to: {LORA_DIR}")

    return model

# ==========================================
# 5) Carregar/treinar
# ==========================================
if not os.path.exists(LORA_DIR):
    model = train_two_phases()
else:
    print("Loading existing LoRA...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)

model.eval()

# ==========================================
# 6) Geração: MCQ JSON
# ==========================================
def generate_mcq(topic):
    prompt = (
        "<|system|>\n"
        "You generate multiple-choice questions for 9th-grade cybersecurity classes. "
        "Output must be valid JSON with keys: stem, options (5 items), correctOption (0-4), explanation. "
        "Exactly one correct option. Do not include any extra text outside the JSON object.\n"
        "<|end|>\n"
        "<|user|>\n"
        f"Create 1 multiple-choice question about: {topic}.\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # recorta só o JSON (para teste rápido)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text.strip()

# Teste rápido
for t in ["Phishing", "Two-Factor Authentication", "Public Wi-Fi"]:
    print(f"\n--- {t} ---")
    print(generate_mcq(t))
