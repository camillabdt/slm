import os, json, torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

# Configurações
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_FILE = "datasetoficial.jsonl"
LORA_DIR = "tinyllama_lora_final"
MAX_LENGTH = 384

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_dataset(items, tokenizer):
    def format_chatml(messages):
        return "".join([f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in messages])
    
    formatted = [{"text": format_chatml(x["messages"])} for x in items]
    return Dataset.from_list(formatted).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"),
        batched=True, remove_columns=["text"]
    )

def make_trainer(model, tokenizer, ds, lr, steps):
    args = TrainingArguments(
        output_dir="./temp_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=steps,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[]
    )
    return Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

# Execução do Treino
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

raw_data = load_jsonl(DATASET_FILE)
ds_content = build_dataset([x for x in raw_data if x.get("task") == "content"], tokenizer)
ds_mcq = build_dataset([x for x in raw_data if x.get("task") == "mcq"], tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(base_model, peft_config)

print("\n--- Iniciando Fase 1 (Content) ---")
make_trainer(model, tokenizer, ds_content, 1e-4, 400).train()

print("\n--- Iniciando Fase 2 (MCQ) ---")
make_trainer(model, tokenizer, ds_mcq, 8e-5, 250).train()

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"\nTreino concluído! Modelo salvo em: {LORA_DIR}")