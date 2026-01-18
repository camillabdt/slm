import json, torch, os
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configurações de Caminho e Modelo
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_FILE = "datasetoficial.jsonl"
LORA_DIR = "tinyllama_lora_final"
MAX_LENGTH = 384

def load_jsonl(path):
    if not os.path.exists(path):
        print(f"Erro: Arquivo {path} não encontrado.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def format_chat(messages, tokenizer):
    fixed = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, dict):
            c = json.dumps(c, ensure_ascii=False)
        fixed.append({"role": m["role"], "content": c})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            fixed, tokenize=False, add_generation_prompt=False
        )
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in fixed])

def build_dataset(items, tokenizer):
    formatted = [{"text": format_chat(x["messages"], tokenizer)} for x in items]
    ds = Dataset.from_list(formatted)
    ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    return ds

def make_trainer(model, tokenizer, ds, lr, steps, phase_name):
    # Definindo diretório específico para os checkpoints desta fase
    output_dir = f"./checkpoints_{phase_name}"
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=steps,
        learning_rate=lr,
        logging_steps=20,
        
        # --- Configurações de Salvamento ---
        save_strategy="steps",       # Salva por passos, não por época
        save_steps=100,              # Salva a cada 100 passos
        save_total_limit=2,          # Mantém apenas os 2 últimos checkpoints (evita encher o disco)
        # ----------------------------------
        
        report_to=[],
        fp16=torch.cuda.is_available()
    )
    
    return Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

# 1. Preparação
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

raw_data = load_jsonl(DATASET_FILE)
if not raw_data:
    exit()

ds_content = build_dataset([x for x in raw_data if x.get("task") == "content"], tokenizer)
ds_mcq = build_dataset([x for x in raw_data if x.get("task") == "mcq"], tokenizer)

# 2. Carregamento do Modelo
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(base_model, peft_config)

# 3. Treinamento - Fase 1
print("\n--- Phase 1 (Content) ---")
trainer_content = make_trainer(model, tokenizer, ds_content, 1e-4, 400, "content")
# Para retomar de um erro, use: trainer_content.train(resume_from_checkpoint=True)
trainer_content.train() 

# 4. Treinamento - Fase 2
print("\n--- Phase 2 (MCQ) ---")
trainer_mcq = make_trainer(model, tokenizer, ds_mcq, 8e-5, 250, "mcq")
# Para retomar de um erro, use: trainer_mcq.train(resume_from_checkpoint=True)
trainer_mcq.train()

# 5. Salvamento Final
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"\nSucesso! Modelo final salvo em: {LORA_DIR}")