import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# =========================
# CONFIGURAÇÕES
# =========================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE  = "mcq_train.jsonl"
OUT_DIR    = "tinyllama_lora_mcq"

EPOCHS     = 3
BATCH      = 1
GRAD_ACCUM = 8
LR         = 2e-4
LOG_STEPS  = 10
SAVE_STEPS = 200
SEED       = 42
MAX_LEN    = 512

DEVICE = "cpu"  # compatível com seu ambiente atual

# =========================
# HELPERS
# =========================
def safe_json_loads(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def build_text_from_record(instruction, inp, mcq_obj):
    """
    Monta um texto de treino causal LM:
    prompt (instruction+input) -> completion (output JSON MCQ)
    """
    out_json = json.dumps(mcq_obj, ensure_ascii=False)

    text = (
        "<|system|>\nYou are a cybersecurity teacher.\n<|end|>\n"
        "<|user|>\n"
        f"{instruction}\n"
        f"{inp}\n"
        "<|end|>\n"
        "<|assistant|>\n"
        f"{out_json}\n"
        "<|end|>\n"
    )
    return text

def normalize_mcq(mcq_obj):
    """
    Garante que MCQ tem chaves esperadas e opcoes 4 itens.
    """
    if not isinstance(mcq_obj, dict):
        return None
    needed = ["pergunta", "opcoes", "correta", "feedback"]
    if not all(k in mcq_obj for k in needed):
        return None
    if not isinstance(mcq_obj["opcoes"], list) or len(mcq_obj["opcoes"]) != 4:
        return None
    if not isinstance(mcq_obj["correta"], int) or not (0 <= mcq_obj["correta"] <= 3):
        return None
    return mcq_obj

# =========================
# MAIN
# =========================
def main():
    print("📌 Device:", DEVICE)
    print("📌 Base model:", BASE_MODEL)
    print("📌 Data:", DATA_FILE)
    print("📌 Out:", OUT_DIR)

    # -------------------------
    # TOKENIZER
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # DATASET
    # -------------------------
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)

    print(f"✅ Dataset loaded: train={len(dataset['train'])} | eval={len(dataset['test'])}")

    def format_example(example):
        """
        Seu mcq_train.jsonl tem: instruction, input, output (string JSON)
        """
        instruction = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        output_str = example.get("output", "")

        mcq_obj = safe_json_loads(output_str)
        mcq_obj = normalize_mcq(mcq_obj)

        # Se algum exemplo estiver ruim, retorna texto vazio (vai ser filtrado depois)
        if not instruction or not mcq_obj:
            return {"text": ""}

        text = build_text_from_record(instruction, inp, mcq_obj)
        return {"text": text}

    # cria coluna "text"
    dataset = dataset.map(format_example)

    # filtra exemplos vazios
    dataset["train"] = dataset["train"].filter(lambda x: isinstance(x["text"], str) and len(x["text"]) > 20)
    dataset["test"]  = dataset["test"].filter(lambda x: isinstance(x["text"], str) and len(x["text"]) > 20)

    print(f"✅ After filtering: train={len(dataset['train'])} | eval={len(dataset['test'])}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    # -------------------------
    # MODELO BASE
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )
    model.to(DEVICE)

    # -------------------------
    # LoRA
    # -------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # TRAINING ARGS (compatível com transformers antigos)
    # -------------------------
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # -------------------------
    # TREINAR
    # -------------------------
    print("🚀 Starting fine-tuning...")
    trainer.train()

    # -------------------------
    # SALVAR
    # -------------------------
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("✅ Fine-tuning finalizado com sucesso!")
    print("📦 Adapter salvo em:", OUT_DIR)

if __name__ == "__main__":
    main()
