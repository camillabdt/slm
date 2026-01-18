import os
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
# CONFIGURAÇÕES GERAIS
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

DEVICE = "cpu"

# =========================
# FUNÇÃO PRINCIPAL
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
        Converte cada MCQ em texto de treino (causal LM)
        """
        text = (
            "<|system|>\n"
            "You are a cybersecurity teacher.\n"
            "<|end|>\n"
            "<|user|>\n"
            f"{example['question']}\n"
            + "\n".join(example["options"]) +
            "\n<|end|>\n"
            "<|assistant|>\n"
            f"Correct answer: {example['answer']}\n"
            f"{example['feedback']}\n"
            "<|end|>"
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

    dataset = dataset.map(tokenize, batched=True)

    # -------------------------
    # MODELO BASE
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

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
    # TREINAMENTO
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

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

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()
