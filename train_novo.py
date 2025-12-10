# train_tinyllama_lora.py
import os, torch, warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ===== Config rápida =====
BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_FILE  = os.environ.get("DATA_FILE", "clean_train.jsonl")   # gerado no improve_dataset.py
OUT_DIR    = os.environ.get("OUT_DIR",  "tinyllama_sft")
MAX_LEN    = int(os.environ.get("MAX_LEN", "1024"))
EPOCHS     = int(os.environ.get("EPOCHS", "3"))
LR         = float(os.environ.get("LR", "1e-4"))
BATCH      = int(os.environ.get("BATCH", "2"))
SEED       = int(os.environ.get("SEED", "42"))

def format_example(item):
    msgs = item.get("messages") or []
    sys = ""
    usr = ""
    ans = ""
    for m in msgs:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system": sys = content
        elif role == "user": usr = content
        elif role == "assistant": ans = content
    if not (usr and ans) and ("prompt" in item and "response" in item):
        usr, ans = item["prompt"], item["response"]

    parts = []
    if sys: parts.append(f"<|system|>\n{sys}\n</s>")
    parts.append(f"<|user|>\n{usr}\n</s>")
    parts.append(f"<|assistant|>\n{ans}")
    return "\n".join(parts)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Base: {BASE_MODEL} | Device: {device}")

    # Tokenizer (SentencePiece é exigido por Llama/TinyLlama)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})

    # Modelo (fp16 na GPU, fp32 no CPU)
    load_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"]  = "auto"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
    model.resize_token_embeddings(len(tok))

    # LoRA leve para Llama-like
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset -> campo "text"
    raw = load_dataset("json", data_files={"train": DATA_FILE})["train"]
    ds = raw.map(lambda x: {"text": format_example(x)}, remove_columns=raw.column_names)

    def tok_fn(batch):
        out = tok(
            batch["text"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=2 if device=="cuda" else 8,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=(device=="cuda"),
        bf16=False,
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=(device=="cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
        tokenizer=tok,
    )

    print(f"[info] Itens treino: {len(ds_tok)} | Max len: {MAX_LEN}")
    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("[ok] Salvo em:", OUT_DIR)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
