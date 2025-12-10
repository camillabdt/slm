#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# -------------------------------
# Config por variável de ambiente
# -------------------------------
MODEL_ID     = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TRAIN_PATH   = os.environ.get("TRAIN_PATH", "clean_train.jsonl")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "out-tiny-sft")

BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "16"))
MAX_SEQ_LEN  = int(os.environ.get("MAX_SEQ_LEN", "768"))
EPOCHS       = float(os.environ.get("EPOCHS", "2"))
LR           = float(os.environ.get("LR", "2e-5"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.06"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", "50"))
EVAL_STEPS   = int(os.environ.get("EVAL_STEPS", "0"))  # 0 = sem avaliação
LOG_STEPS    = int(os.environ.get("LOG_STEPS", "10"))
SEED         = int(os.environ.get("SEED", "42"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def _join_messages(msgs: List[Dict[str, str]]) -> str:
    lines = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"[{role.upper()}] {content}")
    return "\n".join(lines).strip()


def _format_mcq(ex: Dict[str, Any]) -> Optional[str]:
    stem = ex.get("stem")
    options = ex.get("options")
    correct = ex.get("correct")
    rationale = ex.get("rationale")

    if stem is None or options is None or correct is None:
        return None

    if isinstance(options, list):
        letters = "ABCDE"
        options = {letters[i]: opt for i, opt in enumerate(options) if i < len(letters)}

    opts_lines = []
    for k in ["A", "B", "C", "D", "E"]:
        if k in options:
            opts_lines.append(f"{k}) {options[k]}")

    rat = f"\n### Explicação:\n{rationale}".strip() if rationale else ""
    text = (
        f"### Pergunta:\n{stem}\n\n"
        f"### Opções:\n" + "\n".join(opts_lines) + "\n\n"
        f"### Resposta correta: {correct}\n"
        f"{rat}"
    ).strip()
    return text


def _to_text_examples(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    cols = ds.column_names

    if "text" in cols:
        return ds

    if "messages" in cols:
        def _m2t(ex):
            try:
                return {"text": _join_messages(ex["messages"])}
            except Exception:
                return {"text": None}
        tmp = ds.map(_m2t)
        return tmp.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

    if "instruction" in cols and "output" in cols:
        def _io2t(ex):
            ins = (ex.get("instruction") or "").strip()
            out = (ex.get("output") or "").strip()
            if not ins and not out:
                return {"text": None}
            return {"text": f"### Instrução:\n{ins}\n\n### Resposta:\n{out}"}
        tmp = ds.map(_io2t)
        return tmp.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

    if "prompt" in cols and "completion" in cols:
        def _pc2t(ex):
            p = (ex.get("prompt") or "").strip()
            c = (ex.get("completion") or "").strip()
            if not p and not c:
                return {"text": None}
            return {"text": f"### Prompt:\n{p}\n\n### Completion:\n{c}"}
        tmp = ds.map(_pc2t)
        return tmp.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

    if all(k in cols for k in ["stem", "options", "correct"]):
        def _mcq2t(ex):
            try:
                return {"text": _format_mcq(ex)}
            except Exception:
                return {"text": None}
        tmp = ds.map(_mcq2t)
        return tmp.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

    raise ValueError(
        f"Não encontrei colunas de texto válidas nas colunas {cols}.\n"
        "Esperado: 'text' ou 'messages' (chat) ou ('prompt','completion') ou ('instruction','output') ou (MCQ 'stem','options','correct')."
    )


def main():
    print(f"[info] Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    set_seed(SEED)

    # --- Modelo e tokenizer ---
    print(f"[info] Carregando modelo: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,      # CPU estável
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # --- Dataset ---
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Arquivo '{TRAIN_PATH}' não encontrado.")

    raw = load_dataset("json", data_files={"train": TRAIN_PATH})["train"]
    print(f"[ok] Dataset bruto: {len(raw)} linhas")

    ds_txt = _to_text_examples(raw, tok)
    print(f"[ok] Após conversão → 'text': {len(ds_txt)} exemplos")

    # Tokenização com truncation (padding fica por conta do collator)
    def _tok_fn(ex):
        return tok(ex["text"], max_length=MAX_SEQ_LEN, truncation=True)

    # >>> AQUI O AJUSTE CRÍTICO: remover TODAS as colunas originais (inclui 'text')
    ds_tok = ds_txt.map(_tok_fn, batched=False, remove_columns=ds_txt.column_names)
    ds_tok = ds_tok.filter(lambda ex: "input_ids" in ex and len(ex["input_ids"]) > 0)

    if len(ds_tok) == 0:
        print("[warn] Nenhum exemplo válido após tokenização.")
        return

    ds = DatasetDict({"train": ds_tok})

    # Collator de LM com padding dinâmico (gera labels a partir de input_ids)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # --- Args de treino (estáveis e com checkpoints) ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOG_STEPS,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="no" if EVAL_STEPS <= 0 else "steps",  # <- substitui evaluation_strategy
        eval_steps=None if EVAL_STEPS <= 0 else EVAL_STEPS,
        report_to=["none"],
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        fp16=False, bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,  # importante p/ evitar drop indevido
        seed=SEED,
    )

    last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_ckpt:
        print(f"[info] Retomando do checkpoint: {last_ckpt}")

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tok,                   # OK (aviso depreciação apenas informativo)
        train_dataset=ds["train"],
        data_collator=collator,
    )

    steps_ep = math.ceil(len(ds["train"]) / max(1, BATCH_SIZE))
    print(f"[info] steps/epoch (aprox): {steps_ep}, exemplos: {len(ds['train'])}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"[ok] Modelo salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
