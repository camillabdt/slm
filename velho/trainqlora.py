# trainqlora.py
# Versão FINAL: TinyLlama 1.1B (SLM) + LoRA (para baixo consumo de RAM) + 8-bit Quantização.

import os, json, math
from dataclasses import dataclass
from typing import Dict, List, Any, Union, Optional

# Certifique-se de que estão instalados: pip install torch transformers datasets accelerate peft bitsandbytes
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
# PEFT é necessário para usar LoRA no modelo quantizado
from peft import LoraConfig, get_peft_model, TaskType

# ----- CONFIG BÁSICA -----
BASE_DIR        = os.path.dirname(__file__)
# Modelo: TinyLlama 1.1B Chat (Melhor equilíbrio entre performance e RAM).
BASE_MODEL      = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0") 
TRAIN_FILE      = os.environ.get("TRAIN_FILE", os.path.join(BASE_DIR, "questions.json"))
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "out-tinyllama-sft"))

# Hiperparâmetros (ajustados para a RAM mínima possível)
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "1")) 
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
LR              = float(os.environ.get("LR", "2e-5"))
EPOCHS          = int(os.environ.get("EPOCHS", "5"))
MAX_LENGTH      = int(os.environ.get("MAX_LENGTH", "512")) 
WARMUP_RATIO    = float(os.environ.get("WARMUP_RATIO", "0.03"))

# LoRA (AGORA É OBRIGATÓRIO PARA USAR 8-BIT)
ENABLE_LORA     = os.environ.get("ENABLE_LORA", "1") == "1" # <-- CORREÇÃO: HABILITADO
LORA_R          = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA      = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT    = float(os.environ.get("LORA_DROPOUT", "0.05"))

# Seed
SEED = int(os.environ.get("SEED", "42"))

# ----- CHECAGENS -----
assert os.path.exists(TRAIN_FILE), f"Dataset não encontrado: {TRAIN_FILE}"

# Limpa o cache de GPU/CPU antes de carregar o modelo (Pode evitar travamentos)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ----- TOKENIZER -----
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

# ----- MODELO -----
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    # Quantização para 8-bit (Reduz a RAM de ~4.4GB para ~2.2GB se funcionar)
    load_in_8bit=True, # OTIMIZAÇÃO CRÍTICA
    torch_dtype=torch.float32, 
    device_map=None, # Mantido None para CPU
)

# LoRA (AGORA ATIVADO E NECESSÁRIO)
if ENABLE_LORA:
    try:
        # Heurística comum para LLaMA/derivados:
        target_modules = []
        for name, module in model.named_modules():
            n = name.lower()
            if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                # Filtra módulos de embedding para evitar erro de float32 vs int8
                if "embed" not in name.lower():
                    target_modules.append(name.split(".")[-1])
        
        target_modules = list(sorted(set(target_modules))) or ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

        lcfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            task_type=TaskType.CAUSAL_LM, target_modules=target_modules,
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    except Exception as e:
        # Se LoRA falhar (biblioteca ausente), ele avisa
        raise RuntimeError(f"Falha ao habilitar LoRA (PEFT). Erro: {e}. Instale 'peft' e 'bitsandbytes'.")

# ----- DATASET E MASKING (Corrigido para o formato Quiz) -----
raw = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

def build_text_and_mask(sample: Dict[str, Any]) -> Dict[str, Any]:
    stem = sample.get("stem", "")
    options = sample.get("options", [])
    correct_idx = sample.get("correctOption", -1) 
    explanation = sample.get("explanation", "")

    # 1. CONSTRUÇÃO DO PROMPT (USER)
    options_str = ""
    for i, opt in enumerate(options):
        options_str += f"\n{chr(65 + i)}. {opt}" 

    prompt = f"System: Você é um especialista. Responda à pergunta escolhendo a melhor opção (A, B, C, D, E).\nUser: {stem}{options_str}\nAssistant:"

    # 2. CONSTRUÇÃO DA RESPOSTA (ASSISTANT)
    try:
        correct_option_text = options[correct_idx]
        correct_letter = chr(65 + correct_idx)
    except IndexError:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    target = f" {correct_letter}. {correct_option_text}\nExplicação: {explanation}{tok.eos_token}"

    # 3. TOKENIZAÇÃO E MASKING
    prompt_ids = tok(prompt, add_special_tokens=False)
    target_ids = tok(target, add_special_tokens=False)

    input_ids = prompt_ids["input_ids"] + target_ids["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids["input_ids"]) + target_ids["input_ids"]

    # Truncagem simples
    if len(input_ids) > MAX_LENGTH:
        overflow = len(input_ids) - MAX_LENGTH
        input_ids = input_ids[overflow:]
        attention_mask = attention_mask[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def format_example(example):
    if not example.get("stem") or not example.get("options"):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    result = build_text_and_mask(example)
    
    if not result.get("input_ids"):
         return {"input_ids": [], "attention_mask": [], "labels": []}
         
    return result

proc = raw.map(format_example, remove_columns=raw.column_names, desc="Processando amostras de Quiz")
proc = proc.filter(lambda x: len(x['input_ids']) > 0, desc="Filtrando amostras vazias/inválidas") 


# ----- COLLATOR (Mantido) -----
@dataclass
class LMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id

        for f in features:
            seq, att, lab = f["input_ids"], f["attention_mask"], f["labels"]
            pad_len = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad_len)
            attention_mask.append(att + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch

collator = LMDataCollator(tokenizer=tok)

# ----- TREINAMENTO (Mantido) -----
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.0,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=1, 
    save_steps=200,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    fp16=False,
    optim="adamw_torch",
    report_to="none",
    seed=SEED,
    dataloader_pin_memory=False, 
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=proc,
    data_collator=collator,
    tokenizer=tok,
)

if __name__ == "__main__":
    print(f"[info] Treinando em: {('GPU' if torch.cuda.is_available() else 'CPU')}")
    print(f"[info] Ex.: {len(proc)} amostras | Max seq len: {MAX_LENGTH}")
    try:
        trainer.train()
        # Limpa o cache novamente após o treinamento
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Salva pesos e tokenizer
        trainer.save_model(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        print(f"[ok] Modelo salvo em: {OUTPUT_DIR}")
    except RuntimeError as e:
        print(f"[ERRO CRÍTICO] Falha durante o treinamento. Provável Out of Memory (OOM) na RAM. Erro: {e}")