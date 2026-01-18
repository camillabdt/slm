import os
import json
import re
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR   = "tinyllama_lora_mcq"   # pasta onde você salvou o LoRA
DEVICE     = "cpu"                  # mude para "cuda" se tiver GPU

OUT_BASE   = "gerado_base.jsonl"
OUT_TUNED  = "gerado_lora.jsonl"

N_PER_TOPIC = 10
TEMPERATURES = [0.2, 0.5, 0.8]
MAX_NEW_TOKENS = 240

TOPIC_BUCKETS = [
    "phishing",
    "passwords_and_auth",
    "social_engineering",
    "privacy",
    "_default"
]

# =========================
# Prompt igual ao treino
# =========================
def build_prompt(topic_bucket: str) -> str:
    instruction = "Generate one multiple-choice question (4 options) for 9th-grade cybersecurity education. Output ONLY JSON."
    inp = f"Topic bucket: {topic_bucket}"
    return (
        "<|system|>\nYou are a cybersecurity teacher.\n<|end|>\n"
        "<|user|>\n"
        f"{instruction}\n{inp}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

# =========================
# Utils JSON extraction
# =========================
def extract_first_json(text: str):
    if not text:
        return None
    # remove markdown fences
    t = text.replace("```", "").strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def is_valid_mcq(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    for k in ["pergunta", "opcoes", "correta", "feedback"]:
        if k not in obj:
            return False
    if not isinstance(obj["opcoes"], list) or len(obj["opcoes"]) != 4:
        return False
    if not isinstance(obj["correta"], int) or not (0 <= obj["correta"] <= 3):
        return False
    return True

# =========================
# Load models
# =========================
def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    return tok, model

def load_tuned():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    return tok, model

# =========================
# Generation
# =========================
def generate_one(tok, model, prompt, temperature):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.12,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return gen

def run_generation(mode_name, tok, model, out_path):
    total = 0
    ok = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for tb in TOPIC_BUCKETS:
            for i in range(N_PER_TOPIC):
                temp = random.choice(TEMPERATURES)
                prompt = build_prompt(tb)

                raw = generate_one(tok, model, prompt, temp)
                parsed = extract_first_json(raw)

                rec = {
                    "mode": mode_name,
                    "topic_bucket": tb,
                    "i": i,
                    "temperature": temp,
                    "raw": raw,
                    "parsed": parsed,
                    "valid": bool(parsed and is_valid_mcq(parsed))
                }

                if rec["valid"]:
                    ok += 1
                total += 1

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                print(f"[{mode_name}] {tb} #{i+1} | temp={temp} | valid={rec['valid']}")
                time.sleep(0.05)

    print(f"\n✅ {mode_name}: {ok}/{total} válidas. Arquivo: {out_path}\n")

def main():
    random.seed(42)

    # BASE
    tok_b, model_b = load_base()
    run_generation("BASE", tok_b, model_b, OUT_BASE)

    # TUNED (LoRA)
    if os.path.isdir(LORA_DIR):
        tok_t, model_t = load_tuned()
        run_generation("TUNED", tok_t, model_t, OUT_TUNED)
    else:
        print(f"⚠️ Pasta LoRA não encontrada: {LORA_DIR}. Pulei TUNED.")

if __name__ == "__main__":
    main()
