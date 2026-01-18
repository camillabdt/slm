import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== AJUSTE AQUI ======
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR   = "tinyllama_lora_mcq"   # pasta do seu LoRA treinado
TOPIC_BUCKET = "phishing"           # ex.: phishing, passwords_and_auth, social_engineering, privacy, _default
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 220
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def extract_json(text: str):
    if not text:
        return None
    t = text.replace("```", "").strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    print(f"Device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    prompt = build_prompt(TOPIC_BUCKET)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.12,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id
        )

    raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    obj = extract_json(raw)

    print("\n===== RAW OUTPUT =====\n")
    print(raw)

    if obj:
        print("\n===== PARSED JSON =====\n")
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print("\n⚠️ Não consegui extrair JSON válido. (Mas o bruto acima está aí.)")

if __name__ == "__main__":
    main()
