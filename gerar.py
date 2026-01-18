import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= CONFIG =================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH  = "tinyllama_lora_mcq"   # ajuste se necessário
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 0.3
MAX_NEW_TOKENS = 260

TEMA = "Phishing in social networks"

# mesmas 5 técnicas
PROMPTS = {
    "zero-shot": (
        "Generate ONE multiple-choice question about cybersecurity.\n"
        f"Topic: {TEMA}\n"
        "Level: 9th grade.\n"
        "Output ONLY JSON with keys: pergunta, opcoes (4 items), correta (0-3), feedback."
    ),
    "few-shot": (
        "Example:\n"
        '{"pergunta":"Which password is strong?","opcoes":["12345","password","Ab#9x!","qwerty"],"correta":2,"feedback":"A strong password mixes letters, numbers, and symbols."}\n\n'
        f"Now generate ONE new question about {TEMA}.\n"
        "Output ONLY JSON with keys: pergunta, opcoes (4 items), correta (0-3), feedback."
    ),
    "chain-of-thought": (
        "Think silently about a common risk and prevention. Do NOT show your reasoning.\n"
        f"Topic: {TEMA}\n"
        "Generate ONE multiple-choice question.\n"
        "Output ONLY JSON with keys: pergunta, opcoes (4 items), correta (0-3), feedback."
    ),
    "exemplar-guided": (
        "Scenario: A student receives a message pretending to be from a social network asking to confirm login data.\n"
        f"Create a similar scenario and generate ONE question about {TEMA}.\n"
        "Output ONLY JSON with keys: pergunta, opcoes (4 items), correta (0-3), feedback."
    ),
    "template-based": (
        "Fill the template as JSON:\n"
        '{"pergunta":"...","opcoes":["...","...","...","..."],"correta":0,"feedback":"..."}\n'
        f"Topic: {TEMA}\n"
        "Output ONLY JSON (no extra text)."
    ),
}

def build_chatml(user_text: str) -> str:
    # bem simples, sem “Check your answer”
    return (
        "<|system|>\nYou are a cybersecurity teacher.\nReturn ONLY JSON.\n<|end|>\n"
        "<|user|>\n"
        f"{user_text}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

# ---------------- JSON extraction & normalization ----------------
def extract_first_json(text: str):
    if not text:
        return None
    t = text.replace("```", "").strip()

    # pega o primeiro bloco {...} (não-guloso)
    m = re.search(r"\{[\s\S]*?\}", t)
    if not m:
        return None
    cand = m.group(0)

    try:
        obj = json.loads(cand)
    except Exception:
        return None

    # normaliza chaves erradas (se aparecerem)
    keymap = {
        "question": "pergunta", "stem": "pergunta",
        "options": "opcoes", "opcions": "opcoes",
        "answer": "correta", "correctOption": "correta", "correctAnswer": "correta",
        "explanation": "feedback",
        "feed": "feedback",
    }
    for k in list(obj.keys()):
        if k in keymap:
            obj[keymap[k]] = obj.pop(k)

    # normaliza correta se vier letra "A/B/C/D"
    if isinstance(obj.get("correta"), str):
        s = obj["correta"].strip().upper()
        if s in ["A", "B", "C", "D"]:
            obj["correta"] = ["A", "B", "C", "D"].index(s)

    return obj

def validate_mcq(obj):
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

def pretty_print_mcq(tecnica: str, obj: dict, show_feedback: bool = True):
    labels = ["A", "B", "C", "D"]
    print(f"\n=== Técnica: {tecnica} ===")
    print(f"Pergunta: {obj['pergunta']}")
    for i, opt in enumerate(obj["opcoes"]):
        print(f"{labels[i]}) {opt}")
    ans_letter = labels[obj["correta"]]
    print(f"Resposta correta: {ans_letter}")
    if show_feedback:
        print(f"Feedback: {obj['feedback']}")

# ---------------- Model load & generation ----------------
def main():
    print(f"📌 Device: {DEVICE}")
    print(f"📌 Base model: {BASE_MODEL}")
    print(f"📌 LoRA: {LORA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(DEVICE)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    for tecnica, user_text in PROMPTS.items():
        prompt = build_chatml(user_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.12,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        obj = extract_first_json(raw)
        if obj and validate_mcq(obj):
            pretty_print_mcq(tecnica, obj, show_feedback=True)
        else:
            print(f"\n=== Técnica: {tecnica} ===")
            print("⚠️ Não consegui extrair uma questão válida (4 opções + correta 0–3).")
            print("RAW (primeiros 500 chars):")
            print(raw[:500])

if __name__ == "__main__":
    main()
