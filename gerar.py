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
MAX_NEW_TOKENS = 250

TEMA = "Phishing in social networks"
# ==========================================

# -------- PROMPTS (mesmos 5) ---------------
PROMPTS = {
    "zero-shot": (
        "Generate ONE multiple-choice question about cybersecurity.\n"
        f"Topic: {TEMA}\n"
        "Level: 9th grade."
    ),

    "few-shot": (
        "Example:\n"
        "Question: Which password is strong?\n"
        "A) 12345\n"
        "B) password\n"
        "C) Ab#9x!\n"
        "D) qwerty\n"
        "Correct answer: C\n\n"
        f"Now generate ONE new question about {TEMA}."
    ),

    "chain-of-thought": (
        "Think step by step:\n"
        "1) Identify a cybersecurity risk.\n"
        "2) Explain the consequence.\n"
        "3) Generate ONE multiple-choice question.\n\n"
        f"Topic: {TEMA}."
    ),

    "exemplar-guided": (
        "Scenario:\n"
        "A student receives a message pretending to be from a social network asking to confirm login data.\n\n"
        f"Create a similar scenario and generate ONE question about {TEMA}."
    ),

    "template-based": (
        "Fill the template:\n"
        "Question: [text]\n"
        "A) [option]\n"
        "B) [option]\n"
        "C) [option]\n"
        "D) [option]\n"
        "Correct answer: [letter]\n\n"
        f"Topic: {TEMA}."
    )
}
# -------------------------------------------


def build_chatml(prompt_text: str) -> str:
    return (
        "<|system|>\n"
        "You are a cybersecurity teacher.\n"
        "Respond ONLY in valid JSON.\n"
        "<|end|>\n"
        "<|user|>\n"
        f"{prompt_text}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )


def extract_json(text: str):
    if not text:
        return None
    text = text.replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def main():
    print(f"📌 Device: {DEVICE}")
    print(f"📌 Base model: {BASE_MODEL}")
    print(f"📌 LoRA: {LORA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    ).to(DEVICE)

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    results = {}

    for tecnica, prompt_text in PROMPTS.items():
        print(f"\n🛠️ Técnica: {tecnica}")

        prompt = build_chatml(prompt_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        raw = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        parsed = extract_json(raw)

        results[tecnica] = {
            "raw_output": raw,
            "parsed_json": parsed
        }

        print("🔹 RAW:")
        print(raw)

        if parsed:
            print("✅ JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            print("⚠️ JSON inválido")

    print("\n🎯 FIM DA GERAÇÃO")


if __name__ == "__main__":
    main()
