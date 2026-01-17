import os
import json
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# 1) Configurações iniciais
# ==========================
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Iniciando geração com SLM BASE (sem LoRA)")
print(f"📌 Modelo: {BASE_MODEL_ID}")
print(f"📌 Device: {DEVICE}")

# ==========================
# 2) Carregar tokenizer e modelo BASE (SEM device_map)
# ==========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float32   # <-- substitui torch_dtype
)

model.to(DEVICE)
model.eval()

# ==========================
# 3) Engenharia de Prompt (Cibersegurança – 9º Ano – ChatML)
# ==========================
INSTRUCAO_CHATML = (
    "\nResponda APENAS no formato ChatML abaixo:\n"
    "<|system|>\nVocê é um professor de cibersegurança para o 9º ano.<|end|>\n"
    "<|user|>\n[Pergunta]<|end|>\n"
    "<|assistant|>\n"
    "Pergunta: [Texto]\n"
    "A) [Opção]\n"
    "B) [Opção]\n"
    "C) [Opção]\n"
    "D) [Opção]\n"
    "Resposta correta: [Letra]<|end|>"
)

PROMPT_TEMPLATES = {
    "zero-shot": (
        "Gere uma questão de múltipla escolha sobre phishing em redes sociais."
        + INSTRUCAO_CHATML
    ),
    "few-shot": (
        "Exemplo:\n"
        "<|user|>\nCrie uma questão sobre senhas fortes.<|end|>\n"
        "<|assistant|>\n"
        "Pergunta: Qual senha é considerada forte?\n"
        "A) 123456\n"
        "B) senha\n"
        "C) @#Abc12\n"
        "D) data123\n"
        "Resposta correta: C<|end|>\n\n"
        "Agora gere uma nova questão sobre autenticação de dois fatores."
        + INSTRUCAO_CHATML
    ),
    "chain-of-thought": (
        "Pense silenciosamente sobre um risco de Wi-Fi público e gere a questão."
        + INSTRUCAO_CHATML
    ),
    "exemplar-guided": (
        "Exemplo: Maria recebeu um SMS falso do banco pedindo seus dados. "
        "Gere uma questão semelhante sobre engenharia social."
        + INSTRUCAO_CHATML
    ),
    "template-based": (
        "Crie uma questão sobre a importância de manter sistemas e antivírus atualizados."
        + INSTRUCAO_CHATML
    )
}

# ==========================
# 4) Parâmetros do experimento
# ==========================
N_QUESTIONS_PER_TECHNIQUE = 5
TEMPERATURAS = [0.2, 0.5, 0.8]
MAX_NEW_TOKENS = 350

# ==========================
# 5) Função de geração
# ==========================
def generate_with_slm(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return generated.replace("```", "").strip()

# ==========================
# 6) Execução do experimento
# ==========================
resultados = []

print("\n📊 Gerando dataset...")

for technique, prompt in PROMPT_TEMPLATES.items():
    for i in range(1, N_QUESTIONS_PER_TECHNIQUE + 1):
        temp = random.choice(TEMPERATURAS)

        print(f"🤖 SLM_BASE | {technique} | Q{i} | Temp: {temp}")

        saida = generate_with_slm(prompt, temp)

        resultados.append({
            "id": f"SLM_BASE_{technique}_Q{i}",
            "modelo": "SLM_BASE",
            "base_model": BASE_MODEL_ID,
            "tecnica": technique,
            "questao_n": i,
            "temperatura": temp,
            "conteudo_gerado": saida
        })

        time.sleep(0.2)

# ==========================
# 7) Salvar dataset
# ==========================
OUTPUT_FILE = "dataset_dissertacao_slm_base.json"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

print(f"\n✅ Dataset gerado com sucesso!")
print(f"📄 Arquivo: {OUTPUT_FILE}")
print(f"📦 Total de questões: {len(resultados)}")
