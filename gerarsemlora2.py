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

print(f"🚀 Refining generation for TinyLlama BASE on {DEVICE}")

# ==========================
# 2) Carregar tokenizer e modelo
# ==========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# IMPORTANTE: Padding à esquerda e configurar EOS para evitar vázios
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)
if DEVICE == "cpu":
    model.to("cpu")

model.eval()

# ==========================
# 3) Prompts Simplificados (Estilo Instruct)
# ==========================
# Para o modelo BASE, usamos um formato que ele reconhece como tarefa.
def format_prompt(instruction):
    return f"Instruction: {instruction}\n\nResponse:\nQuestion:"

PROMPT_TEMPLATES = {
    "zero-shot": "Generate one 9th-grade multiple-choice question about Phishing. Include 4 options and the correct answer.",
    
    "few-shot": "Example: Question: What is a strong password? A) 123 B) abc C) !@#Abc12 D) pass. Answer: C. Now, generate one 9th-grade question about Two-Factor Authentication (2FA).",
    
    "chain-of-thought": "Think about how hackers use fake Wi-Fi. Based on that, create a multiple-choice question for students about Public Wi-Fi risks.",
    
    "exemplar-guided": "Scenario: A student receives a fake message from Instagram asking for a password. Create a similar multiple-choice question about Social Engineering.",
    
    "template-based": "Topic: Software Updates. Create a multiple-choice question following this format: Question, Options A, B, C, D, and Answer."
}

# ==========================
# 4) Parâmetros
# ==========================
N_QUESTIONS_PER_TECHNIQUE = 5
TEMPERATURAS = [0.3, 0.6] # Reduzi para evitar que ele divague demais
MAX_NEW_TOKENS = 256

# ==========================
# 5) Função de geração robusta
# ==========================
def generate_with_slm(instruction, temperature):
    prompt = format_prompt(instruction)
    # Adicionamos a mask explicitamente
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2, # Aumentado para evitar repetições
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Pegamos apenas o que foi gerado após o prompt
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_part = full_text.split("Response:")[-1].strip()
    
    return generated_part

# ==========================
# 6) Execução
# ==========================
resultados = []

for technique, instruction in PROMPT_TEMPLATES.items():
    for i in range(1, N_QUESTIONS_PER_TECHNIQUE + 1):
        temp = random.choice(TEMPERATURAS)
        print(f"Generating: {technique} Q{i}...")
        
        try:
            saida = generate_with_slm(instruction, temp)
            # Se ainda vier vazio, tentamos uma segunda vez com temp maior
            if not saida:
                saida = generate_with_slm(instruction, 0.7)
        except Exception as e:
            saida = f"Error: {str(e)}"

        resultados.append({
            "id": f"SLM_BASE_{technique}_Q{i}",
            "technique": technique,
            "temperature": temp,
            "generated_content": saida
        })

# ==========================
# 7) Salvar
# ==========================
with open("dataset_fixed.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

print("\n✅ Done! Check dataset_fixed.json")