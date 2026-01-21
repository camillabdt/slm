import os
import json
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================
# 1) Configurações iniciais
# ==========================
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "./modelo_final_cpu_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting generation with FINE-TUNED SLM (LoRA)")
print(f"📌 Base: {BASE_MODEL_ID} | Adapter: {LORA_ADAPTER_PATH}")

# ==========================
# 2) Carregar Tokenizer e Modelo com Adaptador
# ==========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)

# Acopla o Fine-Tuning ao modelo base
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model.to(DEVICE)
model.eval()

# ==========================
# 3) Prompts Estruturados (Refletindo o Treinamento)
# ==========================
SYSTEM_PROMPT = "You are a cybersecurity teacher for 9th-grade students. Create high-quality multiple-choice questions."

PROMPT_TEMPLATES = {
    "zero-shot": "Create one MCQ about Phishing risks. Include Question, Options A-D, and Correct Answer.",
    
    "few-shot": "Example: Q: What is 2FA? A) Virus B) Security step C) Game D) App. Ans: B. Now, create a new MCQ about Strong Passwords.",
    
    "chain-of-thought": "Step 1: Explain why hackers use urgency. Step 2: Create an MCQ about Phishing based on this logic.",
    
    "exemplar-guided": "Style: 'A firewall is like a security guard'. Create an MCQ about Antivirus using a similar analogy.",
    
    "template-based": "Follow this template: Topic: [Topic]\nQuestion: [Text]\nOptions: A-D\nCorrect: [Letter]"
}

# ==========================
# 4) Função de Geração com Attention Mask
# ==========================
def generate_with_lora(user_instruction, temperature):
    # Usa o formato de chat que o modelo aprendeu no fine-tuning
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    # Criar a attention_mask explicitamente para evitar os erros do modelo base
    attention_mask = torch.ones(input_ids.shape, device=model.device, dtype=torch.long)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2, # Crítico para evitar repetições no TinyLlama
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decodifica apenas a resposta nova (pós-prompt)
    generated_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    return generated_text.strip()

# ==========================
# 5) Execução do Experimento
# ==========================
resultados_ft = []
N_PER_TECHNIQUE = 5

for technique, instruction in PROMPT_TEMPLATES.items():
    for i in range(1, N_PER_TECHNIQUE + 1):
        temp = random.choice([0.2, 0.5]) # Temperaturas baixas para maior rigor técnico
        print(f"Generating FT: {technique} Q{i}...")
        
        saida = generate_with_lora(instruction, temp)

        resultados_ft.append({
            "id": f"SLM_FT_{technique}_Q{i}",
            "model": "SLM_FINE_TUNED",
            "technique": technique,
            "temperature": temp,
            "generated_content": saida
        })

# ==========================
# 6) Salvar Resultados
# ==========================
with open("dataset_fine_tuned.json", "w", encoding="utf-8") as f:
    json.dump(resultados_ft, f, indent=4, ensure_ascii=False)

print("\n✅ Dataset Fine-Tuned gerado com sucesso!")