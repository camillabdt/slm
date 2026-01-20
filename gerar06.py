import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import random # Importante para variar a geração

# 1. Configuração do Modelo (Mantido original)
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu_v2" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cpu")
model = PeftModel.from_pretrained(base_model, adapter_path).to("cpu")
model.eval()

# 2. Funções de Engenharia de Prompt (Melhoradas para variação)
def obter_prompt(tecnica, pilar, subtema):
    # Adicionamos uma instrução de "variedade" para evitar repetição
    instrucao = "Create a UNIQUE 9th-grade MCQ. Output valid JSON: stem, options (A-D), correctOption, explanation."
    
    prompts = {
        "zero-shot": f"{instrucao}\nTopic: {subtema}",
        "few-shot": f"Example:\nQ: What is a node?\nA) Point B) Line\nAns: A\n\nTask: {instrucao}\nTopic: {subtema}",
        "chain-of-thought": f"Think step-by-step about the logic of {subtema}, then create a complex MCQ.",
        "exemplar-guided": f"Using a creative analogy related to daily life, create an MCQ about {subtema}.",
        "template-based": f"Fill Template: Topic: {subtema}\nQuestion: [..]\nOptions: A-D\nCorrect: [..]\nExplanation: [..]"
    }
    return prompts[tecnica]

# 3. Lista de Conteúdos
conteudos = [
    ("Pensamento Computacional", "Weighted Graphs"),
    ("Pensamento Computacional", "Recursion"),
    ("Pensamento Computacional", "State Machines"),
    ("Mundo Digital", "Packet Switching"),
    ("Mundo Digital", "Asymmetric Encryption"),
    ("Mundo Digital", "Malware Types"),
    ("Cultura Digital", "Digital Divide"),
    ("Cultura Digital", "LGPD Privacy"),
    ("Cultura Digital", "AI Ethics")
]

tecnicas = ["zero-shot", "few-shot", "chain-of-thought", "exemplar-guided", "template-based"]
file_name = "QuestoesSLM.jsonl"
META_QUESTOES = 100 # <--- DEFINE AQUI A QUANTIDADE

if os.path.exists(file_name):
    os.remove(file_name)

# 4. Loop de Geração Robusto
count = 0
print(f"🚀 Iniciando geração de {META_QUESTOES} questões...")

while count < META_QUESTOES:
    # Escolhe um tema e uma técnica aleatória para cada iteração
    pilar, subtema = random.choice(conteudos)
    tecnica = random.choice(tecnicas)
    
    print(f"[{count+1}/{META_QUESTOES}] Gerando: {subtema} ({tecnica})...")
    
    sys_msg = f"You are a CS Professor. Focus: {pilar}. Be creative and vary the scenarios."
    user_msg = obter_prompt(tecnica, pilar, subtema)
    
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        # Aumentamos um pouco a temperatura (0.7) para gerar questões diferentes
        outputs = model.generate(input_ids, max_new_tokens=400, temperature=0.7, do_sample=True)
    
    resposta = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    
    dados = {
        "id": count + 1,
        "pilar": pilar,
        "subtema": subtema,
        "tecnica": tecnica,
        "output": resposta
    }
    
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(json.dumps(dados, ensure_ascii=False) + "\n")
        f.flush()
    
    count += 1

print(f"✨ Concluído! {count} questões salvas em {file_name}")