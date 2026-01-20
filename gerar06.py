import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os

# 1. Configuração do Modelo (Forçando CPU para evitar erros de memória)
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu_v2" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to("cpu")

model = PeftModel.from_pretrained(base_model, adapter_path).to("cpu")
model.eval()

# 2. Funções de Engenharia de Prompt (As mesmas técnicas anteriores)
def obter_prompt(tecnica, pilar, subtema):
    instrucao = "Create a 9th-grade MCQ. Output valid JSON: stem, options (A-D), correctOption, explanation."
    if tecnica == "zero-shot": return f"{instrucao}\nTopic: {subtema}"
    if tecnica == "few-shot": return f"Example:\nQ: What is a node?\nA) Point B) Line\nAns: A\n\nTask: {instrucao}\nTopic: {subtema}"
    if tecnica == "chain-of-thought": return f"Explain the logic of {subtema} step-by-step, then create the MCQ."
    if tecnica == "exemplar-guided": return f"Using the analogy of 'a spider web', create an MCQ about {subtema}."
    return f"Fill Template:\nTopic: {subtema}\nQuestion: [..]\nOptions: A-D\nCorrect: [..]\nExplanation: [..]"

# 3. Lista Global de Conteúdos (BNCC 9º Ano)
# Baseado nas habilidades EF09C001 a EF09C010 [cite: 948, 1246]
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
file_name = "banco_final_seguro.jsonl"

print(f"🚀 Iniciando geração. Os dados serão salvos em {file_name}")

# Limpa o ficheiro se já existir para começar do zero
if os.path.exists(file_name):
    os.remove(file_name)

# 4. Loop de Geração
count = 0
for pilar, subtema in conteudos:
    for tecnica in tecnicas:
        print(f"[{count+1}] Gerando: {subtema} com {tecnica}...")
        
        sys_msg = f"You are a CS Professor. Topic: {pilar}. Format: JSON."
        user_msg = obter_prompt(tecnica, pilar, subtema)
        
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=400, temperature=0.3, do_sample=True)
        
        resposta = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # SALVAMENTO IMEDIATO (Linha por linha)
        dados = {
            "pilar": pilar,
            "subtema": subtema,
            "tecnica": tecnica,
            "output": resposta
        }
        
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(json.dumps(dados, ensure_ascii=False) + "\n")
            f.flush() # Força a escrita no disco agora
        
        count += 1

print(f"✨ Concluído! {count} questões salvas.")