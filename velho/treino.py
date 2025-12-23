import os, json, re, random
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

# ==========================================
# 1) Configurações Gerais
# ==========================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR = "tinyllama_lora_eng_soft"
DATASET_FILE = "questions.json"
MAX_STEPS = 600      # Ideal para ~200 questões em CPU
MAX_LENGTH = 384     # Espaço para questões longas
LETTERS = ["A", "B", "C", "D", "E", "F"]

# ==========================================
# 2) Engenharia de Prompt (ChatML)
# ==========================================
def format_instruction(item):
    """Transforma JSON no formato que o TinyLlama espera ver"""
    options_text = ""
    for i, opt in enumerate(item["options"]):
        if i < len(LETTERS):
            options_text += f"{LETTERS[i]}) {opt}\n"
    
    correct_letter = LETTERS[item["correctOption"]]
    
    return (
        f"<|system|>\nYou are a Software Engineering Professor. Create a professional multiple-choice question following the exact format: Stem, Options, Correct Answer, and Explanation.<|end|>\n"
        f"<|user|>\nGenerate a question about Software Engineering.<|end|>\n"
        f"<|assistant|>\n"
        f"Stem: {item['stem']}\n\n"
        f"Options:\n{options_text.strip()}\n\n"
        f"Correct Answer: {correct_letter}\n"
        f"Explanation: {item['explanation']}<|end|>"
    )

# ==========================================
# 3) Funções de Treino e Carga
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def train():
    print(f"Lendo {DATASET_FILE}...")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Criar Dataset
    formatted_data = [{"text": format_instruction(item)} for item in data]
    ds = Dataset.from_list(formatted_data).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"),
        batched=True
    )

    print("Carregando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,           # Rank aumentado para captar melhor os detalhes de 200 questões
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Mais camadas para melhor aprendizado
    )
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=MAX_STEPS,
        learning_rate=1e-4,
        use_cpu=True,
        logging_steps=20,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print(f"Iniciando treino ({MAX_STEPS} passos)...")
    trainer.train()
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print("Treino concluído e salvo!")
    return model

# Lógica de Inicialização
if not os.path.exists(LORA_DIR):
    model = train()
else:
    print("Carregando modelo já treinado...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, LORA_DIR)

model.eval()

# ==========================================
# 4) Geração de Questões
# ==========================================
def generate(topic):
    prompt = (
        f"<|system|>\nYou are a Software Engineering Professor.<|end|>\n"
        f"<|user|>\nGenerate a NEW multiple-choice question about: {topic}.<|end|>\n"
        f"<|assistant|>\nStem:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,      # Menos "viagem", mais técnica
            top_p=0.9,
            top_k=40,            # Limita vocabulário aleatório
            repetition_penalty=1.3,
            no_repeat_ngram_size=3, # Evita frases repetitivas
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Limpeza para pegar só a resposta
    if "assistant" in text:
        final_output = text.split("assistant")[-1].strip()
    else:
        final_output = text
        
    return "Stem: " + final_output if not final_output.startswith("Stem:") else final_output

# ==========================================
# 5) Teste de Saída
# ==========================================
test_topics = ["SOLID Principles", "Scrum Events", "Black-box Testing"]
for t in test_topics:
    print(f"\n--- GERANDO: {t} ---")
    print(generate(t))