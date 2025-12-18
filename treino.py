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
# 1) Configurações
# ==========================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_output_dir = "tinyllama_software_eng_lora"
num_steps = 150 
max_length = 384 # Aumentado para acomodar questões com 5 opções

# Suporte para até 6 opções
LETTERS = ["A", "B", "C", "D", "E", "F"]

# ==========================================
# 2) Carregamento de Dados
# ==========================================
if not os.path.exists("questions.json"):
    print("Erro: Arquivo questions.json não encontrado!")
    exit()

with open("questions.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ==========================================
# 3) Engenharia de Prompt (ChatML)
# ==========================================
def format_chatml(item):
    """Detecta opções dinamicamente e formata para ChatML"""
    stem = item["stem"]
    options = item["options"]
    correct_idx = item["correctOption"]
    explanation = item["explanation"]
    
    # Mapeia a letra correta baseada no índice do JSON
    correct_letter = LETTERS[correct_idx] if correct_idx < len(LETTERS) else "A"
    
    # Constrói o bloco de opções dinamicamente
    options_txt = ""
    for i, opt in enumerate(options):
        if i < len(LETTERS):
            options_txt += f"{LETTERS[i]}) {opt}\n"

    prompt = (
        f"<|system|>\nYou are a Software Engineering professor. Always create questions with a Stem, Options, Correct Answer, and Explanation.<|end|>\n"
        f"<|user|>\nGenerate a professional question about Software Engineering.<|end|>\n"
        f"<|assistant|>\nStem: {stem}\n\nOptions:\n{options_txt.strip()}\n\n"
        f"Correct Answer: {correct_letter}\nExplanation: {explanation}<|end|>"
    )
    return prompt

# ==========================================
# 4) Treino ou Carregamento
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def train_model():
    print("Iniciando treinamento na CPU...")
    # Corrigido: usando torch_dtype=torch.float32 explicitamente para CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    train_list = [{"text": format_chatml(item)} for item in raw_data]
    
    ds = Dataset.from_list(train_list).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length, padding="max_length"),
        batched=True
    )

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=num_steps,
        learning_rate=2e-4,
        use_cpu=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    return model

if not os.path.exists(lora_output_dir):
    model = train_model()
else:
    print(f"Carregando adaptador salvo de: {lora_output_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, lora_output_dir)

model.eval()

# ==========================================
# 5) Geração de Novas Questões
# ==========================================
def generate_question(topic):
    prompt = (
        f"<|system|>\nYou are a Software Engineering professor.<|end|>\n"
        f"<|user|>\nGenerate a NEW multiple-choice question about: {topic}.<|end|>\n"
        f"<|assistant|>\nStem:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=280,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extração limpa do conteúdo gerado
    try:
        response = full_text.split("assistant")[-1].strip()
        if not response.startswith("Stem:"): 
            response = "Stem: " + response
    except:
        response = full_text
        
    return response

# Exemplo de uso
topics = ["Scrum Ceremonies", "Performance Requirements", "Git Branching Strategy"]
print("\n" + "="*50)
print("QUESTÕES GERADAS PELO SLM")
print("="*50)

for t in topics:
    print(f"\n[Tópico: {t}]")
    print(generate_question(t))
    print("-" * 30)