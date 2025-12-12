import torch
import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ==========================================
# 1. Configurações
# ==========================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "tinyllama_cpu_final"

# ==========================================
# 2. Carregar Tokenizer e Modelo (Modo CPU)
# ==========================================
print("Carregando modelo... (isso pode demorar um pouco na CPU)")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Correção para modelos Llama

# Carregamos o modelo normalmente (float32 é o padrão seguro para CPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu", # Força o uso da CPU
    torch_dtype=torch.float32 # CPUs trabalham melhor com precisão total
)

# ==========================================
# 3. Configurar LoRA (Peft)
# ==========================================
# O LoRA é essencial aqui para não estourar sua memória RAM
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # Rank menor para ser mais leve
    lora_alpha=16, 
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"] # Treinar apenas Atenção (mais leve)
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Mostra quantos % do modelo serão treinados

# ==========================================
# 4. Preparação do Dataset
# ==========================================
# (Mesma lógica do seu arquivo JSON)
try:
    with open('questions.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    # Dados fake apenas para o código rodar se você não tiver o arquivo agora
    raw_data = [{
        "stem": "O que é Requisito Funcional?", 
        "options": ["Cor", "Comportamento", "Preço", "Marca"], 
        "correctOption": 1, 
        "explanation": "Define o que o sistema faz."
    }]

def format_data(data):
    formatted_list = []
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    
    for item in data:
        options_text = ""
        for idx, option in enumerate(item['options']):
            letter = idx_to_letter.get(idx, "?")
            options_text += f"{letter}) {option}\n"
        
        correct_idx = item['correctOption']
        correct_letter = idx_to_letter.get(correct_idx, "X")
        
        text = (
            f"### Instrução:\nCrie uma questão sobre Engenharia de Requisitos.\n\n"
            f"### Resposta:\n{item['stem']}\n\nAlternativas:\n{options_text}\n"
            f"Resposta Correta: {correct_letter}\nExplicação: {item['explanation']}{tokenizer.eos_token}"
        )
        formatted_list.append({"text": text})
    return formatted_list

dataset = Dataset.from_list(format_data(raw_data))

# Tokenização
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256 # Reduzi para 256 para ser mais rápido na CPU
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ==========================================
# 5. Configuração do Treinamento (Trainer)
# ==========================================
trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets,
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # 1 por vez é lei em CPU
        gradient_accumulation_steps=4, # Compensa o batch size pequeno
        warmup_steps=2,
        max_steps=10, # BEM BAIXO para você testar primeiro. Se funcionar, aumente.
        learning_rate=2e-4,
        logging_steps=1,
        use_cpu=True, # Garante que não procure GPU
        save_strategy="no", # Evita salvar checkpoints pesados toda hora
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ==========================================
# 6. Rodar o Treino
# ==========================================
print("Iniciando treinamento na CPU... Paciência é uma virtude!")
trainer.train()

# ==========================================
# 7. Teste Rápido (Inferência)
# ==========================================
print("Gerando teste...")
model.eval()
input_text = "### Instrução:\nCrie uma questão sobre Requisitos Não Funcionais.\n\n### Resposta:\n"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))