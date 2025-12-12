import os
import json
import torch
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
# 1. Configurações Iniciais
# ==========================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "tinyllama_cpu_english"
max_steps = 60  # 60 passos para ele aprender a estrutura (deve levar 20-40 min na CPU)

# ==========================================
# 2. Garantir que existam dados (Prevenção de Erro)
# ==========================================
# Se você não tiver o arquivo questions.json na pasta, criamos um dummy agora
if not os.path.exists('questions.json'):
    print("Arquivo 'questions.json' não encontrado. Criando dados de exemplo...")
    dummy_data = [
        {
            "stem": "What characterizes a \"performance requirement\"?",
            "options": ["Security aspect", "Speed and latency", "User interface", "Database schema", "Legal compliance"],
            "correctOption": 1,
            "explanation": "Performance requirements define how fast the system performs."
        },
        {
            "stem": "Which of these is a Functional Requirement?",
            "options": ["The system shall run on iOS.", "The system shall calculate tax.", "The system shall be available 99% of time.", "The code must be written in Python.", "The app must use blue colors."],
            "correctOption": 1,
            "explanation": "Functional requirements describe behaviors or functions."
        }
    ]
    with open('questions.json', 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f)

# ==========================================
# 3. Carregar Tokenizer e Modelo (Modo CPU)
# ==========================================
print(f"Carregando modelo {model_name}... (Isso usa sua RAM)")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Correção essencial para o Llama não dar erro de padding
tokenizer.pad_token = tokenizer.eos_token 

# Carregamento otimizado para CPU (float32 é mais rápido em CPUs antigas que float16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu", 
    torch_dtype=torch.float32 
)

# ==========================================
# 4. Configurar LoRA (Economia de Memória)
# ==========================================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # Rank 8 é leve e suficiente para aprender formatos
    lora_alpha=16, 
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"] # Foca apenas na "atenção" do modelo
)

model = get_peft_model(model, peft_config)
print("Configuração LoRA aplicada.")

# ==========================================
# 5. Preparação e Formatação do Dataset (INGLÊS)
# ==========================================
with open('questions.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def format_data_english(data):
    formatted_list = []
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    
    for item in data:
        # Criar a lista de opções (A, B, C...)
        options_text = ""
        for idx, option in enumerate(item['options']):
            letter = idx_to_letter.get(idx, "?")
            options_text += f"{letter}) {option}\n"
        
        # Identificar a letra correta
        correct_idx = item['correctOption']
        correct_letter = idx_to_letter.get(correct_idx, "X")
        
        # O PROMPT MÁGICO EM INGLÊS
        text = (
            f"### Instruction:\nGenerate a multiple-choice question about Software Engineering based on the dataset style.\n\n"
            f"### Response:\n"
            f"Stem: {item['stem']}\n\n"
            f"Options:\n{options_text}\n"
            f"Correct Answer: {correct_letter}\n"
            f"Explanation: {item['explanation']}{tokenizer.eos_token}"
        )
        formatted_list.append({"text": text})
    return formatted_list

# Converter lista para Dataset do HuggingFace
dataset = Dataset.from_list(format_data_english(raw_data))

# Tokenizar os textos
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256 # Mantido curto para velocidade na CPU
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ==========================================
# 6. Configuração do Treinamento
# ==========================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, # 1 exemplo por vez (Obrigatório para CPU fraca)
    gradient_accumulation_steps=4, # Acumula para simular um batch maior
    warmup_steps=5,
    max_steps=max_steps,           # 60 Passos
    learning_rate=2e-4,            # Taxa de aprendizado padrão para LoRA
    logging_steps=5,
    use_cpu=True,                  # Força CPU
    save_strategy="no",            # Não salva checkpoints intermediários (economiza disco)
    report_to="none"               # Desabilita logs externos (wandb etc)
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ==========================================
# 7. Executar Treino
# ==========================================
print("\n" + "="*30)
print(f"Iniciando treinamento na CPU por {max_steps} passos...")
print("Pode pegar um café, isso vai demorar um pouco.")
print("="*30 + "\n")

trainer.train()

# ==========================================
# 8. Teste de Inferência (Verificar resultado)
# ==========================================
print("\n" + "="*30)
print("Treino concluído! Gerando uma questão de teste...")
print("="*30 + "\n")

model.eval()

# Prompt de teste: Note que forçamos o início com "Stem:"
input_text = "### Instruction:\nGenerate a multiple-choice question about Non-Functional Requirements.\n\n### Response:\nStem:"

inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, # Espaço suficiente para gerar opções e explicação
        repetition_penalty=1.2 # Evita que ele fique repetindo frases
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)