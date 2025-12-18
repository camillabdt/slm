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
num_steps = 150 # Aumentado levemente para melhor fixação do formato
max_length = 320

NUM_OPTIONS = 4
LETTERS = ["A", "B", "C", "D"]

# Tópicos para geração
topics = [
    "Non-Functional Requirements", "Functional Requirements",
    "Software Testing", "Agile Scrum", "SOLID principles",
    "Design Patterns", "CI/CD basics", "Clean Code"
]

# ==========================================
# 2) Dados de Exemplo
# ==========================================
if not os.path.exists("questions.json"):
    dummy_data = [
        {
            "stem": "What is the primary focus of a Non-Functional Requirement?",
            "options": ["System behavior", "Specific features", "Quality attributes like scalability", "Database tables"],
            "correctOption": 2,
            "explanation": "Non-functional requirements describe how the system works (performance, security) rather than what it does."
        },
        {
            "stem": "Which SOLID principle states that a class should have only one reason to change?",
            "options": ["Open/Closed", "Single Responsibility", "Liskov Substitution", "Dependency Inversion"],
            "correctOption": 1,
            "explanation": "The Single Responsibility Principle (SRP) defines that a class must focus on a single functionality."
        }
    ]
    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, ensure_ascii=False, indent=2)

with open("questions.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ==========================================
# 3) Funções de Formatação (Engenharia de Prompt)
# ==========================================
def format_chatml(stem, options, correct_letter, explanation):
    """Formata os dados para o treino no padrão ChatML do TinyLlama"""
    options_txt = "\n".join([f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options)])
    
    prompt = (
        f"<|system|>\nYou are a Software Engineering professor. Always respond with a question in the format: "
        f"Stem, Options, Correct Answer, and Explanation.<|end|>\n"
        f"<|user|>\nGenerate a question about Software Engineering.<|end|>\n"
        f"<|assistant|>\nStem: {stem}\n\nOptions:\n{options_txt}\n\n"
        f"Correct Answer: {correct_letter}\nExplanation: {explanation}<|end|>"
    )
    return prompt

# ==========================================
# 4) Preparação do Modelo e Tokenizer
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def train_model():
    print("Iniciando treinamento...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # Preparar Dataset
    train_list = []
    for item in raw_data:
        correct_letter = LETTERS[item["correctOption"]]
        text = format_chatml(item["stem"], item["options"], correct_letter, item["explanation"])
        train_list.append({"text": text})
    
    ds = Dataset.from_list(train_list).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length, padding="max_length"),
        batched=True
    )

    args = TrainingArguments(
        output_dir="./tmp_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=num_steps,
        learning_rate=2e-4,
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
    
    trainer.train()
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"Modelo salvo em {lora_output_dir}")
    return model

# ==========================================
# 5) Lógica de Carregamento e Geração
# ==========================================
if not os.path.exists(lora_output_dir):
    model = train_model()
else:
    print("Carregando modelo treinado (LoRA)...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base_model, lora_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(lora_output_dir)

model.eval()

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
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extrair apenas a parte do assistente
    response = full_text.split("assistant")[-1].strip()
    if not response.startswith("Stem:"): response = "Stem: " + response
    return response

# ==========================================
# 6) Execução Final
# ==========================================
print("\n--- GERANDO QUESTÕES ---")
for i in range(3):
    topic = random.choice(topics)
    print(f"\n[Questão {i+1} - Tópico: {topic}]")
    question_text = generate_question(topic)
    print(question_text)
    print("-" * 40)