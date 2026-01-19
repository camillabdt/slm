import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# 1. Carregamento e correção do Dataset (essencial para não bugar a estrutura)
def load_and_fix_jsonl(file_path):
    data_rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assistant_content = row["messages"][-1]["content"]
            if isinstance(assistant_content, dict):
                mcq = assistant_content
                text = f"Question: {mcq['stem']}\n"
                for i, opt in enumerate(mcq['options']):
                    text += f"{i}) {opt}\n"
                text += f"Correct Answer: {mcq['correctOption']}\n"
                text += f"Explanation: {mcq['explanation']}"
                row["messages"][-1]["content"] = text
            data_rows.append(row)
    return Dataset.from_list(data_rows)

# 2. Configuração do Modelo
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    dtype=torch.float32 
)

# 3. LoRA mais "agressivo" (Alpha 32 e Rank 16 para maior fixação)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Adicionamos mais camadas
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_and_fix_jsonl("datasetoficial.jsonl")

# 4. Configuração de Treino Rigoroso
training_args = TrainingArguments(
    output_dir="./tinyllama-cyber-rigoroso",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4, # Taxa de aprendizado levemente menor para precisão
    num_train_epochs=5, # Aumentado de 1 para 5 (mais rigoroso)
    logging_steps=1,
    use_cpu=True,
    report_to="none",
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("Iniciando pesado agora rapaiz (5 épocas)...")
trainer.train()

trainer.save_model("./modelo_final_cpu_v2")
print("Modelo treinado e salvo com sucesso!")