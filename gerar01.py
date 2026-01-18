import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# 1. Função de carregamento corrigida (mantendo sua estrutura original)
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

# 2. Modelo e Tokenizer
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '</s>' + '\n'}}{% endfor %}"

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    dtype=torch.float32 # Corrigido de torch_dtype
)

# 3. LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Dataset
dataset = load_and_fix_jsonl("datasetoficial.jsonl")

# 5. Configurar Treino
training_args = TrainingArguments(
    output_dir="./tinyllama-cyber-cpu",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    use_cpu=True, # Corrigido para remover aviso de no_cuda
    report_to="none"
)

# 6. Treinador (Sem o argumento problemático)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    # O SFTTrainer detecta a coluna 'messages' automaticamente se não passarmos dataset_text_field
)

print("Agora vai! Iniciando treino na CPU...")
trainer.train()