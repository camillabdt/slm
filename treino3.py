import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# 1. Carregamento Robusto do Dataset
def load_and_fix_jsonl(file_path):
    data_rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                messages = row.get("messages", [])
                
                if not messages:
                    continue
                
                last_msg = messages[-1]
                # Verifica se a chave 'content' existe
                if "content" not in last_msg:
                    # Se não houver 'content', mas for um MCQ direto no assistant
                    # Tentamos reconstruir o texto a partir das chaves do objeto
                    content_obj = last_msg
                    if isinstance(content_obj, dict) and "stem" in content_obj:
                        text = f"Question: {content_obj['stem']}\n"
                        for i, opt in enumerate(content_obj.get('options', [])):
                            text += f"{i}) {opt}\n"
                        text += f"Correct Answer: {content_obj.get('correctOption')}\n"
                        text += f"Explanation: {content_obj.get('explanation')}"
                        row["messages"][-1] = {"role": "assistant", "content": text}
                    else:
                        print(f"⚠️ Pulando linha {line_num}: Formato de mensagem inválido.")
                        continue
                
                # Se o content for um dicionário (comum em gerações sintéticas)
                elif isinstance(last_msg["content"], dict):
                    mcq = last_msg["content"]
                    text = f"Question: {mcq.get('stem', '')}\n"
                    for i, opt in enumerate(mcq.get('options', [])):
                        text += f"{i}) {opt}\n"
                    text += f"Correct Answer: {mcq.get('correctOption', '')}\n"
                    text += f"Explanation: {mcq.get('explanation', '')}"
                    row["messages"][-1]["content"] = text
                
                data_rows.append(row)
            except Exception as e:
                print(f"❌ Erro na linha {line_num}: {e}")
                
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

# 3. Configuração LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Carrega o dataset com a nova função
print("Carregando e validando dataset...")
dataset = load_and_fix_jsonl("datasetoficial.jsonl")
print(f"Dataset carregado com {len(dataset)} exemplos válidos.")

# 4. Configuração de Treino
training_args = TrainingArguments(
    output_dir="./tinyllama-cyber-rigoroso",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=5,
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

print("Iniciando TREINO RIGOROSO...")
trainer.train()

trainer.save_model("./modelo_final_cpu_v2")
print("Modelo treinado e salvo com sucesso!")