import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# 1. Função de carregamento corrigida para processar MCQs e conteúdo educativo
def load_and_fix_jsonl(file_path):
    data_rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assistant_content = row["messages"][-1]["content"]
            
            # Converte o objeto de múltipla escolha em texto legível para o modelo
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

# 2. Modelo e Tokenizer (Configurados para CPU)
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    torch_dtype=torch.float32 
)

# 3. Aplicação do LoRA (Apenas uma vez aqui)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Carregar o seu dataset de cibersegurança
dataset = load_and_fix_jsonl("datasetoficial.jsonl")

# 5. Configurar Treino para CPU
training_args = TrainingArguments(
    output_dir="./tinyllama-cyber-cpu",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    use_cpu=True, # Substitui o antigo no_cuda
    report_to="none"
)

# 6. Treinador (Sem o argumento peft_config, pois o modelo já é Peft)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("Tudo pronto! Iniciando o treinamento do seu tutor de cibersegurança...")
trainer.train()

# 7. Salvar o resultado
trainer.save_model("./modelo_final_cpu")