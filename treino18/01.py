import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# 1. Carregar Modelo e Tokenizador (Sem quantização 4-bit, que exige GPU)
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Carregar modelo em modo CPU
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, # Força o uso da CPU
    torch_dtype=torch.float32
)

# 2. Configurar LoRA (Reduz o peso do treino na CPU)
lora_config = LoraConfig(
    r=8, # Reduzi para 8 para ser mais leve
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Carregar seu dataset
dataset = load_dataset("json", data_files="datasetoficial.jsonl", split="train")

# 4. Configurar Treino para CPU
training_args = TrainingArguments(
    output_dir="./tinyllama-cyber-cpu",
    per_device_train_batch_size=1, # Mínimo possível para não travar a RAM
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    use_cpu=True, # Garante o uso da CPU
    no_cuda=True,
    report_to="none"
)

# 5. Iniciar Treinador
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="messages",
)

print("Iniciando treino na CPU... Prepare o café, isso vai demorar.")
trainer.train()