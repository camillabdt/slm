from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configurações Iniciais
# ==========================================
max_seq_length = 2048 # TinyLlama suporta até 2048
dtype = None # None para detecção automática (Float16 ou Bfloat16)
load_in_4bit = True # Otimização chave para economizar memória (QLoRA)

# 2. Carregar o Modelo TinyLlama e o Tokenizer
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/TinyLlama-1.1B-Chat-v1.0", # Versão otimizada
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Configurar os Adaptadores LoRA (Otimização)
# ==========================================
# Isso permite treinar apenas 1-10% dos parâmetros, tornando o processo leve
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: 16 é um bom equilíbrio para SLMs. Se tiver muito dado, tente 32.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # 0 é otimizado para velocidade
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Otimização de memória VRAM
    random_state = 3407,
)

# 4. Preparação do Dataset (Formatação)
# ==========================================
# Template no estilo Alpaca para instruir o modelo
import json
from datasets import Dataset

# 1. Carregar seu arquivo JSON local
# Certifique-se de fazer o upload do 'questions.json' para o Colab ou pasta local
with open('questions.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 2. Função para transformar seu JSON no formato de treino (Alpaca)
def format_data_to_alpaca(data):
    formatted_list = []
    
    # Mapeamento de índice para letra
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    for item in data:
        # Criar a string das alternativas
        options_text = ""
        for idx, option in enumerate(item['options']):
            letter = idx_to_letter.get(idx, "?")
            options_text += f"{letter}) {option}\n"
        
        # Descobrir a letra da resposta correta
        correct_idx = item['correctOption']
        correct_letter = idx_to_letter.get(correct_idx, "Desconhecida")
        
        # Construir a "Resposta ideal" (Output) que o modelo deve aprender a gerar
        final_output = (
            f"{item['stem']}\n\n"
            f"Alternativas:\n{options_text}\n"
            f"Resposta Correta: {correct_letter}\n"
            f"Explicação: {item['explanation']}"
        )

        # Adicionar à lista no formato instruction/input/output
        formatted_list.append({
            "instruction": "Gere uma questão de múltipla escolha sobre Engenharia de Requisitos baseada em conceitos fundamentais.",
            "input": "", # Sem input extra, queremos que ele gere do zero
            "output": final_output
        })
    
    return formatted_list

# 3. Converter para Dataset do HuggingFace
processed_data = format_data_to_alpaca(raw_data)
dataset = Dataset.from_list(processed_data)

# 4. Aplicar o Template do Alpaca (Prompt Engineering)
alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Resposta:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Aqui juntamos o prompt + a resposta desejada + o token de fim
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# Agora 'dataset' está pronto para entrar no 'SFTTrainer'
print(f"Dataset pronto com {len(dataset)} exemplos.")
print("Exemplo de como o modelo vê os dados:")
print(dataset[0]['text'])

# 5. Configuração do Treinador (Trainer)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Pode colocar True para treinar mais rápido se tiver muitas sequências curtas
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Aumente isso para treinar de verdade (ex: num_epochs * len(dataset) / batch)
        # Se quiser usar épocas: num_train_epochs = 1,
        learning_rate = 2e-4, # Taxa de aprendizado mais agressiva para QLoRA
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # Otimizador de 8-bit para economizar memória
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Iniciar o Treino
# ==========================================
trainer_stats = trainer.train()

# 7. Testando o Modelo (Inferência)
# ==========================================
FastLanguageModel.for_inference(model) # Habilita inferência nativa 2x mais rápida

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Crie uma questão sobre Requisitos Não Funcionais.", # Instrução
        "Foco: Segurança e Autenticação em app bancário.", # Entrada (Contexto)
        "", # Saída (deixe em branco para o modelo gerar)
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])

# 8. Salvar o Modelo
# ==========================================
# model.save_pretrained("tinyllama_requisitos_lora") # Salva apenas os adaptadores (leve)
# tokenizer.save_pretrained("tinyllama_requisitos_lora")