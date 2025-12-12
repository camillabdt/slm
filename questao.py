import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. Configuração (Modo CPU Leve)
# ==========================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Carregando o modelo base {model_name}... (Pode demorar uns minutos)")

# Carrega o Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Carrega o Modelo (Usando float32 para estabilidade na CPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu", 
    torch_dtype=torch.float32 
)

# ==========================================
# 2. Definição do Prompt Zero-Shot
# ==========================================
# O segredo do Zero-Shot em modelos pequenos é ser MUITO ESPECÍFICO na instrução.
# Estamos pedindo explicitamente o formato JSON-like ou estruturado.

prompt_system = (
    "You are a strict Software Engineering professor. "
    "Create a multiple-choice question about 'Requirements Engineering'."
)

prompt_instruction = (
    "Generate a question with the following format:\n"
    "Stem: [The question text]\n"
    "Options: [List of 5 options labelled A-E]\n"
    "Correct Answer: [The correct option letter]\n"
    "Explanation: [Why it is correct]\n\n"
    "Make the question about 'Non-Functional Requirements'."
)

# Monta o prompt no formato que o TinyLlama entende (Alpaca/Chat)
final_prompt = (
    f"### Instruction:\n{prompt_system} {prompt_instruction}\n\n"
    f"### Response:\n"
)

# ==========================================
# 3. Geração (Inferência)
# ==========================================
print("\n" + "="*40)
print("Gerando questão Zero-Shot (sem treino)...")
print("="*40 + "\n")

inputs = tokenizer(final_prompt, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,   # Limite para não demorar uma eternidade
        do_sample=True,       # Habilita criatividade
        temperature=0.7,      # 0.7 é um bom equilíbrio (nem muito louco, nem robótico)
        top_p=0.9,
        repetition_penalty=1.2 # Evita que ele repita frases
    )

# Decodifica e imprime
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Limpeza visual para mostrar só a resposta
response_only = generated_text.split("### Response:")[-1].strip()

print(response_only)