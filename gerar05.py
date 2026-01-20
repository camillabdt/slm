import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuração de Carregamento (Modelo treinado com BNCC)
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu_v2" 

print("Carregando Tutor BNCC 9º Ano Otimizado...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def gerar_especialista(system_prompt, user_prompt, temp=0.15):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cpu")
    
    # Máscara de Atenção para evitar o aviso e melhorar a precisão
    attention_mask = torch.ones(input_ids.shape, device="cpu", dtype=torch.long)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=350,
        do_sample=True,
        temperature=temp,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# --- Definições de Engenharia de Prompt (Foco em BNCC 9º Ano) ---
sys_msg = "You are a Computer Science Professor. Generate high-quality educational content for 9th-grade students based on the BNCC standards."

engenharias = {
    "1": ("Zero-Shot (Grafos)", 
          "Create a multiple-choice question about using Graphs to represent social network connections. Include Question, Options A-D, Correct Answer, and a formal Explanation."),
    
    "2": ("Few-Shot (Autómatos)", 
          "### Example:\nQ: What is an event?\nA) A mouse click B) A virus C) A screen\nAns: A\n\n### Task:\nCreate an MCQ about State Transition Diagrams in programming."),

    "3": ("Chain-of-Thought (Criptografia)", 
          "Step 1: Explain the difference between public and private keys. Step 2: Explain how they secure a message. Step 3: Create an MCQ about Asymmetric Encryption based on these steps."),

    "4": ("Exemplar-Guided (Recursão)", 
          "Follow this analogy: 'Recursion is like looking into two mirrors facing each other.' Create a question about the importance of the 'base case' in recursive functions."),

    "5": ("Template-Based (Infraestrutura)", 
          "Fill exactly:\nTopic: Packet Switching\nQuestion: [Explain how data travels in packets]\nOptions: A) [..] B) [..] C) [..] D) [..]\nCorrect: [Letter]\nExplanation: [Why packets are reconstructed at the destination]")
}

# --- Menu Interativo ---
print("\n--- TESTADOR DE ENGENHARIAS BNCC (V4) ---")
for k, v in engenharias.items():
    print(f"{k}. {v[0]}")

escolha = input("\nEscolha a técnica para testar o modelo: ")

if escolha in engenharias:
    nome, prompt_texto = engenharias[escolha]
    print(f"\n[Gerando com {nome}...]\n")
    # Temperatura bem baixa para manter o rigor técnico do 9º ano
    print(gerar_especialista(sys_msg, prompt_texto, temp=0.1))
else:
    print("Opção inválida.")