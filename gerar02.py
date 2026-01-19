import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configurações de Carregamento
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu" 

print("Carregando modelo e conhecimentos de cibersegurança (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    torch_dtype=torch.float32
)

# Acopla o seu treinamento ao modelo base
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def gerar_questao(system_prompt, user_prompt, temp=0.7):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cpu")
    
    # Parâmetros otimizados para evitar repetições e alucinações
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.2, # Impede que ele fique repetindo palavras
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# --- Definição das Engenharias de Prompt ---

sys_msg = "You are a professional teacher creating cybersecurity questions for 9th-grade students (14 years old). Be clear and educational."

prompt_engs = {
    "1": ("Zero-Shot", "Create a multiple-choice question about the risks of Phishing."),
    
    "2": ("Few-Shot", """Example 1:
Question: What is a strong password?
Options: A) 123456 B) password C) Ab9!xP2 D) admin
Answer: C

Example 2:
Question: What is 2FA?
Options: A) A type of virus B) An extra security step C) A game D) A browser
Answer: B

Now, create Example 3 about 'Digital Footprint':"""),

    "3": ("Chain-of-Thought", "I need a question about 'Social Media Safety'. First, explain a common mistake students make online, then create a question that tests how to avoid that specific mistake."),

    "4": ("Exemplar-Guided", """Here is a high-quality exemplar from our curriculum:
'Question: Why is it risky to click links in urgent messages?
Options: 1. They use fear to trick you. 2. They are always safe. 3. They save time. 4. They are from friends.
Correct: 1. Explanation: Scammers use urgency to make you act without thinking.'

Now, create a question following this exact quality level about 'Cyberbullying'. """),

    "5": ("Template-Based", """Follow this template exactly:
Topic: [Topic]
Question: [Text]
Options: [A, B, C, D]
Correct Answer: [Letter]
Explanation: [Why it is correct]

Topic: Personal Data Privacy""")
}

# --- Menu Interativo ---
print("\n" + "="*30)
print(" MENU DE ENGENHARIA DE PROMPT ")
print("="*30)
for k, v in prompt_engs.items():
    print(f"{k}. {v[0]}")

escolha = input("\nEscolha a técnica (1-5): ")

if escolha in prompt_engs:
    nome, prompt_texto = prompt_engs[escolha]
    print(f"\n[Gerando com {nome}...]")
    
    # Para o Template-Based, usamos uma temperatura menor (0.2) para ser mais rígido
    t = 0.2 if escolha == "5" else 0.7
    
    resultado = gerar_questao(sys_msg, prompt_texto, temp=t)
    print("\n--- QUESTÃO GERADA ---")
    print(resultado)
else:
    print("Opção inválida.")