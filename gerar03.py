import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuração de Carregamento
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu_v2" 

print("Carregando Tutor V2 com Attention Mask corrigida...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def gerar_com_rigor(system_prompt, user_prompt, temp=0.2):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Aplica o template de chat
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cpu")
    
    # SOLUÇÃO DA MENSAGEM: Criando a máscara de atenção explicitamente
    attention_mask = torch.ones(input_ids.shape, device="cpu", dtype=torch.long)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask, # Define onde o modelo deve focar
        max_new_tokens=256,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decodifica apenas a resposta nova
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# --- Prompts Otimizados ---
sys_msg = "You are an educational cybersecurity tutor. Create one high-quality question for 9th-grade students."

engenharias = {
    "1": ("Zero-Shot", "Create a multiple-choice question about Phishing risks. Include: Question, Options A-D, Correct Answer, and Explanation."),
    "2": ("Few-Shot", "### Example:\nQ: What is 2FA?\nA) Virus B) Extra security C) Game\nAns: B\n\n### Task:\nCreate a question about 'Digital Footprint'."),
    "3": ("Chain-of-Thought", "Explain step-by-step why urgent messages are dangerous in scams. Then, create a question about this."),
    "4": ("Exemplar-Guided", "Using the style: 'A password is like a key to your house'. Explain 'Two-Factor Authentication' and create a question."),
    "5": ("Template-Based", "Fill exactly:\nTopic: Privacy\nQuestion: [..]\nOptions: A) [..] B) [..] C) [..] D) [..]\nCorrect: [..]\nExplanation: [..]")
}

print("\n--- MENU DE ENGENHARIAS ---")
for k, v in engenharias.items():
    print(f"{k}. {v[0]}")

escolha = input("\nEscolha a técnica: ")

if escolha in engenharias:
    nome, prompt_texto = engenharias[escolha]
    print(f"\n[Gerando com {nome}...]\n")
    # Temperatura mais baixa para ser mais fiel ao treino
    resultado = gerar_com_rigor(sys_msg, prompt_texto, temp=0.15)
    print(resultado)
else:
    print("Opção inválida.")