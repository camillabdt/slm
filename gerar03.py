import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Carregamento do Modelo V2
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu_v2" 

print("Carregando Tutor de Cibersegurança V2 ...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, adapter_path)

def gerar(system_prompt, user_prompt, temp=0.3, max_tokens=200):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cpu")
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# --- Configuração das Engenharias ---
sys_msg = "You are an educational cybersecurity tutor for 9th-grade students. Use clear and simple language, short explanations, and focus on safe behavior."

prompts = {
    "1": ("Zero-Shot", "Create a multiple-choice question about the risks of sharing location online."),
    "2": ("Few-Shot", "User: What is phishing?\nAssistant: Phishing is a scam where someone pretends to be trusted to steal info.\nUser: What is a digital footprint?\nAssistant:"),
    "3": ("Chain-of-Thought", "Explain why urgent messages are a red flag for scams. Think step-by-step before giving a final answer."),
    "4": ("Exemplar-Guided", "Follow this style: 'Phishing is like a digital fishing hook.' Now, explain 'Two-Factor Authentication' using a simple analogy."),
    "5": ("Template-Based", "Topic: Passwords\nQuestion: [Insert]\nOptions: [A,B,C,D]\nCorrect: [Answer]\nExplanation: [Why]")
}

print("\n--- TESTE DE ENGENHARIAS V2 ---")
for k, v in prompts.items():
    print(f"\n[{v[0]}]")
    # Temperatura baixa para templates e fatos, alta para analogias
    t = 0.1 if k in ["1", "5"] else 0.5
    print(gerar(sys_msg, v[1], temp=t))