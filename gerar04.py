import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu_v2" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"}, dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, adapter_path)

def gerar(system_prompt, user_prompt, temp=0.2):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cpu")
    attention_mask = torch.ones(input_ids.shape, device="cpu", dtype=torch.long)
    
    outputs = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=256,
        do_sample=True, temperature=temp, repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# CONTEXTO ÚNICO: Mensagens urgentes e Phishing
sys_msg = "You are a cybersecurity teacher for 9th-grade students."
tema_unificado = "the risk of urgent messages and phishing links"

engenharias = {
    "1": ("Zero-Shot", f"Create an MCQ question about {tema_unificado}. Include options A-D and explanation."),
    
    "2": ("Few-Shot", f"""Example: Q: What is a scam? A: A trick to get money.
Task: Create an MCQ about {tema_unificado}."""),

    "3": ("Chain-of-Thought", f"""Step 1: Explain why scammers use urgency. 
Step 2: Explain what happens if you click a link.
Step 3: Create an MCQ about {tema_unificado} based on these steps."""),

    "4": ("Exemplar-Guided", f"""Follow this style: 'A phishing link is like a trapdoor.' 
Create a question about {tema_unificado} using a similar educational style."""),

    "5": ("Template-Based", f"""Fill exactly:
Topic: Phishing
Question: [About {tema_unificado}]
Options: A) [..] B) [..] C) [..] D) [..]
Correct: [..]
Explanation: [..]""")
}

print("\n--- TESTE DE COMPARAÇÃO (MESMO CONTEXTO) ---")
escolha = input("Escolha (1-5): ")
if escolha in engenharias:
    print(f"\nResultado ({engenharias[escolha][0]}):")
    print(gerar(sys_msg, engenharias[escolha][1]))