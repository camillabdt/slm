import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configurações de ambiente
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
# O Chat Template define como o modelo separa System, User e Assistant
tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '</s>' + '\n'}}{% endfor %}"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, adapter_path)

def gerar_resposta_com_template(role_system, user_query):
    # Organiza a conversa no formato que o modelo foi treinado
    messages = [
        {"role": "system", "content": role_system},
        {"role": "user", "content": user_query}
    ]
    
    # Converte para os tokens especiais <|user|>, etc.
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cpu")

    outputs = model.generate(
        input_ids, 
        max_new_tokens=200, 
        temperature=0.3, # Baixa temperatura evita que ele fale do Boris Johnson
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decodifica apenas a parte que o modelo respondeu
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# --- Executando os Testes ---

system_prompt = "You are an educational cybersecurity tutor for 9th-grade students. Use clear language."

print("\n--- Zero-Shot (Cibersegurança) ---")
print(gerar_resposta_com_template(system_prompt, "What is phishing?"))

print("\n--- Template-Based (Gerando Questão MCQ) ---")
template = "Create a multiple-choice question about digital safety. Format: Question, Options, Correct Answer, Explanation."
print(gerar_resposta_com_template(system_prompt, template))