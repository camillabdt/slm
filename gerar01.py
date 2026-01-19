import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Forçamos o pad_token para evitar erros de geração
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map={"": "cpu"}, 
    torch_dtype=torch.float32
)

# Carregar o modelo com o seu treinamento
model = PeftModel.from_pretrained(base_model, adapter_path)

def responder(pergunta):
    # System Prompt IGUAL ao do seu dataset para ativar o conhecimento
    system_message = "You are an educational cybersecurity tutor for 9th-grade students. Use clear and simple language."
    
    # IMPORTANTE: Use inglês se o seu dataset estiver em inglês para testar se funcionou
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": pergunta}
    ]
    
    # O template coloca as tags <|user|> e <|assistant|> corretamente
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cpu")

    outputs = model.generate(
        input_ids, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1, # Quase zero para ele não inventar
        top_p=0.9,
        repetition_penalty=1.2, # FORÇA o modelo a não repetir a mesma frase
        eos_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# Teste com algo que está no seu dataset (Phishing ou 2FA)
print("Resposta do Tutor:")
print(responder("Explain what is phishing to a student."))