import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuração de carregamento para CPU
model_id = "tinyllama/tinyllama-1.1b-chat-v1.0"
adapter_path = "./modelo_final_cpu" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"}, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, adapter_path)

def testar_prompt(estilo, prompt_text):
    print(f"\n--- Testando: {estilo} ---")
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 1. ZERO-SHOT: O modelo responde sem exemplos, confiando apenas no treino.
prompt_zero = "Explique o que é Phishing e dê um exemplo de como ele acontece." 
# O modelo usará a base de que Phishing é um golpe que usa mensagens falsas[cite: 1, 3].

# 2. FEW-SHOT: Fornecemos exemplos para ditar o formato da resposta.
prompt_few = """
Usuário: O que são dados pessoais?
Assistente: São informações que identificam você, como nome e endereço[cite: 102].
Usuário: Por que devemos proteger nossa privacidade?
Assistente: Para evitar golpes e uso indevido de nossas informações[cite: 104].
Usuário: O que é pegada digital?
Assistente:""" 
# O modelo deve completar com base no conceito de rastro de informação online[cite: 132].

# 3. CHAIN-OF-THOUGHT (CoT): Estimula o modelo a raciocinar passo a passo.
prompt_cot = "Pergunta: Por que é perigoso clicar em links urgentes de desconhecidos? Pense passo a passo antes de responder."
# O modelo deve explicar que a urgência impede o pensamento crítico e leva ao erro[cite: 12, 14].

# 4. EXEMPLAR-GUIDED: Guia o modelo a seguir um padrão específico de um exemplo do seu dataset.
prompt_exemplar = "Com base no exemplo de segurança de senhas, crie uma recomendação sobre o uso de Autenticação de Dois Fatores (2FA)."
# Foca na ideia de que o 2FA é uma camada extra de proteção mesmo se a senha vazar[cite: 268, 282].

# 5. TEMPLATE-BASED: Usa uma estrutura fixa (como as MCQs que você criou).
prompt_template = """Crie uma questão de múltipla escolha seguindo este formato:
ID: [ID]
Tópico: Fake News
Dificuldade: Easy
Pergunta: [Pergunta]
Opções: [5 opções]
Resposta Correta: [Número]
Explicação: [Explicação]"""
# O modelo usará o padrão de questões que você inseriu no dataset (RIFN-MCQ)[cite: 255, 256].

# Execução
testar_prompt("Zero-Shot", prompt_zero)
testar_prompt("Few-Shot", prompt_few)
testar_prompt("Chain-of-Thought", prompt_cot)
testar_prompt("Exemplar-Guided", prompt_exemplar)
testar_prompt("Template-Based", prompt_template)