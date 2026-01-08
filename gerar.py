# import torch, json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # Mesmas configs do treino
# BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# LORA_PATH = "tinyllama_lora_final"

# class GeradorCyber:
#     def __init__(self):
#         print("Carregando cérebro da IA...")
#         self.tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
#         base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
#         self.model = PeftModel.from_pretrained(base, LORA_PATH)
#         self.model.eval()

#     def criar_questao(self, tema):
#         prompt = (
#             "<|system|>\nYou generate multiple-choice questions for 9th-grade cybersecurity classes. "
#             "Output valid JSON: stem, options (5 items), correctOption (0-4), explanation.<|end|>\n"
#             f"<|user|>\nCreate 1 question about: {tema}.<|end|>\n<|assistant|>\n"
#         )
        
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         with torch.no_grad():
#             output = self.model.generate(
#                 **inputs, max_new_tokens=300, do_sample=True, 
#                 temperature=0.7, repetition_penalty=1.2,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
        
#         resposta = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
#         # Tenta extrair apenas o JSON
#         try:
#             start = resposta.find("{")
#             end = resposta.rfind("}")
#             return json.loads(resposta[start:end+1])
#         except:
#             return f"Erro ao gerar JSON. Resposta bruta:\n{resposta}"

# # --- USO PRÁTICO ---
# if __name__ == "__main__":
#     gerador = GeradorCyber()
    
#     while True:
#         tema = input("\nSobre qual tema de Segurança Digital quer uma questão? (ou 'sair'): ")
#         if tema.lower() == 'sair': break
        
#         questao = gerador.criar_questao(tema)
#         print("\nQUESTÃO GERADA:")
#         print(json.dumps(questao, indent=4, ensure_ascii=False))


import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configurações do modelo
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "tinyllama_lora_final"

class GeradorCyberHibrido:
    def __init__(self):
        print("🔧 Carregando cérebro local (SLM)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, LORA_PATH)
        self.model.eval()

    def preparar_prompt(self, tema, tecnica):
        # Instrução de sistema fixa para manter o alinhamento pedagógico
        sys_msg = "<|system|>\nVocê é um professor de cibersegurança para o 9º ano. Gere questões em JSON.<|end|>\n"
        
        if tecnica == "zero-shot":
            user_msg = f"<|user|>\nGere uma questão de múltipla escolha sobre: {tema}.<|end|>\n"
            
        elif tecnica == "few-shot":
            user_msg = (
                "<|user|>\nExemplo: Sobre Senhas. "
                "JSON: {'stem': 'Qual senha é mais forte?', 'options': ['123', 'Abc@123', 'nome', '111', '000'], 'correctOption': 1, 'explanation': 'Mistura caracteres.'}\n"
                f"Agora, crie uma sobre: {tema}.<|end|>\n"
            )
            
        elif tecnica == "chain-of-thought":
            user_msg = (
                f"<|user|>\nPense passo a passo: 1. Defina um risco de {tema}. 2. Explique a prevenção. "
                f"3. Gere a questão final em JSON.<|end|>\n"
            )
            
        elif tecnica == "exemplar-guided":
            user_msg = (
                f"<|user|>\nUse este estilo: 'Maria recebeu um link falso...'. "
                f"Crie uma questão contextualizada sobre {tema}.<|end|>\n"
            )
        
        else: # template-based
            user_msg = f"<|user|>\nPreencha o template para {tema}: {{'stem': '...', 'options': [...], 'correctOption': 0, 'explanation': '...'}}<|end|>\n"

        return f"{sys_msg}{user_msg}<|assistant|>\n"

    def criar_questao(self, tema, tecnica="zero-shot"):
        prompt = self.preparar_prompt(tema, tecnica)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=350, 
                do_sample=True, 
                temperature=0.7, # Temperatura ajustada para equilíbrio
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        resposta = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extração do JSON
        try:
            start = resposta.find("{")
            end = resposta.rfind("}")
            return json.loads(resposta[start:end+1])
        except:
            return {"erro": "Falha na estrutura JSON", "resposta_bruta": resposta}

# --- TESTE EXPERIMENTAL ---
if __name__ == "__main__":
    gerador = GeradorCyberHibrido()
    tecnicas = ["zero-shot", "few-shot", "chain-of-thought", "exemplar-guided", "template-based"]
    
    tema = "Phishing"
    print(f"\n📊 Inciando testes locais para o tema: {tema}")
    
    for t in tecnicas:
        print(f"\n--- Testando Técnica: {t.upper()} ---")
        questao = gerador.criar_questao(tema, t)
        print(json.dumps(questao, indent=4, ensure_ascii=False))