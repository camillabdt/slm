

#GERAÇÃO COM SLM LOCAL SIMPLES  


# # import torch, json
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


<<<<<<< HEAD
#GERAÇÃO COM SLM LOCAL E VÁRIAS TÉCNICAS DE PROMPTING

import torch, json
=======
import torch, json, re
>>>>>>> 54b48eb (att)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class GeradorCyberOtimizado:
    def __init__(self, base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="tinyllama_lora_final"):
        print("🔧 Carregando SLM otimizado...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        # Carregando em float32 para estabilidade em CPU/GPU simples
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()

    def preparar_prompt(self, tema, tecnica):
        """
        Prompts simplificados para as 5 técnicas (Foco em modelos de 1.1B).
        """
        # Instrução de sistema ultra-curta
        sys = "<|system|>\nProfessor de Cibersegurança. Responda apenas em JSON e Português.<|end|>\n"
        
        if tecnica == "zero-shot":
            user = f"<|user|>\nGere uma questão sobre {tema}.<|end|>\n"
            
        elif tecnica == "few-shot":
            user = (
                f"<|user|>\nExemplo: Tema: Senhas. Resposta: {{\"pergunta\": \"O que é senha forte?\", \"opcoes\": [\"123\", \"Abc@12\", \"nome\", \"111\"], \"correta\": 1}}\n"
                f"Agora faça do Tema: {tema}.<|end|>\n"
            )
            
        elif tecnica == "chain-of-thought":
            user = f"<|user|>\nExplique o risco de {tema}, defina a prevenção e gere a questão em JSON.<|end|>\n"
            
        elif tecnica == "exemplar-guided":
            user = f"<|user|>\nCrie um cenário com um personagem sobre {tema} e gere a questão em JSON.<|end|>\n"
            
        else: # template-based
            user = f"<|user|>\nPreencha: {{\"pergunta\": \"...\", \"opcoes\": [\"A\", \"B\", \"C\", \"D\"], \"correta\": 0}} para o tema {tema}.<|end|>\n"

        # O segredo: terminar o prompt com '{' para forçar o início do JSON
        return f"{sys}{user}<|assistant|>\n{{"

    def limpar_e_carregar_json(self, texto):
        """
        Tenta recuperar o JSON mesmo que o modelo gere lixo ao redor.
        """
        try:
            # Adiciona a chave de abertura que forçamos no prompt
            texto_completo = "{" + texto 
            # Busca o bloco mais externo de chaves
            match = re.search(r"(\{.*\})", texto_completo, re.DOTALL)
            if match:
                str_json = match.group(1)
                # Remove possíveis quebras de linha ou caracteres de escape que quebram o parser
                str_json = str_json.replace("\n", " ").replace(".safe()", "")
                return json.loads(str_json)
        except:
            return None
        return None

    def criar_questao(self, tema, tecnica):
        prompt = self.preparar_prompt(tema, tecnica)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=250, 
                temperature=0.1,         # Baixíssima para evitar "alucinação idiomática"
                repetition_penalty=1.1,  # Leve para não quebrar a sintaxe JSON
                do_sample=True,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Pega apenas a parte gerada após o prompt
        gerado = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        resultado = self.limpar_e_carregar_json(gerado)
        
        if resultado:
            return resultado
        else:
            return {"erro": "Falha na estrutura", "bruto": gerado}

# --- TESTE ---
if __name__ == "__main__":
    gerador = GeradorCyberOtimizado()
    tema = "Phishing em Redes Sociais"
    tecnicas = ["zero-shot", "few-shot", "chain-of-thought", "exemplar-guided", "template-based"]

    for t in tecnicas:
        print(f"🛠️ Testando {t}...")
        q = gerador.criar_questao(tema, t)
        print(json.dumps(q, indent=2, ensure_ascii=False))