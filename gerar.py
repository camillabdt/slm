

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

SCHEMA = """{
  "pergunta": "",
  "opcoes": ["", "", "", ""],
  "correta": 0,
  "feedback": ""
}"""

class GeradorCyberOtimizado:
    def __init__(self, base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="tinyllama_lora_final"):
        print("🔧 Carregando SLM otimizado...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()

    def preparar_prompt(self, tema, tecnica):
        system = (
            "<|system|>\n"
            "Você é um professor de Cibersegurança.\n"
            "Gere APENAS um objeto JSON válido.\n"
            "Não escreva explicações fora do JSON.\n"
            "<|end|>\n"
        )

        if tecnica == "zero-shot":
            user = f"""
Gere UMA questão objetiva sobre {tema}.
Use exatamente este formato:
{SCHEMA}
Pense silenciosamente antes de responder.
"""

        elif tecnica == "few-shot":
            user = f"""
Exemplo:
{{
  "pergunta": "O que caracteriza uma senha forte?",
  "opcoes": ["123456", "senha", "Abc@1234", "nome123"],
  "correta": 2,
  "feedback": "Senhas fortes combinam letras, números e símbolos."
}}

Agora gere UMA questão sobre {tema}, no mesmo formato.
"""

        elif tecnica == "chain-of-thought":
            user = f"""
Pense silenciosamente sobre riscos e prevenção relacionados a {tema}.
Depois gere UMA questão objetiva seguindo este formato:
{SCHEMA}
"""

        elif tecnica == "exemplar-guided":
            user = f"""
Crie um pequeno cenário realista envolvendo {tema}.
Depois gere UMA questão objetiva baseada nesse cenário.
Use exatamente este formato:
{SCHEMA}
"""

        else:  # template-based
            user = f"""
Preencha corretamente o seguinte template para o tema {tema}:
{SCHEMA}
"""

        return f"{system}<|user|>{user}<|end|>\n<|assistant|>\n{{"

    def limpar_e_carregar_json(self, texto):
        try:
            texto = "{" + texto
            match = re.search(r"\{.*\}", texto, re.DOTALL)
            if not match:
                return None
            candidato = match.group(0)
            candidato = candidato.replace("\n", " ").replace("\t", " ")
            return json.loads(candidato)
        except Exception:
            return None

    def criar_questao(self, tema, tecnica):
        prompt = self.preparar_prompt(tema, tecnica)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.05,
                repetition_penalty=1.1,
                do_sample=True,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )

        gerado = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        resultado = self.limpar_e_carregar_json(gerado)
        if resultado:
            resultado["tecnica"] = tecnica
            return resultado
        else:
            return {"erro": "JSON inválido", "tecnica": tecnica, "bruto": gerado}


# ==========================
# TESTE COMPARATIVO
# ==========================
if __name__ == "__main__":
    gerador = GeradorCyberOtimizado()
    tema = "Phishing em Redes Sociais"
    tecnicas = ["zero-shot", "few-shot", "chain-of-thought", "exemplar-guided", "template-based"]

    for t in tecnicas:
        print(f"\n🛠️ Técnica: {t}")
        q = gerador.criar_questao(tema, t)
        print(json.dumps(q, indent=2, ensure_ascii=False))
