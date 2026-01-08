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

# Configurações do modelo - Certifique-se de que os caminhos estão corretos
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "tinyllama_lora_final"

class GeradorCyberHibrido:
    def __init__(self):
        print("🔧 Carregando cérebro local (SLM)...")
        # Carrega o tokenizador e o modelo base
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Carregando com precisão float32 para CPU (ou float16 se tiver GPU)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        # Acopla o seu treinamento LoRA ao modelo base
        print("🧠 Integrando conhecimento especializado (LoRA)...")
        self.model = PeftModel.from_pretrained(base, LORA_PATH)
        self.model.eval()

    def preparar_prompt(self, tema, tecnica):
        """
        Define a estratégia de Engenharia de Prompt para a SLM.
        Ajustado para Português para evitar confusão idiomática.
        """
        sys_msg = "<|system|>\nVocê é um professor de cibersegurança do 9º ano. Responda apenas com JSON.<|end|>\n"
        
        if tecnica == "zero-shot":
            user_msg = f"<|user|>\nGere uma questão sobre: {tema}.<|end|>\n"
            
        elif tecnica == "few-shot":
            user_msg = (
                "<|user|>\nExemplo: Sobre Senhas. "
                "JSON: {'pergunta': 'Qual senha é mais forte?', 'opcoes': ['123', '@#Abc12', 'nome', '111'], 'correta': 1}\n"
                f"Agora, crie uma sobre: {tema}.<|end|>\n"
            )
            
        elif tecnica == "chain-of-thought":
            user_msg = (
                f"<|user|>\nPense passo a passo: 1. Defina um risco de {tema}. 2. Explique a prevenção. "
                f"3. Gere a questão em JSON.<|end|>\n"
            )
            
        elif tecnica == "exemplar-guided":
            user_msg = (
                f"<|user|>\nSiga o estilo: 'João recebeu um SMS suspeito...'. "
                f"Crie uma questão contextualizada sobre {tema}.<|end|>\n"
            )
        
        else: # template-based
            user_msg = f"<|user|>\nPreencha o template para {tema}: {{'pergunta': '...', 'opcoes': ['...', '...', '...', '...'], 'correta': 0}}<|end|>\n"

        return f"{sys_msg}{user_msg}<|assistant|>\n"

    def criar_questao(self, tema, tecnica="few-shot"):
        prompt = self.preparar_prompt(tema, tecnica)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=250,      # Limite para evitar divagações
                do_sample=True, 
                temperature=0.1,         # Temperatura baixíssima para estabilidade
                repetition_penalty=1.3,  # Evita repetições infinitas
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        resposta_completa = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extrair apenas a parte do Assistant para o JSON
        try:
            # Tenta encontrar o último bloco JSON na resposta
            inicio_json = resposta_completa.rfind("{")
            fim_json = resposta_completa.rfind("}") + 1
            json_str = resposta_completa[inicio_json:fim_json]
            return json.loads(json_str.replace("'", "\"")) # Converte aspas simples se houver
        except:
            return {
                "erro": "Falha na estrutura JSON",
                "tecnica": tecnica,
                "resposta_bruta": resposta_completa.split("<|assistant|>")[-1].strip()
            }

# --- BLOCO DE EXPERIMENTAÇÃO ---
if __name__ == "__main__":
    gerador = GeradorCyberHibrido()
    tecnicas = ["zero-shot", "few-shot", "chain-of-thought", "exemplar-guided", "template-based"]
    
    # Tema focado no seu dataset
    tema_teste = "Segurança em Redes Sociais"
    
    print(f"\n🚀 Iniciando bateria de testes locais")
    print(f"Tema: {tema_teste}")
    print("-" * 40)

    resultados_finais = []

    for t in tecnicas:
        print(f"📡 Processando técnica: {t}...")
        questao = gerador.criar_questao(tema_teste, t)
        
        # Adiciona ao relatório
        resultados_finais.append({
            "tecnica": t,
            "resultado": questao
        })

    # Salva os resultados para sua análise de dissertação
    with open("resultado_slm_local.json", "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, indent=4, ensure_ascii=False)

    print("\n✅ Experimento finalizado! Verifique o arquivo 'resultado_slm_local.json'.")