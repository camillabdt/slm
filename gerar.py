

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


import json
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCHEMA_EXATO = '{"pergunta":"...","opcoes":["...","...","...","..."],"correta":0,"feedback":"..."}'

class GeradorCyberOtimizado:
    def __init__(self, base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="tinyllama_lora_final"):
        print("🔧 Carregando SLM otimizado...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()

    # -------------------------------
    # PROMPTS por técnica (compactos)
    # -------------------------------
    def preparar_prompt(self, tema, tecnica):
        system = (
            "<|system|>\n"
            "Você é professor de Cibersegurança.\n"
            "Responda SOMENTE em português do Brasil.\n"
            "Responda SOMENTE com JSON válido.\n"
            "Use exatamente as chaves: pergunta, opcoes, correta, feedback.\n"
            "opcoes deve ter 4 itens. correta é um inteiro 0-3.\n"
            "Não use outras chaves.\n"
            "<|end|>\n"
        )

        if tecnica == "zero-shot":
            user = (
                "<|user|>\n"
                f"Gere 1 questão objetiva sobre: {tema}.\n"
                f"Use exatamente este schema: {SCHEMA_EXATO}\n"
                "<|end|>\n"
            )

        elif tecnica == "few-shot":
            exemplo = (
                '{"pergunta":"Qual atitude reduz o risco de phishing?",'
                '"opcoes":["Clicar em links suspeitos","Verificar remetente e URL","Enviar senha por e-mail","Usar a mesma senha sempre"],'
                '"correta":1,'
                '"feedback":"Verificar remetente e URL ajuda a evitar golpes de phishing."}'
            )
            user = (
                "<|user|>\n"
                f"Exemplo: {exemplo}\n"
                f"Agora gere 1 questão sobre: {tema}, no mesmo schema: {SCHEMA_EXATO}\n"
                "<|end|>\n"
            )

        elif tecnica == "chain-of-thought":
            # pensamento interno (sem imprimir passos)
            user = (
                "<|user|>\n"
                f"Pense silenciosamente e gere 1 questão objetiva sobre: {tema}.\n"
                f"Use exatamente este schema: {SCHEMA_EXATO}\n"
                "<|end|>\n"
            )

        elif tecnica == "exemplar-guided":
            user = (
                "<|user|>\n"
                f"Crie um cenário curto e realista sobre {tema} e gere 1 questão baseada nele.\n"
                f"Use exatamente este schema: {SCHEMA_EXATO}\n"
                "<|end|>\n"
            )

        else:  # template-based
            user = (
                "<|user|>\n"
                f"Preencha corretamente este template para o tema {tema}:\n"
                f"{SCHEMA_EXATO}\n"
                "<|end|>\n"
            )

        # truque: força o início do JSON
        return f"{system}{user}<|assistant|>\n{{"

    # -------------------------------
    # Extrai e normaliza JSON
    # -------------------------------
    def _extrair_objetos_json(self, texto):
        # pega candidatos { ... } (pode haver mais de um)
        return re.findall(r"\{[\s\S]*?\}", texto)

    def _normalizar_chaves(self, obj: dict) -> dict:
        keymap = {
            "stem": "pergunta",
            "question": "pergunta",
            "prompt": "pergunta",

            "opcions": "opcoes",
            "options": "opcoes",
            "alternativas": "opcoes",

            "correctAnswer": "correta",
            "correctOption": "correta",
            "answer": "correta",

            "explanation": "feedback",
            "explicacao": "feedback",
        }

        # renomeia chaves conhecidas
        for k in list(obj.keys()):
            if k in keymap:
                obj[keymap[k]] = obj.pop(k)

        # tenta corrigir "correta" se vier letra
        if isinstance(obj.get("correta"), str):
            m = re.search(r"\b([ABCD])\b", obj["correta"].upper())
            if m:
                obj["correta"] = "ABCD".index(m.group(1))

        # às vezes vem índice 1-4
        if isinstance(obj.get("correta"), int) and obj["correta"] in [1,2,3,4] and len(obj.get("opcoes", [])) == 4:
            # só converte se parecer 1-based
            # (se for 0-based, não mexe)
            if obj["correta"] != 0:
                obj["correta"] = obj["correta"] - 1

        return obj

    def _validar_schema(self, obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False
        if not all(k in obj for k in ["pergunta", "opcoes", "correta", "feedback"]):
            return False
        if not isinstance(obj["pergunta"], str) or len(obj["pergunta"].strip()) < 8:
            return False
        if not isinstance(obj["opcoes"], list) or len(obj["opcoes"]) != 4:
            return False
        if any((not isinstance(o, str) or len(o.strip()) < 1) for o in obj["opcoes"]):
            return False
        if not isinstance(obj["correta"], int) or not (0 <= obj["correta"] <= 3):
            return False
        if not isinstance(obj["feedback"], str) or len(obj["feedback"].strip()) < 5:
            return False
        # evita placeholders
        if "..." in json.dumps(obj, ensure_ascii=False):
            return False
        return True

    def limpar_e_carregar_json(self, texto_gerado: str):
        # limpa lixo comum
        t = (texto_gerado or "").replace("```", "").replace("\t", " ")
        t = t.replace("\n", " ").strip()

        # tenta candidatos em ordem
        for cand in self._extrair_objetos_json(t):
            try:
                obj = json.loads(cand)
            except Exception:
                continue

            obj = self._normalizar_chaves(obj)
            if self._validar_schema(obj):
                return obj

        return None

    # -------------------------------
    # Geração
    # -------------------------------
    def criar_questao(self, tema, tecnica):
        prompt = self.preparar_prompt(tema, tecnica)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.2,         # mais estável
                top_p=0.85,
                repetition_penalty=1.12, # reduz repetição/eco
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # pega só o que foi gerado após o prompt
        gerado = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        resultado = self.limpar_e_carregar_json(gerado)
        if resultado:
            resultado["tecnica"] = tecnica
            resultado["tema"] = tema
            return resultado
        return {"erro": "JSON inválido", "tecnica": tecnica, "tema": tema, "bruto": gerado}


# -----------------------------------------
# EXECUÇÃO: gera N questões por técnica
# -----------------------------------------
if __name__ == "__main__":
    gerador = GeradorCyberOtimizado()

    tema = "Phishing em Redes Sociais"
    tecnicas = ["zero-shot", "few-shot", "chain-of-thought", "exemplar-guided", "template-based"]

    N_POR_TECNICA = 5
    ARQ_SAIDA = "resultados5.jsonl"

    total_ok = 0
    total = 0

    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        for t in tecnicas:
            print(f"\n==================== Técnica: {t} ====================")
            for i in range(N_POR_TECNICA):
                total += 1
                q = gerador.criar_questao(tema, t)

                if "erro" not in q:
                    total_ok += 1
                    print(f"✅ {t} #{i+1}: OK")
                else:
                    print(f"❌ {t} #{i+1}: {q['erro']}")

                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(0.2)

    print(f"\n📌 Finalizado: {total_ok}/{total} questões válidas salvas em {ARQ_SAIDA}")
