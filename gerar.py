import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Mesmas configs do treino
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "tinyllama_lora_final"

class GeradorCyber:
    def __init__(self):
        print("Carregando cérebro da IA...")
        self.tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, LORA_PATH)
        self.model.eval()

    def criar_questao(self, tema):
        prompt = (
            "<|system|>\nYou generate multiple-choice questions for 9th-grade cybersecurity classes. "
            "Output valid JSON: stem, options (5 items), correctOption (0-4), explanation.<|end|>\n"
            f"<|user|>\nCreate 1 question about: {tema}.<|end|>\n<|assistant|>\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=True, 
                temperature=0.7, repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        resposta = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Tenta extrair apenas o JSON
        try:
            start = resposta.find("{")
            end = resposta.rfind("}")
            return json.loads(resposta[start:end+1])
        except:
            return f"Erro ao gerar JSON. Resposta bruta:\n{resposta}"

# --- USO PRÁTICO ---
if __name__ == "__main__":
    gerador = GeradorCyber()
    
    while True:
        tema = input("\nSobre qual tema de Segurança Digital quer uma questão? (ou 'sair'): ")
        if tema.lower() == 'sair': break
        
        questao = gerador.criar_questao(tema)
        print("\nQUESTÃO GERADA:")
        print(json.dumps(questao, indent=4, ensure_ascii=False))