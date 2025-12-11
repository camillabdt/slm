# generateTinny.py
# GERAÇÃO FINAL E SIMPLES: Gera Stem, 5 Opções (A-E) e Gabarito (Answer/Explanation) em texto puro.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re 
from peft import PeftModel 
import warnings 

# Suprime o warning específico do bitsandbytes
warnings.filterwarnings("ignore", "MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")

# ----- CONFIGURAÇÃO -----
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "out-tinyllama-sft"))
MAX_LENGTH_GEN = 600 # Suficiente para toda a questão
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- CARREGAMENTO DO MODELO (Mantido) -----
print(f"[*] Loading model and tokenizer from: {MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True, 
        torch_dtype=torch.float32, 
        device_map=DEVICE,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, MODEL_PATH).eval()
    print(f"[INFO] Base model ({base_model_name}) with LoRA adapters loaded on: {DEVICE}")

except Exception as e:
    print(f"[ERROR] Failed to load model. Error: {e}")
    exit()

print("[*] Model loaded successfully.")

# ----- FUNÇÃO DE GERAÇÃO (Texto Simples A-E) -----

def generate_mcq_text_output():
    """
    Gera uma questão de múltipla escolha completa em formato de texto simples (não JSON), 
    com foco em preencher o Stem e as 5 opções.
    """
    
    topic = "Requirements Engineering"
    
    # Prompt de Saída Direta: Instrução detalhada em INGLÊS, mas sem formatação de código.
    # O modelo deve começar a gerar a pergunta (Stem) imediatamente.
    instruction_prompt = (
    "System: You are a senior instructor in Software Requirements Engineering (aligned with Sommerville and SWEBOK). "
    "Generate ONE high-quality multiple-choice question (MCQ) about Requirements Engineering (RE). "
    "Strictly follow the format below and the rules. Do not add any extra text.\n\n"
    "FORMAT (exactly this):\n"
    "Question (Stem): <one clear question about RE>\n"
    "A. <option A>\n"
    "B. <option B>\n"
    "C. <option C>\n"
    "D. <option D>\n"
    "E. <option E>\n"
    "Answer: <single letter A–E>\n"
    "Explanation: <1–3 short sentences justifying why the answer is correct and why the main distractor is wrong>\n\n"
    "RULES:\n"
    "1) Scope is RE: elicitation, analysis, documentation, negotiation, validation/verification, traceability, prioritization, stakeholders, NFRs, etc.\n"
    "2) Do NOT select topics that belong to QA/testing unless the stem explicitly asks (e.g., regression testing, unit tests, coverage). "
    "   Never pick regression testing as the correct answer for generic RE scope questions.\n"
    "3) Exactly ONE correct option. Make options mutually exclusive, concise, and plausible.\n"
    "4) Make options conceptually distinct (avoid near-duplicates, trivial negations, and giveaways).\n"
    "5) Prefer conceptual understanding over memorization; no math is needed.\n"
    "6) Language: English. No markdown, no code blocks, no bullet points besides the required format.\n"
    "7) Keep option lengths varied, but avoid making the correct one obviously longer or more qualified than the others.\n\n"
    "User: Generate the full text now, starting immediately with \"Question (Stem):\""
)

    
    # Tokeniza a entrada
    inputs = tokenizer(instruction_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    # Define parâmetros de geração
    generation_config = model.generation_config
    generation_config.max_new_tokens = MAX_LENGTH_GEN
    generation_config.do_sample = True
    generation_config.temperature = 0.7
    generation_config.top_p = 0.9
    generation_config.repetition_penalty = 1.05
    generation_config.eos_token_id = tokenizer.eos_token_id


    # Gera a sequência
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            generation_config=generation_config, 
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodifica a resposta (removendo o prompt original)
    output_text = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    
    print("\n" + "="*80)
    print("GENERATED MULTIPLE-CHOICE QUESTION (TEXT FORMAT)")
    print("="*80)
    print(output_text.strip())
    print("="*80 + "\n")


# ----- EXECUTION -----
generate_mcq_text_output()