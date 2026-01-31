import json
import time
import os
import random
from groq import Groq
from dotenv import load_dotenv

# 1. Configurações iniciais
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# IDs estáveis e recomendados da Groq para o experimento
LLMS = {
    "Llama-3.3-70B": "llama-3.3-70b-versatile",   # Meta: O equilíbrio entre velocidade e inteligência.
    "GPT-OSS-120B": "openai/gpt-oss-120b",         # OpenAI (Open): Arquitetura densa de larga escala.
    "Qwen-3-32B": "qwen/qwen3-32b",                # Alibaba: Focado em lógica, matemática e código.
    "Kimi-K2": "moonshotai/kimi-k2-instruct-0905"  # Moonshot: Excelente para contextos longos e raciocínio.
}

# 3. Engenharia de Prompt com Instrução de Saída em INGLÊS
INSTRUCAO_CHATML = (
    "\nIMPORTANT: The generated question, options, and explanation MUST BE IN ENGLISH.\n"
    "Follow this ChatML format strictly:\n"
    "<|system|>\nYou are a Computer Science Professor for 9th-grade students following the Brazilian BNCC curriculum.<|end|>\n"
    "<|user|>\n[Question Prompt]<|end|>\n"
    "<|assistant|>\nQuestion: [Text in English]\n0) [Option]\n1) [Option]\n2) [Option]\n3) [Option]\n4) [Option]\n"
    "Correct Answer: [Number]\nExplanation: [Technical explanation in English based on BNCC standards]<|end|>"
)

PROMPT_TEMPLATES = {
    "zero-shot": (
        "Generate a hard MCQ about 'Weighted Graphs' and social networks (BNCC EF09C001)." + INSTRUCAO_CHATML
    ),
    "few-shot": (
        "Example:\n<|user|>\nCreate about recursion.<|end|>\n"
        "<|assistant|>\nQuestion: What prevents infinite loops in recursion?\n0) Global variables\n1) Base Case\n2) For loops\n3) Print statements\n4) Return 0\nCorrect Answer: 1\nExplanation: The base case provides a stopping condition.<|end|>\n\n"
        "Task: Generate a new one about 'Recursive Functions' and sub-problems." + INSTRUCAO_CHATML
    ),
    "chain-of-thought": (
        "Think step-by-step: 1. Explain Asymmetric Encryption keys. 2. Identify risks of sharing the private key. 3. Create the MCQ (BNCC EF09C005)." + INSTRUCAO_CHATML
    ),
    "exemplar-guided": (
        "Follow this style: 'A data packet is like a letter cut into pieces'. "
        "Generate an MCQ about 'Packet Switching' and reassembly at destination (BNCC EF69C007)." + INSTRUCAO_CHATML
    ),
    "template-based": (
        "Complete the template: Topic: 'AI Ethics'. Question: [About transparency and algorithmic bias]... "
        "Correct Answer: [Number]. Explanation: [Why AI must be auditable] (BNCC EF09C007)." + INSTRUCAO_CHATML
    )
}

# 4. Execução do Experimento
N_QUESTIONS_PER_TECHNIQUE = 5
TEMPERATURAS = [0.2, 0.5, 0.8]
resultados = []

print("🚀 Starting data collection for dissertation (Multi-LLM / BNCC)...")

for llm_name, model_id in LLMS.items():
    for technique, prompt in PROMPT_TEMPLATES.items():
        for i in range(1, N_QUESTIONS_PER_TECHNIQUE + 1):
            temp_atual = random.choice(TEMPERATURAS)
            print(f"🤖 {llm_name} | {technique} | Q{i} | Temp: {temp_atual}")
            
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp_atual,
                    max_tokens=1000
                )
                saida = response.choices[0].message.content.strip()
            except Exception as e:
                saida = f"[ERROR] {str(e)}"
            
            resultados.append({
                "id": f"{llm_name}_{technique}_Q{i}",
                "llm": llm_name,
                "model_id": model_id,
                "tecnica": technique,
                "questao_n": i,
                "temperatura": temp_atual,
                "conteudo_gerado": saida
            })
            time.sleep(2.0) # Rate limit safety

# 5. Salvando os dados
with open("dataset_dissertacao_english.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

print(f"\n✅ Collection complete! File saved as 'dataset_dissertacao_english.json'.")