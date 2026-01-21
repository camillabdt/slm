import json
import time
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# 1. Configurações iniciais
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Modelos que atuarão como "Juízes" (avaliadores)
JUIZES = {
    "Llama-3.3-70B": "llama-3.3-70b-versatile",   # Meta: O equilíbrio entre velocidade e inteligência.
    "GPT-OSS-120B": "openai/gpt-oss-120b",         # OpenAI (Open): Arquitetura densa de larga escala.
    "Qwen-3-32B": "qwen/qwen3-32b",                # Alibaba: Focado em lógica, matemática e código.
    "Kimi-K2": "moonshotai/kimi-k2-instruct-0905"  # Moonshot: Excelente para contextos longos e raciocínio.
}

# 2. Prompt de Avaliação Pedagógica (BNCC)
PROMPT_AVALIACAO = (
    "Analyze the following 9th-grade Computing MCQ (BNCC). "
    "Classify it as 'GOOD' or 'BAD' and justify based on technical accuracy, "
    "clarity for 14-year-olds, and answer key correctness.\n\n"
    "Format: Start your response with 'VERDICT: GOOD' or 'VERDICT: BAD', then provide a short explanation.\n\n"
    "Question Content:\n{}"
)

# 3. Função para carregar e unificar os dois arquivos
def carregar_e_unificar_dados():
    consolidado = []
    
    # Carregar do SLM (Arquivo .jsonl)
    if os.path.exists("questoesSLM.jsonl"):
        with open("questoesSLM.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                consolidado.append({
                    "origem_modelo": "SLM (TinyLlama)",
                    "tecnica_prompt": item.get('tecnica', 'N/A'),
                    "conteudo_avaliar": item.get('output', '')
                })

    # Carregar das LLMs (Arquivo .json)
    if os.path.exists("questoesLLM.json"):
        with open("questoesLLM.json", "r", encoding="utf-8") as f:
            llm_data = json.load(f)
            for item in llm_data:
                consolidado.append({
                    "origem_modelo": f"LLM ({item.get('llm')})",
                    "tecnica_prompt": item.get('tecnica', 'N/A'),
                    "conteudo_avaliar": item.get('conteudo_gerado', '')
                })
    
    return pd.DataFrame(consolidado)

# 4. Função de consulta à Groq
def consultar_juiz(conteudo, model_id):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional educational evaluator for the Brazilian Computing Curriculum (BNCC)."},
                {"role": "user", "content": PROMPT_AVALIACAO.format(conteudo)}
            ],
            model=model_id,
            temperature=0.1 # Rigor na avaliação
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# 5. Execução do Experimento de Avaliação
if __name__ == "__main__":
    df = carregar_e_unificar_dados()
    print(f"📂 Arquivos carregados. Total de questões a avaliar: {len(df)}")

    # Loop pelos Juízes
    for nome_juiz, id_juiz in JUIZES.items():
        print(f"\n👨‍⚖️ Juiz {nome_juiz} iniciando avaliações...")
        resultados_juiz = []
        
        for i, row in df.iterrows():
            print(f"   -> Avaliando questão {i+1}/{len(df)}...")
            veredito = consultar_juiz(row['conteudo_avaliar'], id_juiz)
            resultados_juiz.append(veredito)
            time.sleep(1.5) # Proteção de Rate Limit
            
        # Adiciona o veredito deste juiz como uma nova coluna no DataFrame
        df[f'veredito_{nome_juiz}'] = resultados_juiz

    # 6. Salvar Relatório Final
    df.to_csv("relatorio_comparativo_qualidade.csv", index=False)
    print("\n✅ Relatório 'relatorio_comparativo_qualidade.csv' gerado com sucesso!")