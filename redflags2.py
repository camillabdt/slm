import json
import time
import os
import re
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
from groq import Groq
from dotenv import load_dotenv


# ==========================
# 0) Configurações gerais
# ==========================
INPUT_SLM_JSONL = "questoesSLM.jsonl"
INPUT_LLM_JSON = "questoesLLM.json"
OUTPUT_CSV = "relatorio_comparativo_qualidade.csv"

TEMPERATURE = 0.1
SLEEP_SECONDS = 1.5          # proteção rate limit
MAX_RETRIES = 3              # tentativas em falhas temporárias
RETRY_BACKOFF_BASE = 2.0     # backoff exponencial

# Modelos que atuarão como "Juízes" (avaliadores)
JUIZES = {
    "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "Qwen-3-32B": "qwen/qwen3-32b",
    "Kimi-K2": "moonshotai/kimi-k2-instruct-0905"
}

# Prompt do sistema: papel do avaliador (mais específico e defensável)
SYSTEM_PROMPT = (
    "You are an expert evaluator in basic Computing education, with experience in the "
    "Brazilian National Common Curriculum Base (BNCC). "
    "You critically assess the pedagogical quality of multiple-choice questions for 9th-grade students."
)

# Prompt do usuário (avaliação pedagógica)
PROMPT_AVALIACAO = (
    "Analyze the following multiple-choice question (MCQ) designed for 9th-grade "
    "Computing education according to the Brazilian National Common Curriculum Base (BNCC).\n\n"
    "Classify the MCQ strictly as 'GOOD' or 'BAD'. Your decision must be based on the following criteria:\n"
    "(i) technical correctness of the computing concepts,\n"
    "(ii) clarity and suitability for 14-year-old students, and\n"
    "(iii) correctness and consistency of the indicated answer key.\n\n"
    "Response format:\n"
    "Start with 'VERDICT: GOOD' or 'VERDICT: BAD', followed by a brief justification.\n\n"
    "MCQ Content:\n{}"
)


# ==========================
# 1) Inicialização Groq
# ==========================
def init_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY não encontrado. Verifique seu .env ou variável de ambiente."
        )
    return Groq(api_key=api_key)


# ==========================
# 2) Carregamento dos dados
# ==========================
def carregar_e_unificar_dados() -> pd.DataFrame:
    consolidado: List[Dict[str, Any]] = []

    # Carregar do SLM (.jsonl)
    if os.path.exists(INPUT_SLM_JSONL):
        with open(INPUT_SLM_JSONL, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as e:
                    print(f"⚠️ JSON inválido em {INPUT_SLM_JSONL} linha {line_num}: {e}")
                    continue

                consolidado.append({
                    "origem_modelo": "SLM (TinyLlama)",
                    "tecnica_prompt": item.get("tecnica", "N/A"),
                    "conteudo_avaliar": item.get("output", "")
                })

    # Carregar das LLMs (.json)
    if os.path.exists(INPUT_LLM_JSON):
        with open(INPUT_LLM_JSON, "r", encoding="utf-8") as f:
            try:
                llm_data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Erro lendo {INPUT_LLM_JSON}: {e}")

        if not isinstance(llm_data, list):
            raise RuntimeError(f"{INPUT_LLM_JSON} deveria conter uma lista JSON.")

        for idx, item in enumerate(llm_data, 1):
            consolidado.append({
                "origem_modelo": f"LLM ({item.get('llm', 'UNKNOWN')})",
                "tecnica_prompt": item.get("tecnica", "N/A"),
                "conteudo_avaliar": item.get("conteudo_gerado", "")
            })

    df = pd.DataFrame(consolidado)
    if df.empty:
        raise RuntimeError(
            "Nenhuma questão carregada. Verifique se os arquivos de entrada existem e estão corretos."
        )

    # limpeza básica: remove linhas sem conteúdo
    df["conteudo_avaliar"] = df["conteudo_avaliar"].fillna("").astype(str)
    df = df[df["conteudo_avaliar"].str.strip() != ""].reset_index(drop=True)

    return df


# ==========================
# 3) Parser do veredito
# ==========================
VERDICT_RE = re.compile(r"VERDICT:\s*(GOOD|BAD)", re.IGNORECASE)

def extrair_veredito(resposta: str) -> Tuple[Optional[str], str]:
    """
    Retorna (veredito, justificativa curta).
    veredito ∈ {"GOOD","BAD", None}
    """
    if not resposta:
        return None, ""

    m = VERDICT_RE.search(resposta)
    verdict = m.group(1).upper() if m else None

    # justificativa: remove a linha de VERDICT para deixar um resumo
    justificativa = resposta
    justificativa = re.sub(r"^.*?VERDICT:\s*(GOOD|BAD)\s*", "", justificativa, flags=re.IGNORECASE | re.DOTALL).strip()
    # limita tamanho para CSV ficar leve (opcional)
    if len(justificativa) > 500:
        justificativa = justificativa[:500] + "..."
    return verdict, justificativa


# ==========================
# 4) Consulta ao juiz (com retries)
# ==========================
def consultar_juiz(client: Groq, conteudo: str, model_id: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            chat_completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT_AVALIACAO.format(conteudo)}
                ],
                temperature=TEMPERATURE,
                max_tokens=600
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            wait_s = RETRY_BACKOFF_BASE ** (attempt - 1)
            print(f"⚠️ Falha ({attempt}/{MAX_RETRIES}) no modelo {model_id}: {e}. Retentando em {wait_s:.1f}s...")
            time.sleep(wait_s)
    return f"Error: {str(last_err)}"


# ==========================
# 5) Execução
# ==========================
def main():
    client = init_client()
    df = carregar_e_unificar_dados()

    print(f"📂 Arquivos carregados. Total de questões a avaliar: {len(df)}")
    print(f"🧑‍⚖️ Juízes: {', '.join(JUIZES.keys())}")
    print(f"⚙️ Temperature={TEMPERATURE} | sleep={SLEEP_SECONDS}s | retries={MAX_RETRIES}")

    # Se já existir um CSV anterior, você pode optar por continuar
    if os.path.exists(OUTPUT_CSV):
        print(f"ℹ️ Arquivo {OUTPUT_CSV} já existe. Vou sobrescrever.")
        # Se quiser continuar incremental, posso adaptar depois.
        os.remove(OUTPUT_CSV)

    # Avaliar com cada juiz
    for nome_juiz, id_juiz in JUIZES.items():
        print(f"\n👨‍⚖️ Juiz {nome_juiz} iniciando avaliações...")

        raw_col = f"veredito_{nome_juiz}"
        verdict_col = f"verdict_{nome_juiz}"
        just_col = f"just_{nome_juiz}"

        raw_out = []
        v_out = []
        j_out = []

        for i, row in df.iterrows():
            print(f"   -> Avaliando questão {i+1}/{len(df)}...")
            resposta = consultar_juiz(client, row["conteudo_avaliar"], id_juiz)
            v, just = extrair_veredito(resposta)

            raw_out.append(resposta)
            v_out.append(v if v else "")
            j_out.append(just)

            time.sleep(SLEEP_SECONDS)

        df[raw_col] = raw_out
        df[verdict_col] = v_out
        df[just_col] = j_out

        # salvamento incremental após cada juiz (evita perder progresso)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"✅ Parcial salvo após {nome_juiz} em {OUTPUT_CSV}")

    print(f"\n✅ Relatório final '{OUTPUT_CSV}' gerado com sucesso!")


if __name__ == "__main__":
    main()
