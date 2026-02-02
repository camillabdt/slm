import json
import time
import os
import re
from typing import Dict, Any, Optional, Tuple, List
import random
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
INPUT_SLM_JSONL = "questoesSLM.jsonl"
INPUT_LLM_JSON = "questoesLLM.json"
OUTPUT_CSV = "relatorio_redflag222.csv"

TEMPERATURE = 0.1
MAX_TOKENS_OUT = 140          
MAX_CHARS_MCQ = 900           

SLEEP_SECONDS = 1.2
MAX_RETRIES = 5              
RETRY_BACKOFF_BASE = 2.0

JUIZES = {
    # "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "Qwen-3-32B": "qwen/qwen3-32b",
    "Kimi-K2": "moonshotai/kimi-k2-instruct-0905"
}

SYSTEM_PROMPT = (
    "You are an expert evaluator in basic Computing education aligned with BNCC. "
    "Be strict and concise."
)

PROMPT_AVALIACAO = (
    "Read the MCQ (9th-grade Computing, BNCC). Decide if it is GOOD or BAD.\n"
    "Do NOT include reasoning steps.\n"
    "Write EXPLANATION in max 2 sentences.\n"
    "Write KEY POINT as max 5 words.\n\n"
    "Format exactly:\n"
    "VERDICT: GOOD|BAD\n"
    "EXPLANATION: ...\n"
    "KEY POINT: ...\n\n"
    "MCQ:\n{}"
)

# ==========================
# FUNÇÕES DE APOIO
# ==========================
def init_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não encontrado. Verifique seu .env.")
    return Groq(api_key=api_key)

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def compact_mcq(text: str, max_chars: int = MAX_CHARS_MCQ) -> str:
    t = normalize_text(text)
    pattern = re.compile(
        r"(Question:.*?)(Correct Answer:.*?|Answer:.*?|Resposta correta:.*?|$)(Explanation:.*?|Feedback:.*?|$)",
        re.IGNORECASE
    )
    m = pattern.search(t)
    core = " ".join([x for x in m.groups() if x]).strip() if m else t
    if len(core) <= max_chars: return core
    head = core[: int(max_chars * 0.70)]
    tail = core[-int(max_chars * 0.20):]
    return head + " ... [TRUNCATED] ... " + tail

def carregar_e_unificar_dados() -> pd.DataFrame:
    consolidado = []
    if os.path.exists(INPUT_SLM_JSONL):
        with open(INPUT_SLM_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                consolidado.append({
                    "origem_modelo": "SLM (TinyLlama)",
                    "tecnica_prompt": item.get("tecnica", "N/A"),
                    "conteudo_avaliar": item.get("output", "")
                })
    if os.path.exists(INPUT_LLM_JSON):
        with open(INPUT_LLM_JSON, "r", encoding="utf-8") as f:
            llm_data = json.load(f)
            for item in llm_data:
                consolidado.append({
                    "origem_modelo": f"LLM ({item.get('llm', 'UNKNOWN')})",
                    "tecnica_prompt": item.get("tecnica", "N/A"),
                    "conteudo_avaliar": item.get("conteudo_gerado", "")
                })
    df = pd.DataFrame(consolidado)
    df["conteudo_avaliar"] = df["conteudo_avaliar"].fillna("").astype(str)
    return df.reset_index(drop=True)

# ==========================
# LÓGICA DE AVALIAÇÃO
# ==========================
def consultar_juiz(client: Groq, conteudo: str, model_id: str) -> str:
    compact = compact_mcq(conteudo, MAX_CHARS_MCQ)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            chat_completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT_AVALIACAO.format(compact)}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_OUT
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if "429" in str(e):
                wait_s = (RETRY_BACKOFF_BASE ** attempt) + random.uniform(0, 1)
                print(f"\n⚠️ Rate Limit no {model_id}. Aguardando {wait_s:.2f}s...")
                time.sleep(wait_s)
            else:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
    return f"Error: {str(last_err)}"

VERDICT_RE = re.compile(r"VERDICT:\s*(GOOD|BAD)", re.IGNORECASE)
EXPL_RE = re.compile(r"EXPLANATION:\s*(.*)", re.IGNORECASE)
KEY_RE = re.compile(r"KEY POINT:\s*(.*)", re.IGNORECASE)

def parse_response(text: str) -> Tuple[str, str, str]:
    t = (text or "").strip()
    verdict, explanation, key_point = "NO_VERDICT", "", ""
    m = VERDICT_RE.search(t)
    if m: verdict = m.group(1).upper()
    m = EXPL_RE.search(t)
    if m: explanation = m.group(1).strip()[:280]
    m = KEY_RE.search(t)
    if m: key_point = m.group(1).strip()[:80]
    return verdict, explanation, key_point

# ==========================
# EXECUÇÃO PRINCIPAL
# ==========================
def main():
    client = init_client()
    df = carregar_e_unificar_dados()

    print(f"📂 Total de questões: {len(df)}")

    for nome_juiz, id_juiz in JUIZES.items():
        print(f"\n👨‍⚖️ Juiz {nome_juiz} iniciando...")
        
        # --- CORREÇÃO AQUI: Definindo os nomes das colunas ANTES do uso ---
        verdict_col = f"verdict_{nome_juiz}"
        expl_col = f"explanation_{nome_juiz}"
        key_col = f"keypoint_{nome_juiz}"

        v_out, e_out, k_out = [], [], []

        for i, row in df.iterrows():
            print(f"  -> {i+1}/{len(df)}", end="\r")
            resp = consultar_juiz(client, row["conteudo_avaliar"], id_juiz)
            verdict, expl, key = parse_response(resp)
            v_out.append(verdict)
            e_out.append(expl)
            k_out.append(key)
            time.sleep(SLEEP_SECONDS)

        # Atribuição correta
        df[verdict_col] = v_out
        df[expl_col] = e_out
        df[key_col] = k_out
        
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\n✅ Parcial salvo após {nome_juiz}. Respirando...")
        time.sleep(5)

    print(f"\n✅ Finalizado: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()