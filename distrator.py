# distrator.py
# Converte dataset QA (messages system/user/assistant) em dataset MCQ (4 opções) para fine-tuning.
# Entrada: datasetoficial.jsonl
# Saída: mcq_train.jsonl  (instruction/input/output)
#
# Como rodar:
#   python distrator.py
#
# Ajuste as constantes SRC/OUT e os pools de distratores conforme necessário.

import json
import random
import re
from typing import Any, Dict, List, Optional

# =========================
# CONFIG
# =========================
SRC = "datasetoficial.jsonl"
OUT = "mcq_train.jsonl"

SEED = 42
N_DISTRACTORS = 3
MAX_OPTION_LEN = 140
MAX_FEEDBACK_LEN = 240

random.seed(SEED)

# =========================
# Distratores seguros (educacionais)
# =========================
DISTRACTORS = {
    "phishing": [
        "A method to compress files to save storage.",
        "A safe login method that never requires verification.",
        "A harmless system notification that is always legitimate.",
        "A feature that automatically shares personal data online.",
        "A way to speed up public Wi-Fi connections."
    ],
    "passwords_and_auth": [
        "Using the same password everywhere so you don't forget it.",
        "Sharing your password with friends to stay safe.",
        "Posting your password in a private group for backup.",
        "Using very short passwords because they are easier to type.",
        "Turning off authentication to speed up login."
    ],
    "social_engineering": [
        "A type of antivirus update that always comes from official sources.",
        "A legal technique to collect user data without consent.",
        "A network optimization method used by internet providers.",
        "A feature to automatically accept unknown friend requests.",
        "A safe way to bypass identity checks online."
    ],
    "privacy": [
        "A method to increase screen brightness to protect data.",
        "A way to make your phone battery last longer.",
        "A feature that publicly shares your location by default.",
        "A type of game setting that improves graphics quality.",
        "A tool that automatically posts your personal details online."
    ],
    "_default": [
        "A way to improve device battery life.",
        "A feature that only changes the color of an app.",
        "A method to make games run faster.",
        "A type of keyboard shortcut.",
        "A harmless pop-up that always comes from trusted sites."
    ],
}

# =========================
# Utilitários
# =========================
def normalize_content(content: Any) -> str:
    """
    Converte content em string, mesmo se vier como dict/list.
    - dict: tenta campos comuns; senão concatena valores.
    - list: concatena itens.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        # tenta chaves comuns
        for k in ["text", "content", "answer", "response", "message", "output", "explanation"]:
            if k in content and isinstance(content[k], str):
                return content[k].strip()
        # concatena valores
        return " ".join(str(v) for v in content.values()).strip()

    if isinstance(content, list):
        parts = []
        for it in content:
            parts.append(normalize_content(it))
        return " ".join(p for p in parts if p).strip()

    return str(content).strip()

def compact_text(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "..."

def pick_topic_bucket(topic: str) -> str:
    """
    Mapeia topic do seu dataset para um bucket de distratores.
    Se seu dataset usa outros nomes, ajuste aqui.
    """
    t = (topic or "").lower()
    if "phish" in t:
        return "phishing"
    if "password" in t or "2fa" in t or "auth" in t:
        return "passwords_and_auth"
    if "social" in t or "engineering" in t:
        return "social_engineering"
    if "privacy" in t or "data" in t or "lgpd" in t:
        return "privacy"
    return "_default"

def pick_distractors(topic_bucket: str, correct_text: str, k: int = 3) -> List[str]:
    pool = DISTRACTORS.get(topic_bucket, DISTRACTORS["_default"])
    pool = [p for p in pool if p.strip().lower() != (correct_text or "").strip().lower()]
    if len(pool) <= k:
        return pool[:k]
    return random.sample(pool, k)

def find_message(messages: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == role:
            return m
    return None

def build_mcq(user_q: str, assistant_a: str, topic_bucket: str) -> Dict[str, Any]:
    """
    Gera uma MCQ 4 opções:
    - opção correta = resposta compactada do assistant
    - 3 distratores de pool seguro
    """
    correct = compact_text(assistant_a, MAX_OPTION_LEN)
    distractors = pick_distractors(topic_bucket, correct, k=N_DISTRACTORS)

    options = distractors + [correct]
    random.shuffle(options)
    correct_idx = options.index(correct)

    feedback = compact_text(assistant_a, MAX_FEEDBACK_LEN)

    return {
        "pergunta": compact_text(user_q, 260),
        "opcoes": options,
        "correta": correct_idx,
        "feedback": feedback
    }

# =========================
# Pipeline principal
# =========================
def main():
    n_in = 0
    n_out = 0
    skipped = {}

    with open(SRC, "r", encoding="utf-8") as f_in, open(OUT, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            try:
                obj = json.loads(line)
            except Exception:
                skipped["json_parse_fail"] = skipped.get("json_parse_fail", 0) + 1
                continue

            topic = obj.get("topic", "")
            topic_bucket = pick_topic_bucket(topic)

            messages = obj.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                skipped["missing_messages"] = skipped.get("missing_messages", 0) + 1
                continue

            user_msg = find_message(messages, "user")
            asst_msg = find_message(messages, "assistant")

            if not user_msg or not asst_msg:
                skipped["missing_user_or_assistant"] = skipped.get("missing_user_or_assistant", 0) + 1
                continue

            user_q = normalize_content(user_msg.get("content"))
            assistant_a = normalize_content(asst_msg.get("content"))

            if not user_q:
                skipped["empty_user"] = skipped.get("empty_user", 0) + 1
                continue
            if not assistant_a:
                skipped["empty_assistant"] = skipped.get("empty_assistant", 0) + 1
                continue

            mcq = build_mcq(user_q, assistant_a, topic_bucket)

            # Formato de treino SFT (instruction/input/output)
            record = {
                "instruction": "Generate one multiple-choice question (4 options) for 9th-grade cybersecurity education. Output ONLY JSON.",
                "input": f"Topic bucket: {topic_bucket}",
                "output": json.dumps(mcq, ensure_ascii=False)
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"✅ Conversão concluída.")
    print(f"   Linhas lidas: {n_in}")
    print(f"   Linhas geradas (MCQ): {n_out}")
    if skipped:
        print("⚠️ Motivos de descarte:")
        for k, v in sorted(skipped.items(), key=lambda x: -x[1]):
            print(f"   - {k}: {v}")
    print(f"\n📄 Saída: {OUT}")

if __name__ == "__main__":
    main()
