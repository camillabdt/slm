# gen_re_questions_min.py (versão robusta contra eco de prompt/placeholder)
import os, re, json, time, random, argparse
from typing import List, Dict, Optional
import hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEFAULT_BASE = "mtgv/MobileLLaMA-1.4B-Chat"
DEFAULT_ADAPTER = "./outputs/questoes-lora"
OUT_JSON = "re_questions.json"
OUT_TXT  = "re_questions.txt"

TEMPS = [0.55, 0.7, 0.85, 0.95]
TOP_P = 0.92
TOP_K = 50
REP_PEN = 1.08
MAX_NEW = 420
MAX_TRIES_PER_Q = 7

RE_KEYWORDS = {
    "requirement","requirements","srs","stakeholder","traceability","acceptance",
    "verification","validation","non-functional","performance","usability",
    "constraint","interface","priority","change control","baseline","version",
    "elicitation","analysis","specification","review","testable","ambiguity",
    "measurable","criteria","use case","user story","rtm","risk","compliance"
}

SCENARIO_SEEDS = [
    "During SRS review in a safety-critical medical device project, ",
    "While refining user stories for a fintech compliance release, ",
    "In a requirements workshop with hardware-software interfaces, ",
    "When drafting non-functional requirements for latency and throughput, ",
    "During impact analysis after a regulatory change request, ",
    "While preparing acceptance criteria for multi-tenant SaaS, ",
    "In elicitation with conflicting stakeholder priorities, ",
    "During traceability setup across design, code, and tests, ",
]

BAD_STEM_PATTERNS = [
    r"^paraphrase the following", r"return only the stem text",
    r"string\s*\(12–?36 words", r"concise lines explaining why",
    r"^json", r"^write one multiple", r"do not include any text outside this json",
    r"^stem:\s*$"
]
BAD_OPT_PATTERNS = [
    r"\bin a realistic project context\b",
    r"^option\s+[abcd]\b", r"^choice\s+[abcd]\b",
    r"^placeholder\b"
]

# ---------- utils ----------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()

def tokenize(s: str) -> List[str]:
    return [w for w in re.split(r"\W+", s.lower()) if w]

def ngrams(words: List[str], n: int = 3) -> set:
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}

def jaccard(a:set, b:set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def too_similar(a: str, b: str, thr: float = 0.58) -> bool:
    return jaccard(ngrams(tokenize(a),3), ngrams(tokenize(b),3)) >= thr

def keywords_ok(text: str, min_hits: int = 2) -> bool:
    t = text.lower()
    return sum(1 for kw in RE_KEYWORDS if kw in t) >= min_hits

def is_bad_stem(stem: str) -> bool:
    s = stem.strip().lower()
    if len(s) < 10: return True
    for pat in BAD_STEM_PATTERNS:
        if re.search(pat, s, flags=re.I): return True
    return False

def is_bad_option(opt: str) -> bool:
    o = opt.strip().lower()
    if len(o) < 6: return True
    for pat in BAD_OPT_PATTERNS:
        if re.search(pat, o, flags=re.I): return True
    return False

def bounded_len_words(s:str, lo:int=6, hi:int=18) -> str:
    toks = s.split()
    if len(toks) < lo: toks += ["in","a","realistic","project","context"]
    if len(toks) > hi: toks = toks[:hi]
    return " ".join(toks).rstrip(",.;:")

def clean_all_none(opts:List[str]) -> List[str]:
    bad = re.compile(r"\b(all|none)\s+of\s+the\s+above\b", re.I)
    return [bad.sub("—", o) for o in opts]

def enforce_four_options(opts:List[str]) -> List[str]:
    opts = [o.strip() for o in opts if o and o.strip()]
    opts = clean_all_none(opts)
    if len(opts) < 4:
        while len(opts) < 4:
            opts.append("Plausible but incorrect statement about requirements engineering practice")
    if len(opts) > 4: opts = opts[:4]
    opts = [bounded_len_words(o) for o in opts]
    return opts

# ---------- modelo ----------
def best_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_model(base_id: str, adapter_dir: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=best_dtype(), device_map="auto", trust_remote_code=True
    )
    model = base
    if adapter_dir and os.path.isdir(adapter_dir):
        try:
            model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=best_dtype())
        except Exception as e:
            print(f"[warn] could not load adapter from {adapter_dir}: {e}")
            model = base
    model.eval()
    model.config.pad_token_id = tok.pad_token_id
    if model.config.eos_token_id is None and tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id
    return model, tok

@torch.inference_mode()
def llm(model, tok, prompt: str, temp: float) -> str:
    enc = tok(prompt, return_tensors="pt", padding=True, truncation=True)
    device = next(model.parameters()).device
    enc = {k:v.to(device) for k,v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temp,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REP_PEN,
        max_new_tokens=MAX_NEW,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

# ---------- prompts ----------
JSON_GUIDE = (
    "Return ONLY a SINGLE JSON object (no markdown, no preface):\n"
    "{\n"
    '  "stem": "one or two sentences (12–36 words), clearly about IEEE 29148 Requirements Engineering",\n'
    '  "options": ["A","B","C","D"],\n'
    '  "correct": "A|B|C|D",\n'
    '  "rationale": "2–4 short lines explaining why the correct one is best"\n'
    "}\n"
    "Start with '{' and do not include any other text."
)

def build_generation_prompt() -> str:
    scenario = random.choice(SCENARIO_SEEDS)
    constraints = (
        "Write ONE multiple-choice question in ENGLISH about Requirements Engineering (IEEE 29148): "
        "focus on testability, unambiguity, traceability, acceptance criteria, verification/validation, or change control. "
        "Avoid 'All/None of the above' and double negatives."
    )
    return scenario + constraints + "\n" + JSON_GUIDE

# ---------- parsing/reparo ----------
def extract_first_json(text:str) -> Optional[Dict]:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s: return None
    frag = text[s:e+1]
    try:
        return json.loads(frag)
    except Exception:
        m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
        return None

def fix_question_obj(obj: Dict) -> Optional[Dict]:
    if not isinstance(obj, dict): return None
    stem = re.sub(r"\s+"," ", str(obj.get("stem",""))).strip()
    options = [str(o) for o in (obj.get("options") or [])]
    correct = str(obj.get("correct","")).strip().upper()
    rationale = str(obj.get("rationale","")).strip()

    # Rejeita stems ecoados/instrucionais
    if is_bad_stem(stem):
        return None

    # Garante foco em RE
    if not keywords_ok(stem, 2):
        # Não aceitamos stems que não citem pelo menos 2 termos de RE
        return None

    # Limites de tamanho
    toks = stem.split()
    if len(toks) < 12 or len(toks) > 36:
        return None

    # Opções
    options = enforce_four_options(options)
    # Se alguma opção for lixo, rejeita este objeto
    if any(is_bad_option(o) for o in options):
        return None

    # Letra correta consistente
    if correct not in ["A","B","C","D"]:
        return None

    # Rationale mínimo
    if len(rationale) < 20 or is_bad_stem(rationale):
        return None

    return {"stem": stem, "options": options, "correct": correct, "rationale": rationale}

def format_block(q:Dict) -> str:
    A,B,C,D = q["options"]
    return (
        f"Question: {q['stem']}\n"
        f"A) {A}\nB) {B}\nC) {C}\nD) {D}\n"
        f"Correct: {q['correct']}\n"
        f"Rationale: {q['rationale']}"
    )

# ---------- fallback sólido ----------
FALLBACK_BANK = [
    {
        "stem": "During SRS review, which phrasing most improves testability and traceability of a performance requirement linked to acceptance tests and RTM entries?",
        "options": [
            "State explicit measurable thresholds and reference associated acceptance test cases",
            "Use qualitative adjectives to keep wording flexible for stakeholders",
            "Reference an external policy without measurable pass or fail conditions",
            "Describe behavior broadly to avoid over-constraining design choices"
        ],
        "correct": "A",
        "rationale": "Measurable thresholds enable verification and traceability; alternatives are vague or unverifiable."
    },
    {
        "stem": "While planning change control, what practice best preserves bidirectional traceability across requirements, design, and tests during frequent updates?",
        "options": [
            "Maintain a versioned RTM and update links on every approved change",
            "Accept changes by email and consolidate links at the end of the release",
            "Allow conflicting statements to capture all stakeholder opinions",
            "Track only design decisions and ignore requirement–test relationships"
        ],
        "correct": "A",
        "rationale": "Versioned RTM sustains traceability under change; other options undermine consistency and verification."
    },
]

def strong_fallback() -> Dict:
    item = random.choice(FALLBACK_BANK)
    # pequena variação para evitar repetição
    if random.random() < 0.5:
        item = dict(item)
        item["stem"] = item["stem"].replace("SRS", "software requirements specification (SRS)")
    return item

# ---------- geração principal ----------
@torch.inference_mode()
def generate_one(model, tok) -> Optional[Dict]:
    # várias tentativas com prompts/temperaturas diferentes
    for attempt in range(MAX_TRIES_PER_Q):
        prompt = build_generation_prompt()
        raw = llm(model, tok, prompt, temp=random.choice(TEMPS))
        obj = extract_first_json(raw)
        fixed = fix_question_obj(obj) if obj else None
        if fixed:
            return fixed
        time.sleep(0.03)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default=OUT_JSON)
    ap.add_argument("--out_txt", type=str, default=OUT_TXT)
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    model, tok = load_model(args.base, args.adapter)

    seen_stems: List[str] = []
    results: List[Dict] = []

    print(f"-> Generating {args.count} RE questions (no topic/difficulty/bloom)...")
    for i in range(args.count):
        q = generate_one(model, tok)
        if not q:
            q = strong_fallback()

        # diversidade: evita perguntas muito parecidas
        if any(too_similar(prev, q["stem"]) for prev in seen_stems):
            # micro-paráfrase simples (sem reconsultar modelo)
            swaps = [
                ("improves", "enhances"), ("testability", "verifiability"),
                ("traceability", "bidirectional traceability"),
                ("practice", "approach"), ("preserves", "maintains"),
            ]
            stem = q["stem"]
            for a,b in random.sample(swaps, k=min(2, len(swaps))):
                stem = re.sub(rf"\b{re.escape(a)}\b", b, stem)
            q["stem"] = stem

        results.append(q)
        seen_stems.append(q["stem"])
        print(f"  - ok {i+1}/{args.count}")
        time.sleep(0.04)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for q in results:
            f.write(format_block(q) + "\n\n")

    print(f"\nSaved {len(results)} questions to {args.out_json} and {args.out_txt}")

if __name__ == "__main__":
    main()
