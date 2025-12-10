import json, re, random, math, hashlib
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# -------- Configurações ----------
INPUT = "questions.json"        # seu dataset de entrada
OUT_SFT = "clean_train.jsonl"   # itens bons e variados para SFT
OUT_PREF = "pref_pairs.jsonl"   # pares bom vs ruim (DPO/ORPO)
MIN_STEM_WORDS = 10
MAX_STEM_WORDS = 40
SIM_NEAR_DUP = 0.90  # similaridade para dedup parcial
TARGET_PER_SUBAREA_MIN = 10     # mínimo por subárea
TARGET_ITEMS_TOTAL = 200        # alvo de itens após enriquecimento
# ---------------------------------

RE_TERMS = [
    "SRS", "RTM", "acceptance test", "traceability", "verifiable",
    "bidirectional traceability", "baseline", "change control",
    "non-functional requirement", "performance", "security", "reliability",
    "usability", "availability", "testable", "measurable"
]

SUBAREAS = [
    "elicitation", "stakeholder analysis", "SRS wording",
    "traceability/RTM", "prioritization", "change control",
    "NFR/performance", "NFR/security", "NFR/reliability", "NFR/usability",
    "acceptance & validation", "conflict resolution", "attributes", "baseline/versioning"
]

DOMAINS = [
    ("e-commerce", "checkout latency",      "p99 ≤ {ms} ms",      [150, 200, 250, 300]),
    ("telemedicina", "upload de exame",     "taxa de sucesso ≥ {pct}%", [92, 95, 97, 99]),
    ("banco digital", "autenticação MFA",   "falhas ≤ {ppm} ppm",  [50, 80, 100, 150]),
    ("IoT industrial", "telemetria",        "perda de pacote ≤ {pct}%", [0.5, 1.0, 2.0, 5.0]),
    ("edtech", "renderização de aula",      "tempo p95 ≤ {s} s",   [1.0, 1.5, 2.0, 3.0]),
]

STOP_PREFIXES = [
    "Paraphrase the following", "Return ONLY the stem",
    "Paraphrase the following question STEM", "Paraphrase the question"
]

BAD_OPTIONS_MARKERS = ["all of the above", "none of the above", "todas as anteriores", "nenhuma das anteriores"]

def load_any(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    try:
        # tenta json list
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # tenta jsonl
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        items.append(json.loads(line))
    return items

def normalize_item(x):
    # tenta mapear campos previstos
    stem = x.get("stem") or x.get("question") or x.get("prompt") or ""
    options = x.get("options") or x.get("choices") or []
    correct = x.get("correct") or x.get("answer") or ""
    rationale = x.get("rationale") or x.get("explanation") or ""
    subarea = x.get("subarea") or ""
    domain = x.get("domain") or ""

    # limpa espaços
    stem = re.sub(r"\s+", " ", stem).strip()
    options = [re.sub(r"\s+", " ", str(o)).strip() for o in options if str(o).strip()]
    correct = str(correct).strip().upper()

    # se correct não é letra, tenta cascatear por texto
    letters = ["A","B","C","D","E"]
    if correct not in letters and options:
        idx = None
        for i, o in enumerate(options):
            if o.startswith("Correct:") or o.startswith("Gabarito:"):
                idx = i
                break
        if idx is not None and 0 <= idx < len(letters):
            correct = letters[idx]

    return {
        "stem": stem, "options": options, "correct": correct,
        "rationale": rationale, "subarea": subarea, "domain": domain
    }

def is_control_stem(stem: str):
    s = stem.lower()
    return any(stem.startswith(p) for p in STOP_PREFIXES) or "return only the stem" in s

def has_re_term(stem: str):
    s = stem.lower()
    return any(t.lower() in s for t in RE_TERMS)

def ok_length(stem: str):
    n = len(stem.split())
    return MIN_STEM_WORDS <= n <= MAX_STEM_WORDS

def options_unique(opts):
    return len(opts) == len(set(opts))

def ban_all_none(opts):
    s = " ".join(opts).lower()
    return not any(m in s for m in BAD_OPTIONS_MARKERS)

def hash_exact(stem, opts):
    h = hashlib.sha256()
    h.update(stem.encode())
    for o in opts:
        h.update(b"||")
        h.update(o.encode())
    return h.hexdigest()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def dedup(items):
    seen_hashes = set()
    kept = []
    # exato
    for it in items:
        h = hash_exact(it["stem"], it["options"])
        if h in seen_hashes: continue
        seen_hashes.add(h)
        kept.append(it)
    # near-duplicate pelo stem
    final = []
    stems = []
    for it in kept:
        dup = False
        for s in stems:
            if similar(it["stem"], s) >= SIM_NEAR_DUP:
                dup = True
                break
        if not dup:
            stems.append(it["stem"])
            final.append(it)
    return final

def hard_negatives_from(spec: str):
    vague = re.sub(r"(≤|≥|<|>|=)\s*[\d\.]+\s*\w*", "is adequate for stakeholders", spec)
    no_test = "aligns with policy without explicit pass/fail"
    wrong = spec
    nums = re.findall(r"[\d\.]+", spec)
    if nums:
        n = float(nums[0])
        wrong_val = n - 10 if n > 10 else n + 0.1
        wrong = spec.replace(str(n).rstrip('0').rstrip('.') if '.' in str(n) else str(int(n)),
                             str(round(wrong_val, 2)))
    return list(dict.fromkeys([vague, no_test, wrong]))[:3]

def enrich_item(it):
    """Força testabilidade/RTM se o item for genérico; cria distratores fortes."""
    domain, scenario, spec_tpl, values = random.choice(DOMAINS)
    # se já tinha domínio/subárea, mantenha; senão preencha
    subarea = it["subarea"] or random.choice(SUBAREAS)
    val = random.choice(values)
    spec = spec_tpl.format(ms=val, pct=val, ppm=val, s=val)

    # se o stem é fraco, reescreva preservando tema
    stem = it["stem"]
    if not has_re_term(stem) or not ok_length(stem):
        stem = (
            f"During {subarea} for a {domain} system, the SRS must express the {scenario} "
            f"requirement in a traceability- and acceptance-test-oriented way: how should the "
            f"clause be written to be testable and linked to RTM entries?"
        )

    # gabarito mensurável + RTM
    correct = f"State: {spec} and reference acceptance test IDs and RTM links"

    # distratores a partir do gabarito
    d1, d2, d3 = hard_negatives_from(spec)
    negs = [
        f"Use qualitative wording: {d1}",
        f"Reference policy only: {d2}",
        f"Keep broad description: {d3}"
    ]

    # constrói alternativas (garantindo 4 ou 5 sem repetições)
    options = [correct] + negs
    options = [o for i, o in enumerate(options) if o and o not in options[:i]]
    # Se tiver só 4, ok. Se quiser 5, pode adicionar um distrator genérico plausível:
    if len(options) < 4:
        options.append("Emphasize stakeholder satisfaction without measurable thresholds")
    options = options[:5]

    # embaralha e define letra correta
    random.shuffle(options)
    letters = ["A","B","C","D","E"]
    correct_letter = letters[options.index(correct)]

    # higiene final
    if not ok_length(stem): return None
    if not has_re_term(stem): return None
    if not options_unique(options): return None
    if not ban_all_none(options): return None

    return {
        "stem": stem,
        "options": options,
        "correct": correct_letter,
        "rationale": "Explicit, measurable threshold + test IDs/RTM → verificável e rastreável; demais são vagas ou sem pass/fail.",
        "subarea": subarea,
        "domain": it.get("domain") or domain
    }

def make_bad_variant(good):
    """
    Gera uma variante ruim do mesmo item (para pares de preferência).
    - Mapeia corretamente a letra do gabarito para o índice (A–E) considerando o nº de opções.
    - Degrada o texto correto removendo mensurabilidade/RTM, mantendo plausibilidade.
    """
    options = good.get("options", [])
    if not options or len(options) < 2:
        return None

    # mapa de letra -> índice, respeitando a quantidade de opções do item
    letters = ["A", "B", "C", "D", "E"][:len(options)]
    try:
        correct_letter = str(good.get("correct", "")).strip().upper()
        correct_idx = letters.index(correct_letter)
    except ValueError:
        # fallback seguro
        correct_idx = 0

    chosen_text = options[correct_idx]

    # Enfraquece o gabarito: remove mensuração e referências explícitas a RTM/testes
    rejected = chosen_text
    # remove números + unidades comuns e comparadores
    rejected = re.sub(r"(p\d{1,2}\s*)?(≤|≥|<|>|=)\s*[\d\.]+\s*(ms|s|ppm|%|req/s|req\/s)?", " is adequate ", rejected, flags=re.I)
    # troca RTM / acceptance test por algo vago
    rejected = re.sub(r"(RTM|acceptance test(s)?( IDs?)?)", "project documentation", rejected, flags=re.I)
    # remove palavras que reforçam testabilidade
    rejected = re.sub(r"\b(testable|verifiable|measurable)\b", "clear", rejected, flags=re.I)

    # se ainda ficou igual, cria um negativo genérico
    if rejected.strip() == chosen_text.strip():
        rejected = "Describe behavior broadly without measurable thresholds or RTM links"

    # evita colisão com alguma opção existente
    if rejected in options:
        rejected = rejected + " (without pass/fail conditions)"

    # prompt piorado (mantém tema, remove terminologia forte)
    stem = re.sub(r"(testable|verifiable|traceability|acceptance test|RTM)", "clear",
                  good.get("stem", ""), flags=re.I)
    if not stem.strip():
        stem = "Rewrite the requirement focusing on clarity for stakeholders."

    return {
        "prompt": stem,
        "chosen": chosen_text,   # bom
        "rejected": rejected,    # ruim
        "meta": {
            "subarea": good.get("subarea", ""),
            "domain": good.get("domain", "")
        }
    }

def balance_correct_letters(items):
    # balancear distribuição A–E
    buckets = {"A":0,"B":0,"C":0,"D":0,"E":0}
    letters = list(buckets.keys())
    balanced = []
    for it in items:
        cur = it["correct"]
        if cur not in buckets:
            # se vier com 4 opções, limite A-D
            buckets4 = {k:v for k,v in buckets.items() if k in ["A","B","C","D"]}
            target = min(buckets4, key=buckets4.get)
            # rotaciona alternativas
            idx = ["A","B","C","D"].index(cur) if cur in ["A","B","C","D"] else 0
            shift = (["A","B","C","D"].index(target) - idx) % 4
            opts = it["options"][:4]
            for _ in range(shift):
                opts = opts[1:] + opts[:1]
            it["options"] = opts
            it["correct"] = target
            buckets[target]+=1
        else:
            target = min(buckets, key=buckets.get)
            # se o atual já é o menor, mantém
            if target != cur:
                # rotaciona listado total (até 5)
                L = it["options"]
                idx = letters.index(cur)
                shift = (letters.index(target) - idx) % len(L)
                opts = L[:]
                for _ in range(shift):
                    opts = opts[1:] + opts[:1]
                # corta para 5 máx
                opts = opts[:len(L)]
                it["options"] = opts
                it["correct"] = target
            buckets[it["correct"]] += 1
        balanced.append(it)
    return balanced, buckets

def main():
    raw = load_any(INPUT)
    # normaliza e filtra base
    norm = [normalize_item(x) for x in raw]
    filtered = []
    for it in norm:
        if not it["stem"] or len(it["options"]) < 4: continue
        if is_control_stem(it["stem"]): continue
        if not ban_all_none(it["options"]): continue
        filtered.append(it)

    # dedup
    deduped = dedup(filtered)

    # enriquecer e corrigir
    enriched = []
    for it in deduped:
        fix = enrich_item(it)
        if fix: enriched.append(fix)

    # garantir cobertura de subáreas
    by_sub = defaultdict(list)
    for it in enriched:
        by_sub[it["subarea"]].append(it)
    pool = []
    # coleta alvo mínimo por subárea
    for s in SUBAREAS:
        pool.extend(by_sub.get(s, [])[:max(TARGET_PER_SUBAREA_MIN, 0)])
    # completa até TARGET_ITEMS_TOTAL
    if len(pool) < TARGET_ITEMS_TOTAL:
        rest = [x for x in enriched if x not in pool]
        random.shuffle(rest)
        pool.extend(rest[:max(0, TARGET_ITEMS_TOTAL - len(pool))])
    final = pool if pool else enriched

    # balancear gabaritos
    final, buckets = balance_correct_letters(final)

    # dedup final
    final = dedup(final)

    # escreve SFT
    with open(OUT_SFT, "w", encoding="utf-8") as f:
        for it in final:
            f.write(json.dumps(it, ensure_ascii=False)+"\n")

    # gera pares de preferência (1 ruim por bom)
    pref = []
    for it in final:
        pv = make_bad_variant(it)
        pref.append(pv)

    with open(OUT_PREF, "w", encoding="utf-8") as f:
        for p in pref:
            f.write(json.dumps(p, ensure_ascii=False)+"\n")

    # Relatório
    print(f"[ok] Entrada: {len(raw)} | Após filtro: {len(filtered)} | Dedup: {len(deduped)} | Enriquecidos válidos: {len(enriched)}")
    print(f"[ok] Saída SFT: {len(final)} → {OUT_SFT}")
    print(f"[ok] Saída Pref: {len(pref)} → {OUT_PREF}")
    print("[dist gabaritos]", buckets)
    print("[dist subáreas]", Counter([x['subarea'] for x in final]))
    print("[dist domínios]", Counter([x['domain'] for x in final]))

if __name__ == "__main__":
    main()
