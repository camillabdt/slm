import os, json, re, random
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ==========================================
# 1) Configurações
# ==========================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "tinyllama_cpu_english"
max_steps = 120
max_length = 256

NUM_OPTIONS = 4
LETTERS = ["A", "B", "C", "D"][:NUM_OPTIONS]

# geração (diversidade)
GEN_TEMPERATURE = 0.95
GEN_TOP_P = 0.92
GEN_TOP_K = 50
GEN_REP_PENALTY = 1.12
GEN_NO_REPEAT_NGRAM = 3
MAX_NEW_TOKENS = 220

# ==========================================
# 2) Dados dummy se não existir
# ==========================================
if not os.path.exists("questions.json"):
    print("Arquivo 'questions.json' não encontrado. Criando dados de exemplo...")
    dummy_data = [
        {
            "stem": "What characterizes a performance requirement?",
            "options": ["Security aspect", "Speed and latency", "User interface", "Database schema"],
            "correctOption": 1,
            "explanation": "Performance requirements define how fast the system performs."
        },
        {
            "stem": "Which of these is a Functional Requirement?",
            "options": ["The system shall run on iOS.", "The system shall calculate tax.", "The system shall be available 99% of time.", "The code must be written in Python."],
            "correctOption": 1,
            "explanation": "Functional requirements describe behaviors or functions."
        }
    ]
    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, ensure_ascii=False, indent=2)

with open("questions.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ==========================================
# 3) Tokenizer + Modelo (CPU)
# ==========================================
print(f"Carregando modelo {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

# ==========================================
# 4) LoRA
# ==========================================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
print("LoRA aplicado.")

# ==========================================
# 5) Dataset (treino para seguir formato, mas pedindo VARIAÇÃO)
# ==========================================
def build_train_examples(data):
    idx_to_letter = {i: LETTERS[i] for i in range(len(LETTERS))}
    ex = []
    for item in data:
        options = item["options"][:NUM_OPTIONS]
        correct_idx = int(item["correctOption"])
        correct_letter = idx_to_letter.get(correct_idx, "A")

        prompt = (
            "### Instruction:\n"
            "Generate a NEW multiple-choice question about Software Engineering.\n"
            f"- Provide exactly {NUM_OPTIONS} options labeled {', '.join(LETTERS)}.\n"
            "- Use different wording than the examples.\n"
            "- Do not copy stems or options from the training data.\n"
            "Return strictly in this format:\n"
            "Stem: <text>\n"
            "Options:\n"
            "A) ...\n"
            "B) ...\n"
            "...\n"
            "Correct Answer: <LETTER>\n"
            "Explanation: <text>\n\n"
            "### Response:\n"
        )

        options_text = ""
        for i, opt in enumerate(options):
            options_text += f"{LETTERS[i]}) {opt}\n"

        completion = (
            f"Stem: {item['stem']}\n\n"
            f"Options:\n{options_text}\n"
            f"Correct Answer: {correct_letter}\n"
            f"Explanation: {item['explanation']}{tokenizer.eos_token}"
        )

        ex.append({"text": prompt + completion})
    return ex

dataset = Dataset.from_list(build_train_examples(raw_data))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

tokenized = dataset.map(tokenize_function, batched=True)

# ==========================================
# 6) Treino
# ==========================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=max_steps,
    learning_rate=2e-4,
    logging_steps=10,
    use_cpu=True,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\n" + "="*30)
print(f"Treinando na CPU por {max_steps} passos...")
print("="*30 + "\n")
trainer.train()

# ==========================================
# 7) Geração + parsing + filtro
# ==========================================
def make_prompt(topic: str):
    return (
        "### Instruction:\n"
        f"Generate a NEW multiple-choice question about {topic}.\n"
        f"- Provide exactly {NUM_OPTIONS} options labeled {', '.join(LETTERS)}.\n"
        "- Make options plausible and distinct.\n"
        "- Avoid repeating wording.\n"
        "Return strictly in this format:\n"
        "Stem: <text>\n"
        "Options:\n"
        "A) ...\n"
        "B) ...\n"
        "...\n"
        "Correct Answer: <LETTER>\n"
        "Explanation: <text>\n\n"
        "### Response:\n"
        "Stem:"
    )

def parse_question(txt: str):
    if "### Response:" in txt:
        txt = txt.split("### Response:", 1)[1].strip()

    stem_m = re.search(r"Stem:\s*(.+)", txt)
    stem = stem_m.group(1).strip() if stem_m else ""

    options = []
    for L in LETTERS:
        m = re.search(rf"^{re.escape(L)}\)\s*(.+)$", txt, flags=re.MULTILINE)
        options.append(m.group(1).strip() if m else "")

    ca_m = re.search(r"Correct Answer:\s*([A-D])", txt)
    correct = ca_m.group(1).strip() if ca_m else ""

    exp_m = re.search(r"Explanation:\s*(.+)", txt)
    explanation = exp_m.group(1).strip() if exp_m else ""

    return {"stem": stem, "options": options, "correct": correct, "explanation": explanation, "raw": txt}

def is_good(q, seen_stems):
    if not q["stem"] or len(q["stem"]) < 12:
        return False
    if any((not o) or len(o) < 3 for o in q["options"]):
        return False
    if len(set(o.lower() for o in q["options"])) < len(q["options"]):
        return False
    if q["correct"] not in LETTERS:
        return False
    if not q["explanation"] or len(q["explanation"]) < 12:
        return False

    norm = re.sub(r"\s+", " ", q["stem"].strip().lower())
    if norm in seen_stems:
        return False

    if "Return strictly in this format" in q["raw"]:
        return False

    return True

def generate_one(topic, seed=None):
    if seed is None:
        seed = random.randint(1, 10_000_000)
    torch.manual_seed(seed)

    prompt = make_prompt(topic)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REP_PENALTY,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return parse_question(text)

def print_question(q):
    print(f"Stem: {q['stem']}\n")
    print("Options:")
    for i, opt in enumerate(q["options"]):
        print(f"{LETTERS[i]}) {opt}")
    print(f"\nCorrect Answer: {q['correct']}")
    print(f"Explanation: {q['explanation']}\n")

# ==========================================
# 8) Gerar lote (só imprimir)
# ==========================================
topics = [
    "Non-Functional Requirements",
    "Functional Requirements",
    "Software Testing",
    "UML Use Case Diagrams",
    "Agile Scrum",
    "SOLID principles",
    "Design Patterns",
    "Version control with Git",
    "CI/CD basics",
    "Software architecture (MVC, layered architecture)",
]

N = 10
MAX_TRIES = 8

model.eval()
seen_stems = set()

print("\n" + "="*30)
print(f"Gerando {N} questões (sem salvar)...")
print("="*30 + "\n")

for i in range(N):
    q_ok = None
    for _ in range(MAX_TRIES):
        topic = random.choice(topics)
        q = generate_one(topic)
        if is_good(q, seen_stems):
            q_ok = q
            break

    if q_ok is None:
        q_ok = q  # imprime mesmo que não passou no filtro

    norm = re.sub(r"\s+", " ", q_ok["stem"].strip().lower())
    if q_ok["stem"]:
        seen_stems.add(norm)

    print(f"===== Question {i+1}/{N} =====")
    print_question(q_ok)
