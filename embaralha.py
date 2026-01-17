import argparse
import json
import random
from pathlib import Path
from collections import Counter

def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {i}: invalid JSON ({e})")
    return items

def write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def ensure_triplet_messages(obj):
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        raise ValueError(f"Invalid messages for id={obj.get('id')}")
    # pega o primeiro system, user, assistant na ordem
    sys = next((m for m in msgs if m.get("role") == "system"), None)
    user = next((m for m in msgs if m.get("role") == "user"), None)
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if not sys or not user or not asst:
        raise ValueError(f"Missing roles in messages for id={obj.get('id')}")
    obj["messages"] = [sys, user, asst]
    return obj

def mcq_assistant_content_to_string(obj):
    # Se for MCQ e assistant.content for dict, converte para string JSON
    if obj.get("task") != "mcq":
        return obj
    msgs = obj["messages"]
    asst = msgs[2]
    content = asst.get("content")
    if isinstance(content, dict):
        asst["content"] = json.dumps(content, ensure_ascii=False)
    return obj

def summarize(items, title):
    c_task = Counter()
    c_topic = Counter()
    c_diff = Counter()
    for obj in items:
        c_task[obj.get("task")] += 1
        c_topic[obj.get("topic")] += 1
        c_diff[obj.get("difficulty")] += 1
    print(f"\n== {title} ==")
    print("Total:", len(items))
    print("By task:", dict(c_task))
    print("By difficulty:", dict(c_diff))
    print("Top topics:", dict(c_topic.most_common(10)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files (one or more)")
    ap.add_argument("--out_dir", default="dataset_out", help="Output directory")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (e.g., 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--fix_mcq", action="store_true", help="Convert MCQ assistant.content dict -> JSON string")
    args = ap.parse_args()

    items = []
    for p in map(Path, args.inputs):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        items.extend(read_jsonl(p))

    # valida formato básico + (opcional) normaliza MCQ
    cleaned = []
    for obj in items:
        for k in ["id", "task", "topic", "difficulty", "messages"]:
            if k not in obj:
                raise ValueError(f"Missing key '{k}' in item id={obj.get('id')}")
        obj = ensure_triplet_messages(obj)
        if args.fix_mcq:
            obj = mcq_assistant_content_to_string(obj)
        cleaned.append(obj)

    random.seed(args.seed)
    random.shuffle(cleaned)

    n = len(cleaned)
    n_val = int(round(n * args.val_ratio))
    val = cleaned[:n_val]
    train = cleaned[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    summarize(cleaned, "FULL (shuffled)")
    summarize(train, "TRAIN")
    summarize(val, "VAL")

    print("\nSaved:")
    print("-", out_dir / "train.jsonl")
    print("-", out_dir / "val.jsonl")

if __name__ == "__main__":
    main()
