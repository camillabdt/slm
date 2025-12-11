
import os, re, argparse
from typing import List
import argostranslate.package as argos_pkg
import argostranslate.translate as argos_trans

DEFAULT_INPUT = "re_questions.txt"
DEFAULT_OUTPUT = "re_questions_pt.txt"
SRC_LANG = "en"
TGT_LANG = "pt"

CHOICE_PATTERN = re.compile(r"(?m)^([A-D])\)\s*(.*)$")
LABEL_PATTERN  = re.compile(r"(?m)^(Question:|Rationale:|Topic:|Bloom:|Difficulty:|Correct:)\s*")

def ensure_argos_package(src: str, tgt: str):
    # Verifica se já há pacote instalado EN->PT; se não, baixa o oficial
    installed = argos_pkg.get_installed_packages()
    for p in installed:
        if p.from_code.lower() == src and p.to_code.lower() == tgt:
            return
    # Baixar índice e instalar o pacote correto
    argos_pkg.update_package_index()
    available = argos_pkg.get_available_packages()
    for p in available:
        if p.from_code.lower() == src and p.to_code.lower() == tgt:
            path = p.download()  # baixa .argosmodel
            argos_pkg.install_from_path(path)
            return
    raise RuntimeError(f"Pacote Argos {src}->{tgt} não encontrado.")

def split_blocks(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def join_blocks(blocks: List[str]) -> str:
    return "\n\n".join(blocks)

def protect_block(block: str) -> str:
    protected = block
    def _mark_label(m):
        label = m.group(1)[:-1]
        return f"__KEEP_LABEL__{label}__: "
    protected = LABEL_PATTERN.sub(_mark_label, protected)
    def _mark_choice(m):
        letter, text = m.group(1), m.group(2)
        return f"__KEEP_CHOICE__{letter}) __TRANS__{text}__ENDTRANS__"
    protected = CHOICE_PATTERN.sub(_mark_choice, protected)
    protected = re.sub(r"(?m)^__KEEP_LABEL__Correct__:\s*([A-D])\s*$",
                       r"__KEEP_LABEL__Correct__: __KEEP_LETTER__\1", protected)
    def _add_markers(line: str) -> str:
        for base in ["Question", "Rationale", "Topic", "Bloom", "Difficulty"]:
            prefix = f"__KEEP_LABEL__{base}__:"
            if line.startswith(prefix):
                rest = line[len(prefix):].lstrip()
                return f"{prefix} __TRANS__{rest}__ENDTRANS__"
        return line
    return "\n".join(_add_markers(ln) for ln in protected.splitlines())

def unprotect_block(block: str) -> str:
    restored = block.replace("__TRANS__", "").replace("__ENDTRANS__", "")
    for base in ["Question", "Rationale", "Topic", "Bloom", "Difficulty", "Correct"]:
        restored = restored.replace(f"__KEEP_LABEL__{base}__:", f"{base}:")
    restored = re.sub(r"__KEEP_CHOICE__([A-D])\)", r"\1)", restored)
    restored = re.sub(r"__KEEP_LETTER__([A-D])", r"\1", restored)
    return re.sub(r"[ \t]+", " ", restored).strip()

def translate_text(text: str) -> str:
    # traduz só os trechos marcados com __TRANS__...__ENDTRANS__
    out_lines = []
    for line in text.splitlines():
        if "__TRANS__" in line:
            def repl(m):
                inner = m.group(1)
                return argos_trans.translate(inner, SRC_LANG, TGT_LANG)
            line = re.sub(r"__TRANS__(.*?)__ENDTRANS__", lambda m: repl(m), line)
        out_lines.append(line)
    return "\n".join(out_lines)

def main():
    ap = argparse.ArgumentParser("Tradução TXT→TXT com Argos Translate (sem chave).")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT)
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # Garante que o pacote EN->PT esteja instalado (baixa 1x se faltar)
    ensure_argos_package(SRC_LANG, TGT_LANG)

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = split_blocks(content)
    if not blocks:
        print("Nenhum bloco encontrado no arquivo de entrada."); return

    protected = [protect_block(b) for b in blocks]
    translated_protected = [translate_text(b) for b in protected]
    restored = [unprotect_block(b) for b in translated_protected]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(join_blocks(restored))

    print(f"Tradução concluída: {len(blocks)} bloco(s). Saída: {args.output}")

if __name__ == "__main__":
    main()
