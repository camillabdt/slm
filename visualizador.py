import json
import re
import os

def processar_texto_sujo(texto):
    """Extrai os dados mesmo quando o modelo repete palavras ou erra o JSON."""
    # 1. Tentar capturar o primeiro JSON válido se existir
    json_match = re.search(r'\{.*\}', texto, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get('stem', ''), data.get('options', []), data.get('correctOption', ''), data.get('explanation', '')
        except: pass

    # 2. Se for texto puro (estilo Question/Option/Correct Answer)
    # Limpa repetições comuns que vimos no seu arquivo
    stem = re.search(r'(?:Question|stem):\s*(.*?)(?=\n|Option|0\)|$)', texto, re.I)
    stem = stem.group(1).strip() if stem else "Analise o conteúdo técnico abaixo:"
    
    # Busca opções começando com números (0, 1, 2) ou letras (A, B)
    options = re.findall(r'(?:[0-4A-D][\)\-\.])\s*(.*?)(?=\n|$)', texto)
    
    correct = re.search(r'(?:Correct Answer|Ans|Correct):\s*(.*)', texto, re.I)
    correct = correct.group(1).strip() if correct else "Ver explicação"
    
    # Se houver múltiplas explicações, pega a primeira significativa
    explanation = re.search(r'Explanation:\s*(.*?)(?=Question|$)', texto, re.I | re.S)
    explanation = explanation.group(1).strip() if explanation else texto

    return stem, options, correct, explanation

def gerar_html():
    file_input = "banco_final_seguro.jsonl"
    file_output = "simulado_bncc_corrigido.html"
    
    if not os.path.exists(file_input):
        print(f"Erro: O arquivo {file_input} não foi encontrado!")
        return

    html_head = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Simulado BNCC 9º Ano - Computação</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f0f2f5; padding: 40px; color: #1c1e21; }
            .container { max-width: 800px; margin: auto; }
            h1 { text-align: center; color: #1877f2; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            .card { background: white; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-top: 8px solid #007bff; }
            .pilar { font-size: 11px; font-weight: bold; color: #65676b; text-transform: uppercase; margin-bottom: 10px; }
            .stem { font-size: 18px; font-weight: 600; line-height: 1.4; margin-bottom: 20px; }
            .option { background: #f7f8fa; border: 1px solid #ebedf0; padding: 12px; margin: 8px 0; border-radius: 8px; cursor: default; }
            .answer-btn { background: #42b72a; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; margin-top: 15px; }
            .answer-content { display: none; margin-top: 20px; padding: 15px; background: #fff9e6; border-left: 4px solid #ffba00; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Banco de Questões BNCC (9º Ano)</h1>
    """

    html_cards = ""
    with open(file_input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            stem, opts, corr, expl = processar_texto_sujo(data['output'])
            
            # Define cor por pilar (Habilidades BNCC)
            color = "#007bff" if "Pensamento" in data['pilar'] else "#f39c12" if "Mundo" in data['pilar'] else "#27ae60"
            
            html_cards += f"""
            <div class="card" style="border-top-color: {color}">
                <div class="pilar">{data['pilar']} | {data['subtema']} | Técnica: {data['tecnica']}</div>
                <div class="stem">{i+1}. {stem}</div>
                <div class="options">
            """
            for opt in opts:
                html_cards += f'<div class="option">{opt}</div>'
            
            html_cards += f"""
                </div>
                <button class="answer-btn" onclick="this.nextElementSibling.style.display='block'">Ver Gabarito</button>
                <div class="answer-content">
                    <strong>Gabarito:</strong> {corr}<br><br>
                    <strong>Explicação:</strong> {expl}
                </div>
            </div>
            """

    html_footer = "</div></body></html>"
    
    with open(file_output, "w", encoding="utf-8") as f:
        f.write(html_head + html_cards + html_footer)
    
    print(f"✅ Sucesso! O arquivo '{file_output}' foi criado.")

if __name__ == "__main__":
    gerar_html()