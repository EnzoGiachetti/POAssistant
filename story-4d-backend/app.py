import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)
CORS(app)

# CONFIGURAÇÃO GEMINI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ ERRO: GEMINI_API_KEY não encontrada no arquivo .env")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Função auxiliar para chamar o Gemini
def call_gemini(system_instruction, user_prompt):
    try:
        # Configura o modelo para responder SEMPRE em JSON
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )

        response = model.generate_content(user_prompt)
        
        # O Gemini já retorna o texto, basta fazer o parse
        return json.loads(response.text)
    except Exception as e:
        print(f"Erro na API Gemini: {e}")
        return None

# --- ROTA 1: GERAR HISTÓRIA ---
@app.route('/api/generate-story', methods=['POST'])
def generate_story():
    data = request.json
    idea = data.get('idea', '')
    user = data.get('user', '')
    action = data.get('action', '')
    benefit = data.get('benefit', '')

    system = """
    Você é um Product Owner experiente (PO Coach).
    Sua tarefa é criar ou refinar uma User Story.
    Retorne JSON com chaves: user, action, benefit, formattedStory.
    """
    
    prompt = f"""
    Ideia: {idea}
    Usuário sugerido: {user}
    Ação sugerida: {action}
    Benefício sugerido: {benefit}
    
    Analise e refine verbos passivos. Gere o rascunho canônico.
    """

    result = call_gemini(system, prompt)
    if result: return jsonify(result)
    return jsonify({"error": "Erro no Gemini"}), 500

# --- ROTA 2: GERAR PERGUNTAS ---
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    context = data.get('context', '')

    system = """
    Você é um Analista de Negócios. Extraia elementos para regras de negócio (Gatilho, Conteúdo, Dependências, Autorização).
    Retorne JSON: { "questions": [{"label": "...", "ph": "..."}] }
    """

    result = call_gemini(system, f"Contexto da História: {context}")
    if result: return jsonify(result['questions'])
    return jsonify({"error": "Erro no Gemini"}), 500

# --- ROTA CONSOLIDAR REGRAS ---
@app.route('/api/consolidate-rules', methods=['POST'])
def consolidate_rules():
    data = request.json
    context = data.get('context', '')
    qa_pairs = data.get('qaPairs', [])
    qa_text = "\n".join([f"P: {i['question']} R: {i['answer']}" for i in qa_pairs])

    system = """
    Transforme o contexto e respostas em Regras de Negócio formais.
    Retorne JSON: { "rules": [{"id": "RN-XX", "text": "..."}] }
    """
    
    result = call_gemini(system, f"História: {context}\nEntrevista: {qa_text}")
    if result: return jsonify(result['rules'])
    return jsonify({"error": "Erro no Gemini"}), 500

# --- ROTA 3: GERAR GHERKIN ---
@app.route('/api/generate-gherkin', methods=['POST'])
def generate_gherkin():
    data = request.json
    rules = data.get('rules', [])

    system = """
    QA Engineer especialista em BDD. Para CADA regra, escreva um cenário Gherkin em PT-BR.
    Use tags HTML <span> com classes .gherkin-keyword, .gherkin-variable, .gherkin-string.
    Retorne JSON: { "scenarios": [{"ruleId": "...", "originalRule": "...", "gherkinText": "..."}] }
    """

    result = call_gemini(system, f"Regras: {json.dumps(rules)}")
    if result: return jsonify(result['scenarios'])
    return jsonify({"error": "Erro no Gemini"}), 500

# --- ROTA 4: VALIDAR ---
@app.route('/api/validate-story', methods=['POST'])
def validate_story():
    data = request.json
    story = data.get('story', '')
    rules = data.get('rules', [])
    scenarios = data.get('scenarios', [])

    system = """
    Agile Coach especialista em INVEST.
    Analise qualidade (0-100). Se score < 70 ou 'Grande', sugira splitting (Indicador, Fluxo, Usuário).
    Retorne JSON: { "score": int, "message": str, "isLarge": bool, "splittingSuggestions": [...] }
    """

    prompt = f"Story: {story}\nRules: {json.dumps(rules)}\nScenarios: {json.dumps(scenarios)}"
    
    result = call_gemini(system, prompt)
    if result: return jsonify(result)
    return jsonify({"error": "Erro no Gemini"}), 500

if __name__ == '__main__':
    print("🚀 Servidor Story-4D (Gemini Edition) rodando na porta 5000")
    app.run(debug=True, port=5000)
