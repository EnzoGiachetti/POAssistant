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

# --- CONFIGURAÇÃO GEMINI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("\n⚠️  ERRO CRÍTICO: GEMINI_API_KEY não encontrada!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Chave carregada com sucesso.")

def call_gemini(system_instruction, user_prompt):
    """
    Função robusta: Tenta usar o modelo Flash, se falhar, tenta o Pro.
    """
    if not GEMINI_API_KEY:
        return {"error": "Chave de API não configurada."}

    # Lista de modelos para tentar (na ordem de prioridade)
    models_to_try = ["gemini-1.5-flash", "gemini-pro", "gemini-2.5-flash"]
    
    last_error = None

    for model_name in models_to_try:
        try:
            print(f"📡 Tentando usar o modelo: {model_name}...")
            
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config={"response_mime_type": "application/json"}
            )

            response = model.generate_content(user_prompt)
            print(f"✅ Sucesso com {model_name}!")
            
            return json.loads(response.text)

        except Exception as e:
            print(f"⚠️ Falha com {model_name}: {e}")
            last_error = e
            continue # Tenta o próximo da lista
    
    print(f"❌ Todas as tentativas falharam. Erro final: {last_error}")
    return {"error": str(last_error)}

# --- ROTA 1: GERAR HISTÓRIA (Prompt Rico Restaurado) ---
@app.route('/api/generate-story', methods=['POST'])
def generate_story():
    data = request.json
    idea = data.get('idea', '')
    user = data.get('user', '')
    action = data.get('action', '')
    benefit = data.get('benefit', '')

    system = """
    Você é um Product Owner experiente (PO Coach). 
    Sua tarefa é criar ou refinar uma User Story baseada em inputs parciais.

    INSTRUÇÃO DE REFINAMENTO:
    Analise o rascunho gerado a partir da ideia e dos inputs. 
    Se o verbo da ação for passivo (ex: 'visualizar', 'ver') ou o valor for vago (ex: 'facilitar', 'melhorar'),
    sugira um refinamento no verbo e na frase 'para que' para torná-los mais acionáveis, específicos e orientados a valor.

    O objetivo é gerar o rascunho canônico de alta qualidade: "Como um [Persona], eu quero [Ação], para que [Valor]."

    Retorne APENAS um JSON no formato:
    {
        "user": "string (quem é o usuário)",
        "action": "string (o que ele quer fazer - refinado se necessário)",
        "benefit": "string (para que - refinado se necessário)",
        "formattedStory": "string (HTML formatado com tags <strong> para Como, Quero, Para)"
    }
    Se algum campo estiver vazio, infira o melhor conteúdo baseado no campo 'idea', aplicando as regras de refinamento acima.
    """
    
    prompt = f"""
    Ideia Bruta: {idea}
    Usuário sugerido: {user}
    Ação sugerida: {action}
    Benefício sugerido: {benefit}
    """

    result = call_gemini(system, prompt)
    if result and "error" in result: return jsonify(result), 400
    if result: return jsonify(result)
    return jsonify({"error": "Falha na IA"}), 500

# --- ROTA 2: PERGUNTAS ---
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    context = data.get('context', '')
    system = """
    Atue como um Analista de Negócios Sênior e Especialista em QA.
    
    Sua missão é analisar o Contexto da História de Usuário e identificar lacunas, riscos e regras não explícitas.
    Gere perguntas estratégicas para "blindar" essa funcionalidade tente identificar pontos que não foram mencionados anteriormente.

    Retorne JSON: { "questions": [{"label": "...", "ph": "..."}] }
    """
    result = call_gemini(system, f"Contexto: {context}")
    if result and "error" in result: return jsonify(result), 400
    if result: return jsonify(result['questions'])
    return jsonify({"error": "Falha na IA"}), 500

# --- ROTA CHAT ---
@app.route('/api/chat-rules', methods=['POST'])
def chat_rules():
    data = request.json
    context = data.get('context', '')
    user_message = data.get('message', '')
    
    system = """
    Você é um PO Coach. O usuário está adicionando regras manualmente.
    Reconheça a regra e pergunte se há mais.
    Retorne JSON: { "reply": "..." }
    """
    
    prompt = f"Contexto: {context}\nUsuário disse: {user_message}"
    result = call_gemini(system, prompt)
    if result and "error" in result: return jsonify(result), 400
    if result: return jsonify(result)
    return jsonify({"error": "Falha na IA"}), 500

# --- ROTA 2b: CONSOLIDAR ---
@app.route('/api/consolidate-rules', methods=['POST'])
def consolidate_rules():
    data = request.json
    context = data.get('context', '')
    qa_pairs = data.get('qaPairs', [])
    chat_history = data.get('chatHistory', [])
    
    qa_text = "\n".join([f"P: {i['question']} R: {i['answer']}" for i in qa_pairs])
    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    system = """
    Converta tudo (Quiz + Chat) em Regras Formais.
    Retorne JSON: { "rules": [{"id": "RN-01", "text": "..."}] }
    """
    result = call_gemini(system, f"História: {context}\nQuiz: {qa_text}\nChat: {chat_text}")
    if result and "error" in result: return jsonify(result), 400
    if result: return jsonify(result['rules'])
    return jsonify({"error": "Falha na IA"}), 500

# --- ROTA 3: GHERKIN ---
@app.route('/api/generate-gherkin', methods=['POST'])
def generate_gherkin():
    data = request.json
    rules = data.get('rules', [])
    system = """
    QA BDD. Crie cenários Gherkin. Use tags HTML <span> com classes .gherkin-keyword, .gherkin-variable.
    Retorne JSON: { "scenarios": [{"ruleId": "...", "originalRule": "...", "gherkinText": "..."}] }
    """
    result = call_gemini(system, f"Regras: {json.dumps(rules)}")
    if result and "error" in result: return jsonify(result), 400
    if result: return jsonify(result['scenarios'])
    return jsonify({"error": "Falha na IA"}), 500

# --- ROTA 4: VALIDAR ---
@app.route('/api/validate-story', methods=['POST'])
def validate_story():
    data = request.json
    story = data.get('story', '')
    rules = data.get('rules', [])
    scenarios = data.get('scenarios', [])
    system = """
    Agile Coach. Avalie (0-100) e sugira splitting se necessario.
    Retorne JSON: 
    { "score": int, "message": "str", "isLarge": bool, "splittingSuggestions": [{"type": "...", "title": "...", "description": "..."}] }
    """
    prompt = f"Story: {story}\nRules: {len(rules)}\nScenarios: {len(scenarios)}"
    result = call_gemini(system, prompt)
    if result and "error" in result: return jsonify(result), 400
    if result: return jsonify(result)
    return jsonify({"error": "Falha na IA"}), 500

if __name__ == '__main__':
    print("🚀 Servidor Story-4D rodando em http://localhost:5000")
    app.run(debug=True, port=5000)
