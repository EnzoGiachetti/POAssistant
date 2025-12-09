import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI(title="Story-4D Enterprise API")

# Configuração CORS (Permitir acesso do frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção na empresa, restrinja para o domínio correto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURAÇÃO CORPORATIVA ---
API_URL = os.getenv("url_interna")
API_KEY = os.getenv("key_interna")
MODEL_NAME = os.getenv("GPT_MODEL", "gpt-3.5-turbo") # Fallback se não definido

if not API_KEY or not API_URL:
    print("⚠️ AVISO: Variáveis de ambiente 'url_interna' ou 'key_interna' não configuradas.")

# --- FUNÇÃO AUXILIAR PARA GPT INTERNO ---
def call_internal_gpt(system_prompt: str, user_prompt: str):
    """
    Função genérica para chamar a API interna via requests.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Chave de API interna não configurada.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Estrutura padrão OpenAI (a maioria das APIs internas segue esse padrão)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }

    try:
        print(f"📡 Enviando requisição para: {MODEL_NAME}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extração da resposta (Adapte se o JSON da sua empresa for diferente)
        # Padrão OpenAI: choices[0].message.content
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Limpeza de Markdown (caso a IA devolva ```json ... ```)
        if content.strip().startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        elif content.strip().startswith("```"):
            content = content.replace("```", "")
            
        return json.loads(content.strip())

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de Conexão: {e}")
        raise HTTPException(status_code=502, detail="Erro de comunicação com a API interna.")
    except json.JSONDecodeError:
        print(f"❌ Erro de Parse JSON. Conteúdo recebido: {content}")
        raise HTTPException(status_code=500, detail="A IA não retornou um JSON válido.")
    except Exception as e:
        print(f"❌ Erro genérico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- MODELOS DE DADOS (PYDANTIC) ---
# Isso garante validação automática dos dados que chegam do Front
class StoryRequest(BaseModel):
    idea: str
    user: Optional[str] = ""
    action: Optional[str] = ""
    benefit: Optional[str] = ""

class QuestionRequest(BaseModel):
    context: str

class ChatRequest(BaseModel):
    context: str
    message: str

class QAPair(BaseModel):
    question: str
    answer: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ConsolidateRequest(BaseModel):
    context: str
    qaPairs: List[QAPair]
    chatHistory: List[ChatMessage] = []

class Rule(BaseModel):
    id: str
    text: str

class GherkinRequest(BaseModel):
    rules: List[Rule]

class Scenario(BaseModel):
    ruleId: str
    gherkinText: str

class ValidateRequest(BaseModel):
    story: str
    rules: List[Rule]
    scenarios: List[Scenario]

# --- ROTAS (ENDPOINTS) ---

@app.post("/api/generate-story")
def generate_story(req: StoryRequest):
    system = """
    Você é um Product Owner experiente (PO Coach). 
    Sua tarefa é criar ou refinar uma User Story baseada em inputs parciais.
    INSTRUÇÃO DE REFINAMENTO:
    Analise o rascunho. Se o verbo for passivo, torne-o acionável. Se o benefício for vago, torne-o específico.
    Retorne APENAS JSON:
    {
        "user": "string",
        "action": "string",
        "benefit": "string",
        "formattedStory": "HTML string com tags <strong>"
    }
    """
    prompt = f"Ideia: {req.idea}\nUsuário: {req.user}\nAção: {req.action}\nBenefício: {req.benefit}"
    return call_internal_gpt(system, prompt)

@app.post("/api/generate-questions")
def generate_questions(req: QuestionRequest):
    system = """
    Atue como um Analista de Negócios Sênior e Especialista em QA.
    Gere de 3 a 5 perguntas estratégicas para "blindar" essa funcionalidade (Exceção, Validação, Segurança).
    Retorne APENAS JSON:
    { "questions": [{ "label": "Pergunta?", "ph": "Exemplo" }] }
    """
    return call_internal_gpt(system, f"Contexto: {req.context}")

@app.post("/api/chat-rules")
def chat_rules(req: ChatRequest):
    system = """
    Você é um PO Coach. Reconheça a regra informada e pergunte se há mais. Seja breve.
    Retorne JSON: { "reply": "..." }
    """
    prompt = f"Contexto: {req.context}\nUsuário disse: {req.message}"
    return call_internal_gpt(system, prompt)

@app.post("/api/consolidate-rules")
def consolidate_rules(req: ConsolidateRequest):
    qa_text = "\n".join([f"P: {qa.question} R: {qa.answer}" for qa in req.qaPairs])
    chat_text = "\n".join([f"{msg.role}: {msg.content}" for msg in req.chatHistory])
    
    system = """
    Arquiteto de Software. Consolidar TUDO (respostas + chat) em Regras Formais.
    Retorne JSON: { "rules": [{"id": "RN-01", "text": "..."}] }
    """
    prompt = f"Contexto: {req.context}\nQuestionário:\n{qa_text}\nChat:\n{chat_text}"
    return call_internal_gpt(system, prompt)

@app.post("/api/generate-gherkin")
def generate_gherkin(req: GherkinRequest):
    # Converte Pydantic models para dict para serializar JSON
    rules_dict = [r.dict() for r in req.rules]
    
    system = """
    QA BDD. Crie cenários Gherkin em PT-BR.
    Use tags HTML <span> com classes: .gherkin-keyword, .gherkin-variable, .gherkin-string.
    Retorne JSON: { "scenarios": [{"ruleId": "...", "originalRule": "...", "gherkinText": "..."}] }
    """
    return call_internal_gpt(system, f"Regras: {json.dumps(rules_dict)}")

@app.post("/api/validate-story")
def validate_story(req: ValidateRequest):
    rules_dict = [r.dict() for r in req.rules]
    scenarios_dict = [s.dict() for s in req.scenarios]
    
    system = """
    Agile Coach especialista em INVEST.
    1. Dê nota 0-100.
    2. Se nota < 70 ou complexa, 'isLarge': true e sugira splittings.
    Retorne JSON: 
    { "score": int, "message": "str", "isLarge": bool, "splittingSuggestions": [{"type": "...", "title": "...", "description": "..."}] }
    """
    prompt = f"Story: {req.story}\nRules: {json.dumps(rules_dict)}\nScenarios: {json.dumps(scenarios_dict)}"
    return call_internal_gpt(system, prompt)

if __name__ == "__main__":
    import uvicorn
    # Roda o servidor na porta 5000 para manter compatibilidade com seu HTML atual
    print("🚀 Servidor Corporativo Rodando na porta 5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)
```

### Como Rodar com FastAPI

Diferente do Flask (que usamos `python app.py`), o padrão do FastAPI é usar o servidor `uvicorn`, embora o bloco `if __name__` que coloquei no final permita rodar com `python main.py` também.

No terminal:
```bash
python3 main.py
