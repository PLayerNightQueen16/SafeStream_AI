from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OpenAI API key found")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tasks that match openenv.yaml exactly
TASKS = [
    {"id": "task_1", "text": "I hate you", "label": "remove"},
    {"id": "task_2", "text": "You are amazing", "label": "allow"},
    {"id": "task_3", "text": "Wow you're so smart 🙄", "label": "flag"},
]

current_task_idx = 0

class MyEnvV4Action(BaseModel):
    message: str

class Observation(BaseModel):
    echoed_message: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool

class ResetResponse(BaseModel):
    observation: Observation
    done: bool

@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request):
    global current_task_idx
    current_task_idx = 0
    return ResetResponse(
        observation=Observation(echoed_message=TASKS[current_task_idx]["text"]),
        done=False,
    )

@app.post("/step", response_model=StepResponse)
async def step(request: Request):
    global current_task_idx
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    msg = ""
    if "action" in body and isinstance(body["action"], dict) and "message" in body["action"]:
        msg = body["action"]["message"]
    elif "message" in body:
        msg = body["message"]

    true_label = TASKS[current_task_idx]["label"]
    reward = 1.0 if msg.lower().strip() == true_label.lower() else 0.0

    current_task_idx += 1
    done = current_task_idx >= len(TASKS)

    next_text = TASKS[current_task_idx]["text"] if not done else ""

    return StepResponse(
        observation=Observation(echoed_message=next_text),
        reward=reward,
        done=done,
    )

@app.get("/state")
async def state():
    done = current_task_idx >= len(TASKS)
    next_text = TASKS[current_task_idx]["text"] if not done else ""
    return {
        "observation": {"echoed_message": next_text},
        "done": done
    }

class ModerationRequest(BaseModel):
    text: str

from groq import Groq

def groq_moderate(text: str, hf_scores: dict) -> dict:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    relevant_keys = ["toxicity", "severe_toxicity", "insult", "threat", "obscene", "identity_attack"]
    filtered_scores = {k: round(hf_scores.get(k, 0.0), 3) for k in relevant_keys if k in hf_scores}

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": """You are an expert content moderation AI.

You will receive text and toxicity scores (0.0-1.0) from a RoBERTa model.

Make a decision based on FULL CONTEXT and INTENT — not just keywords. Consider:
- Sarcasm or dark humour that looks toxic but isn't harmful
- Context that changes meaning ("I'll destroy you at chess" is fine)
- Whether content genuinely targets a person harmfully
- Mild insults like "idiot" or "stupid" should be FLAG not REMOVE

Respond ONLY with valid JSON, no markdown:
{"decision": "allow" or "flag" or "remove", "confidence": <0.0-1.0>, "explanation": "<1 sentence>"}

allow  = safe content
flag   = mildly toxic, rude, or ambiguous
remove = genuine hate speech, real threats, severe harassment"""},
            {"role": "user", "content": f'Text: "{text}"\nScores: {json.dumps(filtered_scores)}\nModerate this.'}
        ],
        temperature=0.1,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    result["decision"] = result.get("decision", "flag").lower()
    if result["decision"] not in ("allow", "flag", "remove"):
        result["decision"] = "flag"
    result["confidence"] = min(max(float(result.get("confidence", 0.5)), 0.0), 1.0)
    result["explanation"] = result.get("explanation", "No explanation provided.")
    return result

@app.post("/moderate")
def moderate(request: ModerationRequest):
    text = request.text.strip()
    
    # Fast skip validation
    if not text:
        return {
            "decision": "allow",
            "confidence": 1.0,
            "explanation": "Empty input provides no context for moderation.",
            "ai_scores": {
                "toxicity": 0.0,
                "insult": 0.0,
                "threat": 0.0,
                "obscene": 0.0
            }
        }
    
    # Stage 1: Lazy load and classify using HuggingFace RoBERTa 
    try:
        from app.models.toxicity_model import predict_toxicity
        hf_scores = predict_toxicity(text)
    except Exception as e:
        hf_scores = {}
        
    llm_result = groq_moderate(text, hf_scores)
    
    ai_scores = {
        "toxicity": round(hf_scores.get("toxicity", 0.0), 3),
        "insult":   round(hf_scores.get("insult", 0.0), 3),
        "threat":   round(hf_scores.get("threat", 0.0), 3),
        "obscene":  round(hf_scores.get("obscene", 0.0), 3),
    }
    
    return {
        "decision": llm_result["decision"],
        "confidence": llm_result["confidence"],
        "explanation": llm_result["explanation"],
        "ai_scores": ai_scores
    }

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "app", "frontend")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

try:
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
except:
    pass

@app.get("/")
def serve_ui():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"status": "ok"}

if __name__ == "__main__":
    main()
