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

def score_based_moderate(text: str, hf_scores: dict) -> dict:
    toxicity = hf_scores.get("toxicity", 0.0)
    threat   = hf_scores.get("threat", 0.0)
    insult   = hf_scores.get("insult", 0.0)
    obscene  = hf_scores.get("obscene", 0.0)
    severe   = hf_scores.get("severe_toxicity", 0.0)
    identity = hf_scores.get("identity_attack", 0.0)

    top = max(toxicity, threat, insult, obscene, severe, identity)

    # REMOVE: only genuinely harmful content
    if severe > 0.4 or threat > 0.7 or (toxicity > 0.85 and identity > 0.5):
        return {"decision": "remove", "confidence": round(min(0.95, top + 0.1), 2),
                "explanation": "Content contains severe toxicity, a credible threat, or targeted hate speech."}

    # REMOVE: very high combined scores
    elif toxicity > 0.85 and insult > 0.85:
        return {"decision": "remove", "confidence": round(top, 2),
                "explanation": "Highly toxic and insulting content that violates community guidelines."}

    # FLAG: mildly toxic or insulting — needs human review
    elif toxicity > 0.6 or insult > 0.7 or top > 0.6:
        return {"decision": "flag", "confidence": round(top, 2),
                "explanation": "Mildly toxic or insulting content. Flagged for human review."}

    # FLAG: borderline
    elif top > 0.4:
        return {"decision": "flag", "confidence": round(top, 2),
                "explanation": "Potentially offensive content detected. Flagged for review."}

    # ALLOW: safe
    else:
        return {"decision": "allow", "confidence": round(1.0 - top, 2),
                "explanation": "Content appears safe with low toxicity scores."}

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
        
    llm_result = score_based_moderate(text, hf_scores)
    
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
