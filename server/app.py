from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

import json
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

def openai_moderate(text: str, hf_scores: dict) -> dict:
    system_prompt = """You are an expert content moderation AI.

You will receive:
1. The original text submitted by a user
2. Toxicity scores (0.0-1.0) from a HuggingFace RoBERTa model across harm categories

Your job is to make a final moderation decision based on the FULL CONTEXT and INTENT of the text -- not just individual words or scores. Consider:
- Sarcasm, irony, or dark humour that may look toxic but is not genuinely harmful
- Coded language or subtle threats that low scores might miss
- Context that changes meaning (e.g. "I'll destroy you at chess" vs a real threat)
- Whether content targets a specific person or group maliciously

Respond ONLY with a valid JSON object -- no markdown fences, no extra text:
{
  "decision": "allow" or "flag" or "remove",
  "confidence": <float 0.0-1.0>,
  "explanation": "<1-2 sentence plain-English explanation of your reasoning>"
}

allow  = safe content, no harm intended or likely
flag   = ambiguous, mildly toxic, sarcastic, context-dependent -- needs human review
remove = clear hate speech, credible threats, targeted harassment, highly toxic content"""

    user_prompt = f"Text to moderate: {text}\n\nToxicity Scores:\n{json.dumps(hf_scores, indent=2)}"
    
    try:
        response = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        # Security strip markdown edges
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        data = json.loads(content)
        
        # Hard cap enforcements
        decision = data.get("decision", "flag").lower()
        if decision not in ["allow", "flag", "remove"]:
            decision = "flag"
            
        try:
            confidence = float(data.get("confidence", 0.5))
        except:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "decision": decision,
            "confidence": confidence,
            "explanation": str(data.get("explanation", "Automatically flagged due to complex parsing context."))
        }
        
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

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
        scores = predict_toxicity(text)
    except Exception as e:
        scores = {}
        
    ai_scores = {
        "toxicity": float(scores.get("toxicity", 0.0)),
        "insult": float(scores.get("insult", 0.0)),
        "threat": float(scores.get("threat", 0.0)),
        "obscene": float(scores.get("obscene", 0.0))
    }

    # Stage 2: OpenAI Deep Reasoning
    llm_result = openai_moderate(text, ai_scores)
    
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
