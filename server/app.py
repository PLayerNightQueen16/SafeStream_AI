from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = [
    ("I love this!", "allow"),
    ("You are amazing", "allow"),
    ("I hate you", "remove"),
    ("Go die", "remove"),
    ("Wow you're so smart 🙄", "flag"),
    ("Maybe you should disappear", "remove"),
    ("Nice work!", "allow"),
    ("This is trash", "flag")
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
    body = {}
    try:
        body = await request.json()
    except:
        pass
    
    return ResetResponse(
        observation=Observation(echoed_message=data[current_task_idx][0]),
        done=False
    )

@app.post("/step", response_model=StepResponse)
async def step(request: Request):
    global current_task_idx
    body = {}
    try:
        body = await request.json()
    except:
        pass
        
    msg = ""
    if "action" in body and isinstance(body["action"], dict) and "message" in body["action"]:
        msg = body["action"]["message"]
    elif "message" in body:
        msg = body["message"]
        
    true_label = data[current_task_idx][1]
    
    if msg.lower().strip() == true_label.lower():
        reward = 1.0
    else:
        reward = 0.0

    current_task_idx = (current_task_idx + 1) % len(data)
    
    return StepResponse(
        observation=Observation(echoed_message=data[current_task_idx][0]),
        reward=reward,
        done=True
    )

@app.get("/state")
async def state():
    return {
        "observation": {"echoed_message": data[current_task_idx][0]},
        "done": False
    }

class ModerationRequest(BaseModel):
    text: str

@app.post("/moderate")
def moderate(request: ModerationRequest):
    return {"status": "ok"} # Dummy for now if frontend needs it.

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
