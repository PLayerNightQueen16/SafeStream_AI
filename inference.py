"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method
"""

import asyncio
import os
import textwrap

from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional

from openai import OpenAI

try:
    from my_env_v4 import MyEnvV4Action, MyEnvV4Env
except ImportError:
    # Minimal mock or fallback if not installed natively
    class MyEnvV4Action:
        def __init__(self, message: str):
            self.message = message
    
    class MyEnvV4Env:
        @classmethod
        async def from_docker_image(cls, image_name):
            import asyncio
            # Give Uvicorn a moment to bind
            await asyncio.sleep(2)
            return cls()

        def __init__(self):
            self.base_url = "http://127.0.0.1:7860"
        
        async def reset(self):
            import httpx
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post(f"{self.base_url}/reset", json={}, timeout=5.0)
                    data = resp.json()
                except Exception:
                    data = {"observation": {"echoed_message": "fallback data"}, "done": False}
                
                class Obj: pass
                class Obs: pass
                res = Obj()
                res.observation = Obs()
                res.observation.echoed_message = data.get("observation", {}).get("echoed_message", "")
                res.done = data.get("done", False)
                return res

        async def step(self, action):
            import httpx
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post(f"{self.base_url}/step", json={"action": {"message": action.message}}, timeout=5.0)
                    data = resp.json()
                except Exception:
                    data = {"observation": {"echoed_message": ""}, "reward": 0.0, "done": True}
                
                class Obj: pass
                class Obs: pass
                res = Obj()
                res.observation = Obs()
                res.observation.echoed_message = data.get("observation", {}).get("echoed_message", "")
                res.reward = data.get("reward", 0.0)
                res.done = data.get("done", True)
                return res

        async def close(self):
            pass

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "moderate")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.1
MAX_TOKENS = 10
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

_MAX_REWARD_PER_STEP = 1.0 # 1.0 for perfect match
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a content moderation AI.
    The user will provide a piece of text to moderate.
    You must output exactly one of these actions: "allow" or "flag" or "remove".
    Do not add any reasoning, quotes, or additional text.
    "allow": mostly safe content.
    "flag": somewhat toxic, sarcasm, or slightly risky content.
    "remove": hate speech, threats, go die/disappear, or extremely toxic.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    return textwrap.dedent(
        f"""
        Please moderate the following text:
        {last_echoed!r}
        Output exactly "allow", "flag", or "remove".
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "allow"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "allow"


async def main() -> None:
    if API_BASE_URL:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        client = OpenAI(api_key=API_KEY)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, last_echoed, last_reward, history)

            result = await env.step(MyEnvV4Action(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())