---
title: SafeSpaceAI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SafeStream AI — Intelligent Content Moderation

> AI-powered content moderation using rule-based scoring and reinforcement learning for smarter, faster, and more adaptive decisions.

---

## Features

- *AI toxicity analysis* — scores content across multiple harm categories
- *RL-driven decision engine* — outputs one of: Allow / Flag / Remove / Review
- *Confidence scoring* — quantified certainty on every moderation decision
- *Category breakdown* — per-content scores for toxicity, insult, threat, and obscene language
- *Live moderation history* — running log of past decisions in the dashboard
- *Real-time stats* — dashboard metrics updated on every request
- *Modern UI* — clean gradient-styled interface

---

## Architecture


Frontend (HTML/CSS/JS)
        ↓
FastAPI Backend (/moderate)
        ↓
  AI + RL Decision Logic
        ↓
Structured Moderation Output

---

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | HTML, CSS, JavaScript               |
| Backend     | FastAPI (Python)                    |
| Deployment  | Hugging Face Spaces (Docker)        |
| Model logic | Rule-based scoring + AI (extendable)|

---

## Project Structure


.
├── app.py
├── requirements.txt
├── Dockerfile
├── templates/
│   └── index.html
└── static/
    ├── styles.css
    ├── script.js
    └── logo.jpeg

---

## How It Works

1. User submits text via the dashboard
2. Frontend sends a POST request to /moderate
3. Backend analyzes the content using AI scoring + RL logic
4. Response includes a decision, confidence score, explanation, and category breakdown
5. Dashboard updates in real time

---

## API Reference

### POST /moderate

*Request body:*
json
{
  "text": "Your content here"
}


*Response:*
json
{
  "decision": "flag",
  "confidence": 0.85,
  "explanation": "Potentially harmful content detected",
  "ai_scores": {
    "toxicity": 0.8,
    "insult": 0.6,
    "threat": 0.7,
    "obscene": 0.5
  }
}
*Decision values:* allow · flag · remove · review
---
## Running Locally
*1. Clone the repository*
bash
git clone <your-repo-url>
cd safestream-ai
*2. Install dependencies*
bash
pip install -r requirements.txt
*3. Start the server*
bash
uvicorn app:app --reload
*4. Open in browser*
http://127.0.0.1:8000
---
## Deployment
This project is deployed on *Hugging Face Spaces* using Docker.
- Dockerfile handles container setup
- FastAPI app runs on port 7860
---
## Roadmap
- [ ] Integrate real LLM (OpenAI / Anthropic / Perspective API)
- [ ] Train RL agent dynamically on moderation feedback
- [ ] Analytics dashboard with charts
- [ ] Multi-language moderation support
- [ ] User authentication and persistent moderation logs
- [ ] Real-time streaming moderation
- [ ] Webhook support for external integrations
---
## Use Cases
- Social media platforms
- Community forums and Discord servers
- Live chat and messaging apps
- Online gaming platforms
- Content safety pipelines
---
## Author
Built by Team *Good Girls Guide to AI* · Systems · Product
---
## Inspiration
As online content grows exponentially, scalable and intelligent moderation becomes critical infrastructure. SafeStream AI explores how AI and reinforcement learning can work together to make moderation smarter, faster, and more adaptive — reducing both false positives and harmful content slipping through.