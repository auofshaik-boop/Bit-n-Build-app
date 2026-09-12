"""
Sentinel — Real-Time Child Safety Chat Copilot
Deployment-ready version: serves BOTH the API and the demo frontend
from a single app, so you get one URL for everything.

Local run:
    pip install -r requirements.txt
    uvicorn main:app --reload
    Then open http://127.0.0.1:8000 in your browser (not index.html directly!)

Render deployment:
    Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentinel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: using a smaller model than the original bart-large-mnli.
# bart-large-mnli is ~1.6GB and needs more RAM than free hosting tiers give you.
# typeform/distilbert-base-uncased-mnli does the same zero-shot job at a
# fraction of the size (~260MB), so it actually fits on a free server.
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
)

CANDIDATE_LABELS = [
    "sharing personal information like address or school",
    "asking to switch to another app like Snapchat, WhatsApp, or phone number",
    "manipulative pressure, flattery, or secrecy",
    "normal safe conversation",
]

LABEL_TO_TIER = {
    "sharing personal information like address or school": {
        "category": "PII_SHARING",
        "tier": 2,
        "action": "Soft pause: dim the input box, show a countdown, explain why.",
    },
    "asking to switch to another app like Snapchat, WhatsApp, or phone number": {
        "category": "CONTACT_SWITCHING",
        "tier": 3,
        "action": "Hard safety pause: show a calm warning message, then minimize chat and alert parent.",
    },
    "manipulative pressure, flattery, or secrecy": {
        "category": "GROOMING_PRESSURE",
        "tier": 1,
        "action": "Non-intrusive tip overlay for the minor. No interruption.",
    },
    "normal safe conversation": {
        "category": "SAFE_CHAT",
        "tier": 0,
        "action": "No action.",
    },
}


class Message(BaseModel):
    text: str


@app.post("/analyze")
def analyze(message: Message):
    result = classifier(message.text, candidate_labels=CANDIDATE_LABELS)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    tier_info = LABEL_TO_TIER[top_label]

    return {
        "input_text": message.text,
        "matched_label": top_label,
        "confidence": round(top_score, 3),
        "category": tier_info["category"],
        "tier": tier_info["tier"],
        "recommended_action": tier_info["action"],
        "all_scores": [
            {"label": l, "score": round(s, 3)}
            for l, s in zip(result["labels"], result["scores"])
        ],
    }


# Serve index.html at the root URL, so the whole app is reachable
# from ONE single link (both frontend and backend together).
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
