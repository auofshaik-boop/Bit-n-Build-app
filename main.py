"""
Sentinel — Real-Time Child Safety Chat Copilot
LIGHTWEIGHT deployment version.

Instead of downloading and running the AI model on our own server (which
needs more RAM than Render's free tier gives us), this version sends each
message to Hugging Face's own hosted servers and gets the classification
back over the internet. Our app stays small and fits comfortably in 512MB.

Needs one environment variable set on Render: HF_TOKEN
(a free Hugging Face API token — see deployment instructions)
"""

import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Sentinel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

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
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": message.text,
        "parameters": {"candidate_labels": CANDIDATE_LABELS},
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    result = response.json()

    if "labels" not in result:
        # The model may still be "waking up" on Hugging Face's side the
        # very first time it's called — this gives a clear message instead
        # of crashing.
        return {
            "error": "Model is loading on Hugging Face's servers, try again in ~20 seconds.",
            "raw_response": result,
        }

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


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
