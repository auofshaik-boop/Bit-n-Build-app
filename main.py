"""
Sentinel — Real-Time Child Safety Chat Copilot
Backend: FastAPI + Hugging Face zero-shot classification (facebook/bart-large-mnli)

Run with:
    pip install fastapi uvicorn transformers torch --break-system-packages
    uvicorn main:app --reload

Then open index.html in your browser (it calls http://127.0.0.1:8000/analyze)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentinel API")

# Allow the demo frontend (index.html) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the zero-shot classification model once, at startup (not per-request —
# reloading it every call would be extremely slow).
# Zero-shot means: no custom training data needed. We just hand it labels
# and it scores how well the input text matches each one.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# The four categories Sentinel cares about.
CANDIDATE_LABELS = [
    "sharing personal information like address or school",
    "asking to switch to another app like Snapchat, WhatsApp, or phone number",
    "manipulative pressure, flattery, or secrecy",
    "normal safe conversation",
]

# Maps each label to a Tier + suggested action for the frontend to display.
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
    """
    Takes one chat message and returns:
      - which risk category it best matches
      - the model's confidence (0-1)
      - what Tier of response Sentinel recommends
    """
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
        # Full breakdown, useful for debugging / showing judges the model's reasoning
        "all_scores": [
            {"label": l, "score": round(s, 3)}
            for l, s in zip(result["labels"], result["scores"])
        ],
    }


@app.get("/")
def root():
    return {"status": "Sentinel API is running. POST to /analyze with {'text': '...'}"}
