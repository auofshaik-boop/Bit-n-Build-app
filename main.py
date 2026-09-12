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
import json
import requests
from datetime import datetime, timezone
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

# Where captured Tier 3 incidents get saved.
# NOTE: on Render's free tier, this file resets whenever the app restarts or
# redeploys — fine for a hackathon demo, but a real version would use a
# proper database so evidence survives restarts.
INCIDENT_LOG_FILE = "incidents.json"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

CANDIDATE_LABELS = [
    "sharing personal information like address or school",
    "asking to switch to another app like Snapchat, WhatsApp, or phone number",
    "manipulative pressure, flattery, or secrecy",
    "explicit sexual content or sexual solicitation",
    "threats, coercion, blackmail, or intimidation to force compliance",
    "normal safe conversation",
]

SAFE_LABEL = "normal safe conversation"

# How confident the model needs to be before we trust a non-safe label.
# Zero-shot models ALWAYS pick a "winner" out of the candidate labels, even
# when none of them really apply — a low-confidence guess (e.g. 20%) is not
# the same thing as the model actually detecting something. Anything below
# this threshold gets treated as safe instead of escalated.
CONFIDENCE_THRESHOLD = 0.55

# Two dimensions on purpose:
#   "tier"       — decides WHAT the app does (which UI behavior fires)
#   "risk_class" — explains WHY, grouping categories by what kind of risk
#                  they represent. Contact-switching is a step TOWARD future
#                  harm (an escalation pathway); sexual content and threats
#                  ARE the harm, happening right now. Same tier/response
#                  severity, different underlying nature — worth knowing
#                  even though the immediate action taken is the same.
LABEL_TO_TIER = {
    "sharing personal information like address or school": {
        "category": "PII_SHARING",
        "risk_class": "PRIVACY_RISK",
        "tier": 2,
        "action": "10-second pause with a real explanation, then requires an active 'Send anyway' or 'Don't send' choice — no auto-send.",
    },
    "asking to switch to another app like Snapchat, WhatsApp, or phone number": {
        "category": "CONTACT_SWITCHING",
        "risk_class": "ESCALATION_PATHWAY",
        "tier": 3,
        "action": "Hard safety pause: show a calm warning message, then minimize chat and alert parent.",
    },
    "manipulative pressure, flattery, or secrecy": {
        "category": "GROOMING_PRESSURE",
        "risk_class": "MANIPULATION",
        "tier": 1,
        "action": "Non-intrusive tip overlay for the minor. No interruption.",
    },
    "explicit sexual content or sexual solicitation": {
        "category": "SEXUAL_SOLICITATION",
        "risk_class": "HARMFUL_CONTENT",
        "tier": 3,
        "action": "Immediate hard safety pause: evidence saved instantly, chat minimized, parent alerted — treated as the highest-severity category.",
    },
    "threats, coercion, blackmail, or intimidation to force compliance": {
        "category": "THREAT_COERCION",
        "risk_class": "HARMFUL_CONTENT",
        "tier": 3,
        "action": "Immediate hard safety pause: evidence saved instantly, chat minimized, parent alerted — treated as the highest-severity category.",
    },
    SAFE_LABEL: {
        "category": "SAFE_CHAT",
        "risk_class": "SAFE",
        "tier": 0,
        "action": "No action.",
    },
}


class Message(BaseModel):
    text: str


class Incident(BaseModel):
    text: str
    category: str
    confidence: float


def save_incident(incident: dict):
    """Append one incident to the log file, creating it if needed."""
    incidents = []
    if os.path.exists(INCIDENT_LOG_FILE):
        try:
            with open(INCIDENT_LOG_FILE, "r") as f:
                incidents = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            incidents = []
    incidents.append(incident)
    with open(INCIDENT_LOG_FILE, "w") as f:
        json.dump(incidents, f, indent=2)


def _safe_response(text: str, confidence: float = 1.0, note: str = None, error: str = None, raw_response=None):
    """Build a 'treat as safe' response. Used by every fallback path below,
    so a short message, a network hiccup, or a slow-loading model all fail
    to Tier 0 (safe) instead of ever accidentally defaulting to Tier 3."""
    tier_info = LABEL_TO_TIER[SAFE_LABEL]
    response = {
        "input_text": text,
        "matched_label": SAFE_LABEL,
        "confidence": round(confidence, 3),
        "category": tier_info["category"],
        "risk_class": tier_info["risk_class"],
        "tier": tier_info["tier"],
        "recommended_action": tier_info["action"],
        "all_scores": [],
    }
    if note:
        response["note"] = note
    if error:
        response["error"] = error
    if raw_response is not None:
        response["raw_response"] = raw_response
    return response


@app.post("/log_incident")
def log_incident(incident: Incident):
    """
    Saves a copy of a Tier 3 (critical risk) message the INSTANT it's
    detected — independent of the child's device or chat app. Even if the
    other person deletes their messages afterward, this record already
    exists here and can't be erased by anything happening in the chat.
    """
    record = {
        "text": incident.text,
        "category": incident.category,
        "confidence": incident.confidence,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    save_incident(record)
    return {"status": "saved", "captured_at": record["captured_at"]}


@app.get("/incidents")
def get_incidents():
    """View all captured incidents — this is what a parent-facing view
    would read from in a full version of Sentinel."""
    if not os.path.exists(INCIDENT_LOG_FILE):
        return {"incidents": []}
    with open(INCIDENT_LOG_FILE, "r") as f:
        return {"incidents": json.load(f)}


@app.post("/analyze")
def analyze(message: Message):
    text = message.text.strip()

    # --- FIX 1: short-message pre-filter -----------------------------------
    # A message like "Hey" or "lol" has no real content to classify. Rather
    # than forcing the model to guess a "danger" category for near-empty
    # text, treat anything at or under 2 words as safe and skip the API
    # call entirely (also saves you a Hugging Face request).
    if len(text.split()) <= 2:
        return _safe_response(text, confidence=1.0, note="Skipped model call: message too short to meaningfully classify.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": CANDIDATE_LABELS},
    }

    # --- FIX 2: network/timeout errors fail SAFE, not Tier 3 ---------------
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
    except requests.exceptions.RequestException:
        return _safe_response(text, confidence=0.0, error="Could not reach the classification model — treated as safe by default.")

    # The Hugging Face router API returns a LIST of {"label": ..., "score": ...}
    # dicts, sorted highest-confidence first. If it's not shaped like that,
    # the model is probably still loading on their servers.
    if not isinstance(result, list) or len(result) == 0 or "label" not in result[0]:
        return _safe_response(
            text,
            confidence=0.0,
            error="Model is loading on Hugging Face's servers, try again in ~20 seconds.",
            raw_response=result,
        )

    top_label = result[0]["label"]
    top_score = result[0]["score"]

    # --- FIX 3: confidence threshold ----------------------------------------
    # Zero-shot models always produce a "top" label, even when none of the
    # candidates genuinely apply. If the top score doesn't clear the
    # threshold, don't trust it — fall back to safe instead of escalating
    # on a low-confidence guess.
    if top_label != SAFE_LABEL and top_score < CONFIDENCE_THRESHOLD:
        top_label = SAFE_LABEL
        top_score = result[0]["score"]  # keep the original score for transparency

    tier_info = LABEL_TO_TIER[top_label]

    return {
        "input_text": text,
        "matched_label": top_label,
        "confidence": round(top_score, 3),
        "category": tier_info["category"],
        "risk_class": tier_info["risk_class"],
        "tier": tier_info["tier"],
        "recommended_action": tier_info["action"],
        "all_scores": [
            {"label": r["label"], "score": round(r["score"], 3)}
            for r in result
        ],
    }


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
