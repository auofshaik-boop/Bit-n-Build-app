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
    "asking for personal information like home address, school name, or real name",
    "asking to switch to another app like Snapchat, WhatsApp, or phone number",
    "pressuring someone to keep a secret from their parents, isolating them from friends and family, or giving unusual personal compliments to build trust",
    "explicit sexual content or sexual solicitation",
    "threats, coercion, blackmail, or intimidation to force compliance",
    "normal safe conversation",
]

SAFE_LABEL = "normal safe conversation"

# How confident the model needs to be before we trust a non-safe label.
#
# IMPORTANT LESSON FROM TESTING: with 6 candidate labels, the model splits
# its confidence across all of them — a CORRECT detection often only scores
# 30-50%, not 90%+, simply because it has to "beat" 5 other options, not
# hit some high absolute bar. Random/uniform guessing across 6 labels would
# average out to about 1/6 ≈ 17% each. So these thresholds are set just
# above that noise floor, to catch genuinely directionless guesses, NOT to
# filter out real detections that happen to score under 50%.
#
# GROOMING_PRESSURE was lowered from 0.40 -> 0.25 after testing showed real
# examples slipping through at 0.40 (e.g. "you're so much more mature than
# other people your age" scored 0.339; "don't tell your parents, it's just
# between us" scored below 0.40 on this label specifically, even though the
# model's own raw top guess for that message was "safe" at 0.518). Since
# Tier 1's response is just a non-intrusive tip — not a lockdown or parent
# alert — a lower threshold here is low-risk even if it produces more false
# positives: worst case is an unnecessary tip, not a false alarm to a
# parent. RE-TEST after this change to confirm both examples now pass.
CATEGORY_THRESHOLDS = {
    "asking for personal information like home address, school name, or real name": 0.30,
    "asking to switch to another app like Snapchat, WhatsApp, or phone number": 0.30,
    "pressuring someone to keep a secret from their parents, isolating them from friends and family, or giving unusual personal compliments to build trust": 0.25,
    "explicit sexual content or sexual solicitation": 0.30,
    "threats, coercion, blackmail, or intimidation to force compliance": 0.55,
}

# When more than one category clears its threshold, we deliberately prefer
# the MORE SEVERE one, even if a less severe category happened to score a
# few points higher. This directly reflects the product's safety
# philosophy: the cost of under-classifying a genuinely dangerous message
# (e.g. showing a soft "grooming" tip instead of triggering the Tier 3
# evidence-save + parent alert) is far worse than occasionally showing a
# stronger response than strictly necessary. Ordered most severe first.
CATEGORY_PRIORITY = [
    "explicit sexual content or sexual solicitation",
    "threats, coercion, blackmail, or intimidation to force compliance",
    "asking to switch to another app like Snapchat, WhatsApp, or phone number",
    "asking for personal information like home address, school name, or real name",
    "pressuring someone to keep a secret from their parents, isolating them from friends and family, or giving unusual personal compliments to build trust",
]

# Two dimensions on purpose:
#   "tier"       — decides WHAT the app does (which UI behavior fires)
#   "risk_class" — explains WHY, grouping categories by what kind of risk
#                  they represent. Contact-switching is a step TOWARD future
#                  harm (an escalation pathway); sexual content and threats
#                  ARE the harm, happening right now. Same tier/response
#                  severity, different underlying nature — worth knowing
#                  even though the immediate action taken is the same.
LABEL_TO_TIER = {
    "asking for personal information like home address, school name, or real name": {
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
    "pressuring someone to keep a secret from their parents, isolating them from friends and family, or giving unusual personal compliments to build trust": {
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
    """Build a 'treat as safe' response. Used by fallback paths below, so a
    network hiccup or a slow-loading model fails to Tier 0 (safe) instead of
    ever accidentally defaulting to Tier 3."""
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

    # NOTE: an earlier version of this function skipped the AI model
    # entirely for messages of 2 words or fewer, to stop "Hey"/"lol" from
    # misfiring. That shortcut was REMOVED — it created a worse blind spot:
    # a genuinely harmful 2-word message (e.g. "send nudes", "give address")
    # would have skipped classification entirely and auto-returned Safe.
    # The threshold + severity-priority system below already handles short
    # harmless messages correctly on its own (that's what actually produced
    # the clean "Hey/lol/wyd -> Safe" results during testing), so the
    # word-count shortcut was redundant AND unsafe. Every message, however
    # short, now goes through the real classification path below.

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": CANDIDATE_LABELS},
    }

    # Network/timeout errors fail SAFE, not Tier 3.
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

    # Severity-priority selection, not just raw top-1: build a quick lookup
    # of every category's score, then walk through CATEGORY_PRIORITY (most
    # severe first). The first category that clears ITS OWN threshold wins
    # — even if a less severe category scored higher in the raw output.
    scores_by_label = {r["label"]: r["score"] for r in result}

    selected_label = None
    for candidate_label in CATEGORY_PRIORITY:
        candidate_score = scores_by_label.get(candidate_label, 0.0)
        if candidate_score >= CATEGORY_THRESHOLDS[candidate_label]:
            selected_label = candidate_label
            top_score = candidate_score
            break

    if selected_label is not None:
        top_label = selected_label
    elif top_label != SAFE_LABEL:
        # Nothing cleared its severity-priority threshold, and the model's
        # own raw top pick also wasn't safe — treat as safe rather than
        # trust a low-confidence guess.
        top_label = SAFE_LABEL

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
