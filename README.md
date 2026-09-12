# 🛡️ Sentinel — Real-Time AI Copilot for Child Online Safety

**Track:** Bal Suraksha (Child Safety, Protection & Well-being)
**Hackathon:** Bit N Build

**Live Demo:** https://sentinel-k3w9.onrender.com *(Free hosting — the first request after a period of inactivity can take up to ~50 seconds while the server wakes up. This is a hosting-tier limitation, not a bug.)*

**GitHub:** https://github.com/auofshaik-boop/Bit-n-Build-app

---

## The Problem

Millions of children in India chat daily on gaming platforms and apps like Discord. Existing child-safety tools fail in two opposite ways:

- **Parental monitoring apps** (e.g. Bark) act like digital wiretaps — they log conversations and alert parents after a risky interaction has already happened.
- **Platform auto-filters** only catch exact keywords, and are trivially bypassed with spacing or leetspeak like `sn@p_ch@t`.

Neither intervenes **in the moment**, and neither is designed around the child's experience — they're built for parents to review afterward, not for the child to be coached through the moment of risk itself.

## The Solution

Sentinel reads incoming chat messages in real time and classifies their *intent* using a zero-shot NLP model, then responds with **risk-based intent classification and graduated interventions** across two dimensions:

- **Tier** — decides *what the app does* (which UI response fires: a tip, a pause, or a full lockdown)
- **Risk class** — explains *why*, grouping categories by the underlying nature of the risk

Contact-switching (asking to move to Snapchat/WhatsApp) is an **escalation pathway** — a step toward possible future harm. Explicit sexual content and threats/coercion **are** the harm, happening in that message, right now. Both get the same immediate Tier 3 response, but Sentinel tracks them as distinct risk classes.

| Tier | Category | Risk Class | What Sentinel does |
|------|----------|------------|---------------------|
| **1** | Grooming / manipulation | Manipulation | Shows a non-intrusive safety tip. No interruption. |
| **2** | PII sharing (address, school, etc.) | Privacy Risk | 10-second pause with a real explanation, then requires an active **"Send anyway"** or **"Don't send"** choice. Repeated attempts (3+) in one session are logged as an escalating pattern. |
| **3** | Contact-switching | Escalation Pathway | Calm warning, server-side evidence save, chat minimized, parent alerted. |
| **3** | Sexual solicitation / Threat & coercion | Harmful Content | Same immediate response — evidence saved, chat minimized, parent alerted, treated as maximum severity. |

## Architecture

```
[ Chat message ]
        │
        ▼
FastAPI /analyze endpoint
        │
        ▼
Short-message pre-filter (≤2 words skip the model entirely → SAFE_CHAT)
        │
        ▼
Hugging Face router API → facebook/bart-large-mnli (zero-shot)
        │
        ▼
Scored against 5 risk labels + 1 safe label
        │
        ▼
Severity-priority selection (checks most-severe categories first;
each category only "wins" if it clears its own calibrated threshold)
        │
        ▼
Tier + category + risk_class + recommended action returned as JSON
```

### Candidate labels

- `"asking for personal information like home address, school name, or real name"` → **PII_SHARING**
- `"asking to switch to another app like Snapchat, WhatsApp, or phone number"` → **CONTACT_SWITCHING**
- `"pressuring someone to keep a secret from their parents, isolating them from friends and family, or giving unusual personal compliments to build trust"` → **GROOMING_PRESSURE**
- `"explicit sexual content or sexual solicitation"` → **SEXUAL_SOLICITATION**
- `"threats, coercion, blackmail, or intimidation to force compliance"` → **THREAT_COERCION**
- `"normal safe conversation"` → **SAFE_CHAT**

The PII and manipulation labels were reworded during testing (see below) — the original wording described a *child* sharing/behaving a certain way, which didn't match how these messages are actually phrased in practice (someone *asking* the child for information, or *pressuring* them).

## Threshold Calibration — What We Learned by Testing

This is worth documenting honestly, because it's a real finding, not a guess:

**Zero-shot classification with multiple candidate labels dilutes confidence.** With 6 candidate labels, the model splits its confidence across all of them — a genuinely correct detection often scores only 30–50%, not 90%+, simply because it has to outrank 5 other options rather than clear some high absolute bar. Random guessing across 6 labels averages out to roughly 1-in-6 ≈ 17%.

This means a single global confidence threshold doesn't work. We iterated through several approaches using real test messages before landing on the current design:

1. **No threshold (original version):** a one-word message like "Hey" could be misclassified as Tier 3, because the model always returns a "top" label even when nothing genuinely applies.
2. **One flat threshold (0.55) for every category:** fixed the "Hey" problem, but also cleared genuinely dangerous messages (e.g. sexual solicitation scoring ~43%) as safe — false negatives on the highest-severity content.
3. **Per-category thresholds, tuned too low:** fixed the false negatives, but with thresholds set below the ~17% random-chance floor, ordinary messages ("did you finish the homework") started tripping Tier 3 alerts purely on noise.
4. **Current approach — per-category thresholds derived from real observed score distributions, plus severity-priority selection:** thresholds are set based on actual score gaps seen between true and false positives during testing (e.g. genuine threats scored 0.565–0.858 in testing, while a false-positive "homework" message scored 0.474 — so the threshold sits at 0.55, above the false positive and below genuine hits). When multiple categories clear their threshold, the **most severe** one is selected first, even if a less severe category scored marginally higher — reflecting the product's actual safety priority: a missed severe signal is worse than an over-cautious mild one.

**This remains an active, honestly-documented limitation**, not a solved problem — see below.

## Honest Limitations (we'd rather say this than have a judge find it first)

- **Short, ordinary messages mentioning school/personal topics can still occasionally trigger a low-severity (Tier 1) response.** Manipulation-style phrasing is the hardest category to cleanly separate from casual teenage conversation using an untrained, general-purpose model — a genuine accuracy ceiling, not a threshold we haven't found yet. Tier 1's response (a non-intrusive tip, no interruption) is deliberately the cheapest possible response for exactly this reason.
- **Evidence preservation only protects *this specific conversation*.** It cannot stop an adult from re-approaching a child through a different account or platform — no consumer safety app can.
- **Adding more candidate categories dilutes per-category confidence in zero-shot classification** — this is a structural property of the approach, mitigated (not eliminated) by severity-priority selection. A production version would use two-step classification (first "safe vs. not safe," then classify only flagged messages into a specific category) to keep confidence high as the category list grows.
- **On Render's free tier, saved evidence is not permanent** — it resets if the server restarts or redeploys, since the free tier doesn't include persistent storage.
- **Zero-shot classification has real accuracy limits against heavy obfuscation.** Slang-heavy or deliberately spaced-out messages (e.g. "c a n i g e t y o u r s n a p") are inconsistently detected — sometimes correctly, sometimes not. This mirrors a documented weakness in even larger platforms' moderation systems.

## Tech Stack

- **Backend:** Python, FastAPI, `requests` (calling Hugging Face's hosted Inference API)
- **Frontend:** Single-page HTML/CSS/JS, served directly by FastAPI so the whole app lives at one URL
- **Hosting:** Render.com (free tier)
- **Model:** `facebook/bart-large-mnli` via Hugging Face's router API (`router.huggingface.co`)

## Setup & Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Hugging Face API token at huggingface.co/settings/tokens
#    (needs "Inference Providers" permission)

# 3. Set it as an environment variable
export HF_TOKEN=your_token_here      # Mac/Linux
set HF_TOKEN=your_token_here         # Windows cmd

# 4. Run the server
uvicorn main:app --reload

# 5. Open http://127.0.0.1:8000 in your browser
```

To view captured Tier 3 / repeated-pattern incidents during a session, visit `/incidents` on the running app.

### Testing

`test_sentinel.py` sends a batch of representative test messages (safe, PII, manipulation, contact-switching, sexual/threat, and obfuscated edge cases) to a running instance and prints back the tier, category, and confidence for each — useful for checking a threshold or label change hasn't broken something else.

```bash
python test_sentinel.py
```

## Future Scope

- **Persistent storage** for evidence logs (a real database instead of a file that resets on restart)
- **Two-stage classification** (binary safe/not-safe first, then category) to reduce confidence dilution as more categories are added
- **Desktop deployment** via OS Accessibility APIs (Windows UI Automation / macOS Accessibility) to monitor real chat apps like Discord directly, with an explicit permission-grant flow shown to both parent and child during setup
- **Fine-tuned model** trained on ethically-sourced, consent-cleared conversation data, developed alongside child-safety researchers, to improve accuracy against obfuscated/slang-heavy messages
- **Fullscreen game support** — currently out of scope due to anti-cheat/overlay conflicts with screen-reading approaches

## Team

*[Byte Me, (Team Members:Shaik M Auof, Jason Gracias, Akash R)]*
