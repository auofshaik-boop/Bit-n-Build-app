# 🛡️ Sentinel — Real-Time AI Copilot for Child Online Safety

**Track:** Bal Suraksha (Child Safety, Protection & Well-being)
**Hackathon:** Bit N Build

**Live Demo:** https://sentinel-k3w9.onrender.com
_(Free hosting — the first request after a period of inactivity can take up to ~50 seconds while the server wakes up. This is a hosting-tier limitation, not a bug.)_

**GitHub:** https://github.com/auofshaik-boop/Bit-n-Build-app

---

## The Problem

Millions of children in India chat daily on gaming platforms and apps like Discord. Existing child-safety tools fail in two opposite ways:

- **Parental monitoring apps** (e.g. Bark) act like digital wiretaps — they log conversations and alert parents after a risky interaction has already happened, with documented real-world delays even in the market leaders.
- **Platform auto-filters** only catch exact keywords, and are trivially bypassed with spacing or leetspeak like `sn@p_ch@t`.

Neither intervenes **in the moment**, and neither is designed around the child's experience — they're built for parents to review afterward, not for the child to be coached through the moment of risk itself.

## The Solution

Sentinel reads incoming chat messages in real time and classifies their *intent* using a zero-shot NLP model, then responds with **risk-based intent classification and graduated interventions** — not a flat list of tiers, but two dimensions working together:

- **Tier** — decides *what the app does* (which UI response fires: a tip, a pause, or a full lockdown)
- **Risk class** — explains *why*, grouping categories by the underlying nature of the risk

This distinction matters because not everything at the highest severity level is the same *kind* of danger. Contact-switching (asking to move to Snapchat/WhatsApp) is an **escalation pathway** — a step toward possible future harm. Explicit sexual content and threats/coercion **are** the harm, happening in that message, right now. Both get the same immediate response (Tier 3), but Sentinel tracks them as distinct risk classes, which matters for how a parent or reviewer should interpret and prioritize what they're seeing.

| Tier | Category | Risk Class | What Sentinel does | Why it's designed this way |
|---|---|---|---|---|
| **1** | Grooming / manipulation | Manipulation | Shows a non-intrusive safety tip to the minor. No interruption. | Low-confidence signals shouldn't disrupt normal conversation. |
| **2** | PII sharing (address, school, etc.) | Privacy Risk | **10-second pause with a real explanation** ("personal details like this can be used to find you in real life"), then requires an **active choice** — "Send anyway" or "Don't send." Nothing auto-sends. **Repeated attempts (3+) in one session are logged as an escalating pattern**, even if no single message looks critical alone. | A silent countdown that auto-sends is easy for an invested teenager to simply wait out — it creates the appearance of safety without real friction. Forcing an explicit decision, and tracking the pattern rather than just the single message, treats the moment with the seriousness a genuine emotional attachment deserves. |
| **3** | Contact-switching | Escalation Pathway | Shows a calm warning, **immediately saves a server-side copy of the message** (independent of the child's device), then minimizes the chat and alerts a parent. | Parent alerts alone don't survive the other person deleting their messages before the parent looks at their phone. Preserving evidence the instant the risk is detected means the record exists somewhere the conversation itself can't reach. |
| **3** | Sexual solicitation / Threat & coercion | Harmful Content | Same immediate response as above — evidence saved, chat minimized, parent alerted. | These are content-level harms, not steps toward something else — they get the same maximum-severity response as contact-switching, but are logged under a distinct risk class since they represent a qualitatively different kind of danger. |

## Honest Limitations (we'd rather say this than have a judge find it first)

- **Evidence preservation only protects *this specific conversation*.** It cannot stop an adult from re-approaching a child through a different account or platform — no consumer safety app can. Sentinel narrows the window of harm in one conversation; it does not solve predator re-offense broadly.
- **Adding more candidate categories can dilute per-category confidence in zero-shot classification** — with more options for the model to choose between, individual scores tend to run lower even when the ranking is still correct. A production version would likely use two-step classification (first "safe vs. not safe," then classify only flagged messages into a specific category) instead of one flat multi-way choice, to keep confidence high as the category list grows.
- **On Render's free tier, saved evidence is not permanent** — it resets if the server restarts or redeploys, since the free tier doesn't include persistent storage. A production version would use a real database so evidence survives indefinitely.
- **Zero-shot classification has real accuracy limits against heavy obfuscation.** During testing, slang-heavy or deliberately spaced-out messages (e.g. "c a n i g e t y o u r s n a p") scored lower confidence than clear phrasing. This mirrors a documented weakness in even larger platforms' moderation systems — it's a genuinely hard, unsolved problem in this space, not something we're pretending to have fully solved.

## Architecture

```
[ Chat message ]
        │
        ▼
FastAPI /analyze endpoint
        │
        ▼
Hugging Face router API → facebook/bart-large-mnli (zero-shot)
        │
        ▼
Scored against 6 labels, each with its own calibrated confidence threshold:
  - personal information request (PII_SHARING → Privacy Risk)
  - switching to another app (CONTACT_SWITCHING → Escalation Pathway)
  - manipulative pressure / secrecy / isolation (GROOMING_PRESSURE → Manipulation)
  - explicit sexual content (SEXUAL_SOLICITATION → Harmful Content)
  - threats or coercion (THREAT_COERCION → Harmful Content)
  - normal safe conversation (SAFE_CHAT → Safe)
        │
        ▼
Severity-priority resolution: most severe category that clears
its own threshold wins, even if a milder category scored higher
        │
        ▼
Tier + category + risk_class + recommended action returned as JSON
        │
        ▼
Frontend renders the tier-appropriate response:
  Tier 1 → inline tip
  Tier 2 → 10s pause + explicit Send/Don't Send choice (+ pattern logging at 3+ attempts)
  Tier 3 → calm, category-accurate warning + server-side evidence save (/log_incident) + visual lockdown
```

### Why zero-shot classification, called via API instead of run locally

Two deliberate choices worth explaining:

1. **Zero-shot instead of a custom-trained model** — avoids the ethical and legal complexity of sourcing real grooming-conversation data. We give the model plain-language categories and it scores how well a message matches each one, with no training data of our own required.
2. **Called via Hugging Face's hosted API instead of downloaded and run on our own server** — the model itself is ~1.6GB and needs more RAM than free hosting tiers provide. Sending each message to Hugging Face's servers and getting the result back keeps our own app small enough to run on a free instance, at the cost of a small amount of network latency per request.

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

To view captured Tier 3 / repeated-pattern incidents during a session, visit `/incidents` on the running app (e.g. `http://127.0.0.1:8000/incidents`).

## Confidence Calibration & Safety Engineering

Real testing surfaced two genuine bugs after the initial build — both are fixed in the current version, and we're documenting them here rather than hiding them, since the fixes are themselves evidence of how the system is designed to fail safely.

**Bug 1 — naive top-label selection misfired on harmless messages.** With 6 candidate labels, the model splits its confidence across all of them — a correct detection often only scores 30–50%, not 90%+, simply because it has to statistically "beat" 5 other options rather than clear some high absolute bar. Early on, this meant a plain `"Hey"` could nominally "win" a risk category at a low, meaningless confidence and get treated as Tier 3. **Fix:** every non-safe category now has its own calibrated confidence threshold, set from real observed score distributions rather than guesses — e.g. `THREAT_COERCION` needs 0.55 confidence specifically because a genuine false positive ("did you finish the homework") scored 0.474, while real threats in testing scored 0.565–0.858. When multiple categories clear their threshold, Sentinel deliberately selects the **more severe** one, even if a milder category scored a few points higher — reflecting the product's safety philosophy that under-classifying real danger is worse than an occasional stronger-than-necessary response.

**Bug 2 — a length-based shortcut created a blind spot.** An earlier fix skipped AI classification entirely for messages of 2 words or fewer (to stop "Hey"/"lol" from misfiring), but this meant a genuinely harmful 2-word message (e.g. "send nudes") would have skipped classification entirely and auto-returned Safe. **Fix:** the shortcut was removed. The threshold system above already handles short harmless messages correctly on its own, making the length-based bypass both redundant and unsafe — every message, however short, now goes through full classification.

**A further, ongoing tuning pass** found `GROOMING_PRESSURE` (Tier 1) missing real examples like *"you're so much more mature than other people your age"* at its original 0.40 threshold. Since Tier 1's response is just a non-intrusive tip — not a lockdown or parent alert — the threshold was lowered to 0.25, accepting more false positives in exchange for catching more real manipulation, since the cost of a false positive here (an unnecessary tip) is negligible compared to a missed real one.

## Future Scope

- **Persistent storage** for evidence logs (a real database instead of a file that resets on restart)
- **Desktop deployment** via OS Accessibility APIs (Windows UI Automation / macOS Accessibility) to monitor real chat apps like Discord directly, with an explicit permission-grant flow shown to both parent and child during setup
- **Fine-tuned model** trained on ethically-sourced, consent-cleared conversation data, developed alongside child-safety researchers, to improve accuracy against obfuscated/slang-heavy messages
- **Fullscreen game support** — currently out of scope due to anti-cheat/overlay conflicts with screen-reading approaches

## Team

Byte Me — Shaik M Auof, Jason Gracias, Akash R
