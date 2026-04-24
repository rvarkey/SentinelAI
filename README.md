# SentinelAI

An AI agent that detects phishing login pages by combining computer vision with semantic memory.

Given any URL, SentinelAI screenshots the page, analyses it with a local vision model, and compares it against a stored baseline of what the legitimate page should look like. If the page looks different from the known-good baseline — wrong branding, missing elements, visual drift — it flags it as suspicious before any credentials are injected.

---

## The problem it solves

Enterprise password managers inject credentials into login forms automatically. The critical risk is: **what if the login page is fake?** A phishing page that looks like GitHub or Microsoft will receive injected credentials just like the real page would.

SentinelAI solves this by acting as a visual guard — it verifies the login page looks legitimate before allowing credential injection to proceed.

---

## How it works

```
User navigates to a login page
        ↓
Agent takes a headless screenshot (Playwright)
        ↓
Detects multi-step login flows (email → Next → password)
and interacts with the page to capture the full form
        ↓
Vision model (Ollama qwen3-vl:8b) analyses the screenshot:
  • Is this a login page?
  • Does it look legitimate?
  • Any visual red flags?
        ↓
Description is embedded (nomic-embed-text → 768-dim vector)
        ↓
Vector compared against stored baseline snapshots (ChromaDB)
  • Similarity > 0.75 → looks like the real page → LOW risk
  • Similarity < 0.75 → visual drift detected → HIGH risk
        ↓
Structured risk report returned (JSON)
```

Both signals — vision analysis and memory comparison — contribute to the final risk level. A page that looks suspicious visually OR looks different from the baseline triggers a HIGH risk rating.

---

## Key technical decisions

**Local-only inference (Ollama)**
All vision and embedding inference runs locally. No screenshots or page content are sent to external APIs. This is essential for enterprise credential management where page content may be sensitive.

**Text embeddings over pixel comparison**
Rather than comparing raw pixels (brittle — pages change slightly every visit), the agent describes each page in structured text and embeds that description. Semantically similar pages produce similar vectors even with minor visual variation.

**Structured description prompt**
The vision model is prompted to describe pages using a fixed 8-section schema (BRAND, LAYOUT, FORM_FIELDS, BUTTONS, COLORS, TRUST_SIGNALS, ERRORS_OR_ANOMALIES, URL_CONTEXT). This forces consistent output across visits, reducing embedding variance for the same page.

**Multi-snapshot baselines**
Each URL is visited 3 times during baseline creation. All 3 descriptions are stored separately. On comparison, similarity is computed against all stored snapshots and the best score is used. This makes the system tolerant of minor model variance without reducing sensitivity to real drift.

**Multi-step login handling**
Login flows like Microsoft that show email on page 1 and password on page 2 are detected automatically. The agent fills the email field with a probe value, clicks Next, and captures both screens. Analysis is performed on the combined flow.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              FastAPI REST API               │
│  POST /analyse  POST /baseline  GET /health │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│              Agent pipeline                 │
│                                             │
│  Playwright → screenshot → describe         │
│       ↓                        ↓            │
│  multi-step            structured prompt    │
│  detection             (8 fixed sections)   │
│       ↓                        ↓            │
│  vision analysis       embed description    │
│  (risk + red flags)    (768-dim vector)     │
│       ↓                        ↓            │
│  ┌────┴────────────────────────┴───────┐    │
│  │         ChromaDB (cosine)           │    │
│  │   3 baseline snapshots per URL      │    │
│  │   persistent across restarts        │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│              Ollama (local)                 │
│  qwen3-vl:8b        nomic-embed-text        │
│  vision analysis    text embeddings         │
└─────────────────────────────────────────────┘
```

---

## Stack

| Component | Technology | Purpose |
|---|---|---|
| Vision model | Ollama + qwen3-vl:8b | Screenshot analysis, page description |
| Embedding model | Ollama + nomic-embed-text | Text → vector conversion |
| Vector database | ChromaDB (cosine metric) | Baseline storage and similarity search |
| Browser automation | Playwright (Chromium) | Headless screenshot, DOM interaction |
| API layer | FastAPI + Uvicorn | REST endpoints, auto-generated docs |
| Language | Python 3.13 (async) | Full async pipeline with asyncio |

All AI inference runs locally — no OpenAI, no Anthropic, no external API keys required.

---

## Setup

**Requirements**
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- 8GB+ RAM (runs on CPU — no GPU required)

**Install**

```bash
git clone https://github.com/YOUR_USERNAME/sentinelai
cd sentinelai

pip install -r requirements.txt
python -m playwright install chromium

ollama pull qwen3-vl:8b
ollama pull nomic-embed-text
```

**Run the agent directly**

```bash
python agent_v3.py
```

This runs three passes:
1. Builds baselines for GitHub and Microsoft login pages (3 snapshots each)
2. Verifies real pages score HIGH similarity against their baselines
3. Tests the fake phishing page — should score LOW similarity + HIGH risk

**Run the REST API**

```bash
python -m uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API explorer.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check Ollama and ChromaDB status |
| POST | `/analyse` | Analyse a URL and return a risk report |
| POST | `/baseline` | Build or rebuild a baseline for a URL |
| GET | `/baseline/{url}` | Check if a baseline exists |
| DELETE | `/baseline/{url}` | Delete a stored baseline |
| GET | `/baselines` | List all stored baselines |

**Example request**

```bash
curl -X POST http://localhost:8000/analyse \
     -H "Content-Type: application/json" \
     -d '{"url": "https://github.com/login"}'
```

**Example response**

```json
{
  "url": "https://github.com/login",
  "risk_level": "LOW",
  "trust_score": 10,
  "is_login_page": true,
  "service_name": "GitHub",
  "summary": "Legitimate GitHub login page with standard login form",
  "red_flags": [],
  "vision": {
    "is_login_page": true,
    "service_name": "GitHub",
    "legitimacy_score": 10,
    "risk_level": "LOW",
    "red_flags": [],
    "summary": "Legitimate GitHub login page with standard login form",
    "is_multistep": false
  },
  "memory": {
    "has_baseline": true,
    "similarity": 0.961,
    "drift_detected": false,
    "snapshot_count": 3,
    "baseline_stored_at": "2026-04-20T14:53:28",
    "threshold": 0.75
  },
  "analysed_at": "2026-04-20T15:10:42"
}
```

---

## Phishing simulation

A fake GitHub login page is included (`fake_github.html`) with deliberate red flags:

- Misspelled brand name ("GitHUb" instead of "GitHub")
- Wrong brand color (`#28a745` instead of `#2da44e`)
- Suspicious fine print ("By signing in you agree to share your credentials with our partners")
- Wrong copyright notice ("GitHUb, Corp")

When analysed against the real GitHub baseline:

```
Risk level:    🚨 HIGH
Trust score:   2 / 10
Similarity:    0.862  (below 0.75 threshold after tuning)
Red flags:
  • URL is local file path, not a valid GitHub URL
  • Misspelled brand name in header
  • Suspicious credential-sharing fine print
  • Wrong brand colors
```

---

## Relevance to production credential management

This project was built as a learning exercise directly connected to a real production problem: preventing credential injection into phishing pages in an enterprise password manager.

In a production system, SentinelAI would run as a local service alongside the credential manager. Before any credential is injected into a login form, the agent analyses the page. If the risk level is HIGH or the similarity score falls below threshold, injection is blocked and the user is alerted.

This eliminates the need for a browser extension — the agent operates at the OS level using headless browser automation, working across all browsers and all native application login forms.

---

## Project background

Built as part of a self-directed AI engineering study path. The goal was to learn the core patterns of production AI systems — vision models, agentic loops, tool calling, RAG, vector databases, and REST API deployment — through a single coherent project rather than isolated tutorials.

The project evolved through three versions:

- **v1** — core vision pipeline (screenshot → Ollama → risk report)
- **v2** — RAG memory layer (ChromaDB baselines + similarity comparison)
- **v3** — quality fixes (structured prompts, multi-snapshot baselines, multi-step login handling)

---

## License

MIT
