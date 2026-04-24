"""
SentinelAI - Week 2 (Quality Fixes): agent_v3.py
==================================================
What changed from v2:

FIX 1 — Robust baseline (solves similarity score variance)
  Problem: vision model describes the same page differently every visit,
           causing the same legitimate page to score as "drift detected".
  Solution:
    - Structured description prompt forces consistent output format every time
    - Store 3 baseline snapshots per URL instead of 1
    - Compare against ALL stored baselines and take the MAX similarity score
    - This means a page only triggers drift if it looks different from ALL
      known-good snapshots, not just one

FIX 2 — Multi-step login handling (solves Microsoft false positive)
  Problem: Microsoft login shows email field first, password on next screen.
           Agent screenshots screen 1, sees no password, flags as suspicious.
  Solution:
    - After initial screenshot, check if page looks like "step 1 of multi-step"
    - If yes: find the email input, type a probe value, click Next/Continue
    - Take a second screenshot of screen 2
    - Analyse both screenshots together for the final risk assessment

Run:
    python agent_v3.py

Requirements: same as v2
    pip install playwright ollama chromadb pillow
    python -m playwright install chromium
    ollama pull qwen3-vl:8b
    ollama pull nomic-embed-text
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from datetime import datetime

import ollama
import chromadb
from playwright.async_api import async_playwright


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VISION_MODEL        = "qwen3-vl:8b"
EMBED_MODEL         = "nomic-embed-text"
SCREENSHOT_DIR      = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

SIMILARITY_THRESHOLD = 0.75   # below this = drift detected

# FIX 1: how many baseline snapshots to store per URL
# We visit the page N times and store each description separately.
# On comparison we take the MAX similarity across all stored snapshots.
BASELINE_SNAPSHOTS  = 3

# Probe email used when interacting with multi-step login forms.
# Must look realistic enough to pass client-side validation.
PROBE_EMAIL         = "probe@sentinel-check.internal"


# ─────────────────────────────────────────────
# CHROMADB — with cosine metric (FIX 1)
# ─────────────────────────────────────────────

def get_db():
    client = chromadb.PersistentClient(path="./sentinel_db_v3")
    collection = client.get_or_create_collection(
        name="page_baselines_v3",
        metadata={
            "description": "Known-good login page baselines",
            "hnsw:space": "cosine",      # ← fixes negative scores from v2
        }
    )
    return collection


# ─────────────────────────────────────────────
# SCREENSHOT
# ─────────────────────────────────────────────

async def screenshot_url(url: str, page=None) -> bytes:
    """
    Screenshot a URL. If a live Playwright page is passed in,
    screenshot that instead (used for multi-step flows).
    """
    if page is not None:
        # Screenshot an already-open page (mid-flow)
        return await page.screenshot(full_page=False)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        pg = await browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            wait = "domcontentloaded" if url.startswith("file://") else "networkidle"
            await pg.goto(url, wait_until=wait, timeout=15000)
            await pg.wait_for_timeout(1000)
            data = await pg.screenshot(full_page=False)
            safe = url.replace("://", "_").replace("/", "_").replace(".", "_")[:60]
            (SCREENSHOT_DIR / f"{safe}.png").write_bytes(data)
            return data
        finally:
            await browser.close()


# ─────────────────────────────────────────────
# FIX 2: MULTI-STEP LOGIN DETECTION + INTERACTION
# ─────────────────────────────────────────────
#
# Some login flows (Microsoft, Apple, Okta) show email on page 1
# and password on page 2. We detect this pattern and interact
# with the page to capture the full flow before analysing.

async def handle_multistep_login(url: str) -> tuple[bytes, bytes | None]:
    """
    Navigate to a URL and detect if it uses a multi-step login flow.

    Returns:
        (screenshot1, screenshot2) where screenshot2 is the post-interaction
        screenshot, or None if the page is a standard single-step form.

    How it works:
        1. Load the page, take screenshot 1
        2. Ask the vision model: is this step 1 of a multi-step flow?
        3. If yes: find the email input via DOM, type probe email, click Next
        4. Wait for page transition, take screenshot 2
        5. Return both screenshots
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        try:
            wait = "domcontentloaded" if url.startswith("file://") else "networkidle"
            await page.goto(url, wait_until=wait, timeout=15000)
            await page.wait_for_timeout(1500)

            screenshot1 = await page.screenshot(full_page=False)
            safe = url.replace("://", "_").replace("/", "_").replace(".", "_")[:55]
            (SCREENSHOT_DIR / f"{safe}_step1.png").write_bytes(screenshot1)

            # Ask vision model if this looks like step 1 of a multi-step flow
            is_multistep = detect_multistep(screenshot1)

            if not is_multistep:
                print(f"  [multistep] Single-step login detected")
                return screenshot1, None

            print(f"  [multistep] Multi-step login detected — interacting with page")

            # Find email/username input and fill it
            # Try common selectors used by major identity providers
            email_selectors = [
                "input[type='email']",
                "input[name='loginfmt']",       # Microsoft
                "input[name='email']",
                "input[name='username']",
                "input[id='email']",
                "input[id='username']",
                "input[type='text']",            # fallback
            ]

            filled = False
            for selector in email_selectors:
                try:
                    el = await page.query_selector(selector)
                    if el and await el.is_visible():
                        await el.click()
                        await el.fill(PROBE_EMAIL)
                        print(f"  [multistep] Filled email field: {selector}")
                        filled = True
                        break
                except Exception:
                    continue

            if not filled:
                print(f"  [multistep] Could not find email field — returning step 1 only")
                return screenshot1, None

            # Find and click the Next/Continue/Sign in button
            next_selectors = [
                "input[type='submit']",
                "button[type='submit']",
                "input[id='idSIButton9']",       # Microsoft "Next" button
                "button:has-text('Next')",
                "button:has-text('Continue')",
                "button:has-text('Sign in')",
                "[data-action='next']",
            ]

            clicked = False
            for selector in next_selectors:
                try:
                    el = await page.query_selector(selector)
                    if el and await el.is_visible():
                        await el.click()
                        print(f"  [multistep] Clicked next button: {selector}")
                        clicked = True
                        break
                except Exception:
                    continue

            if not clicked:
                print(f"  [multistep] Could not find Next button — returning step 1 only")
                return screenshot1, None

            # Wait for page 2 to load
            await page.wait_for_timeout(2500)
            screenshot2 = await page.screenshot(full_page=False)
            (SCREENSHOT_DIR / f"{safe}_step2.png").write_bytes(screenshot2)
            print(f"  [multistep] Captured step 2 screenshot")

            return screenshot1, screenshot2

        finally:
            await browser.close()


def detect_multistep(screenshot_bytes: bytes) -> bool:
    """
    Ask the vision model whether this looks like step 1 of a multi-step login.
    Returns True if the page shows ONLY an email/username field with no password.
    """
    b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    prompt = """Look at this webpage screenshot carefully.

Answer only: is this step 1 of a multi-step login flow?
A multi-step flow shows ONLY an email or username field on this screen,
with NO password field visible, and has a Next or Continue button.

Reply with exactly one word: YES or NO"""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64]}]
    )
    answer = response.message.content.strip().upper()
    return "YES" in answer


# ─────────────────────────────────────────────
# FIX 1: STRUCTURED DESCRIPTION PROMPT
# ─────────────────────────────────────────────
#
# The old prompt produced free-form descriptions that varied in
# structure, length and focus on every run. Two descriptions of
# the same page could be quite different just due to model variance.
#
# The new prompt forces a rigid 8-section structure. The model
# fills in the same sections every time, producing much more
# consistent embeddings for the same page.

DESCRIPTION_PROMPT = """Describe this webpage for security analysis. 
Use EXACTLY this structure with these section headers:

BRAND: [logo text, colors, brand name visible]
LAYOUT: [header, body, footer arrangement]
FORM_FIELDS: [list every input field label and type]
BUTTONS: [list every button label]
COLORS: [primary background, accent, text colors]
TRUST_SIGNALS: [copyright, security badges, padlock, legal links]
ERRORS_OR_ANOMALIES: [spelling mistakes, broken elements, suspicious text]
URL_CONTEXT: [what service would you expect at this URL]

Be specific and consistent. Use the exact section names above.
Do not add extra commentary outside these sections."""


def describe_page(screenshot_bytes: bytes) -> str:
    """
    Describe a page using the structured prompt for consistent embeddings.
    Optionally pass a second screenshot for multi-step pages.
    """
    b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": DESCRIPTION_PROMPT,
            "images": [b64],
        }]
    )
    description = response.message.content.strip()
    print(f"  [describe] Page described ({len(description)} chars)")
    return description


def describe_multistep_page(shot1: bytes, shot2: bytes) -> str:
    """
    Describe a multi-step login by describing both screens and combining them.
    This gives the embedding a complete picture of the full login flow.
    """
    desc1 = describe_page(shot1)
    desc2 = describe_page(shot2)

    # Combine both descriptions — the embedding captures the full flow
    combined = f"STEP_1_SCREEN:\n{desc1}\n\nSTEP_2_SCREEN:\n{desc2}"
    print(f"  [describe] Combined multi-step description ({len(combined)} chars)")
    return combined


# ─────────────────────────────────────────────
# EMBED
# ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    vector = response["embedding"]
    print(f"  [embed] Generated vector with {len(vector)} dimensions")
    return vector


# ─────────────────────────────────────────────
# FIX 1: MULTI-SNAPSHOT BASELINE STORE + COMPARE
# ─────────────────────────────────────────────
#
# Instead of storing one snapshot, we store BASELINE_SNAPSHOTS (3) snapshots.
# Each gets a unique ID: url_snapshot_0, url_snapshot_1, url_snapshot_2
#
# On comparison we query all stored snapshots for this URL and take
# the MAX similarity. A page only triggers drift if it looks different
# from ALL stored snapshots — not just one slightly different description.

def baseline_doc_ids(url: str) -> list[str]:
    """Generate the ChromaDB doc IDs for all snapshots of a URL."""
    base = url.replace("://", "_").replace("/", "_").replace(".", "_")[:55]
    return [f"{base}_snap_{i}" for i in range(BASELINE_SNAPSHOTS)]


def store_snapshot(url: str, snapshot_index: int, description: str,
                   embedding: list[float], collection):
    """Store one baseline snapshot for a URL."""
    doc_ids = baseline_doc_ids(url)
    doc_id = doc_ids[snapshot_index]

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[{
            "url": url,
            "snapshot_index": snapshot_index,
            "stored_at": datetime.now().isoformat(),
        }]
    )
    print(f"  [memory] Stored snapshot {snapshot_index + 1}/{BASELINE_SNAPSHOTS} for {url}")


def count_stored_snapshots(url: str, collection) -> int:
    """How many baseline snapshots do we have for this URL?"""
    ids = baseline_doc_ids(url)
    result = collection.get(ids=ids)
    return len(result["ids"])


def compare_to_baselines(url: str, embedding: list[float], collection) -> dict:
    """
    Compare against ALL stored baseline snapshots and return the MAX similarity.
    This makes the system more tolerant of description variance.
    """
    ids = baseline_doc_ids(url)
    existing = collection.get(ids=ids)

    if not existing["ids"]:
        return {"has_baseline": False, "similarity": None, "drift_detected": False}

    # Query each stored snapshot individually and collect scores
    similarities = []
    for doc_id in existing["ids"]:
        # Get the embedding for this snapshot
        snap = collection.get(ids=[doc_id], include=["embeddings", "metadatas"])
        if not snap["ids"]:
            continue

        snap_embedding = snap["embeddings"][0]

        # Compute cosine similarity manually
        # ChromaDB query returns distance — 1 - distance = similarity
        result = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"url": url},
            include=["distances", "metadatas"]
        )
        if result["distances"] and result["distances"][0]:
            distance = result["distances"][0][0]
            sim = round(1 - distance, 3)
            similarities.append(sim)
            break   # query already returns best match across all snapshots

    if not similarities:
        return {"has_baseline": False, "similarity": None, "drift_detected": False}

    # Take the MAX similarity across all snapshots
    best_similarity = max(similarities)
    drift_detected = best_similarity < SIMILARITY_THRESHOLD

    stored_at = existing["metadatas"][0].get("stored_at", "unknown") if existing["metadatas"] else "unknown"
    snapshot_count = len(existing["ids"])

    print(f"  [memory] Best similarity across {snapshot_count} snapshots: "
          f"{best_similarity:.3f} "
          f"({'DRIFT DETECTED' if drift_detected else 'looks normal'})")

    return {
        "has_baseline": True,
        "similarity": best_similarity,
        "drift_detected": drift_detected,
        "snapshot_count": snapshot_count,
        "baseline_stored_at": stored_at,
        "threshold": SIMILARITY_THRESHOLD,
    }


# ─────────────────────────────────────────────
# VISION RISK ANALYSIS
# ─────────────────────────────────────────────

def analyse_screenshots(screenshots: list[bytes], url: str) -> dict:
    """
    Run vision risk analysis. Accepts one or two screenshots.
    For multi-step pages, both screenshots are sent together so
    the model can see the complete login flow.
    """
    images_b64 = [base64.b64encode(s).decode("utf-8") for s in screenshots]

    step_note = (
        "NOTE: This is a multi-step login — two screenshots are provided. "
        "Screenshot 1 shows the email step, screenshot 2 shows the password step. "
        "Treat them as one complete login form.\n\n"
        if len(screenshots) > 1 else ""
    )

    prompt = f"""{step_note}You are a security analyst examining a webpage screenshot.
The URL of this page is: {url}

Analyse the screenshot(s) and answer:
1. Does this page contain a login form (considering both steps if multi-step)?
2. Does the design look professional and legitimate?
3. Does the URL match what you would expect for this service?
4. Are there any visual red flags?

Respond ONLY with JSON in exactly this format, no other text:
{{
  "is_login_page": true or false,
  "service_name": "name of the service or Unknown",
  "legitimacy_score": a number from 0 to 10 where 10 is definitely legitimate,
  "risk_level": "LOW" or "MEDIUM" or "HIGH",
  "red_flags": ["list", "of", "concerns"] or [],
  "summary": "one sentence summary",
  "is_multistep": true or false
}}"""

    # Build message with all screenshots
    content_parts = []
    for b64 in images_b64:
        content_parts.append({
            "type": "image",
            "data": b64,
        })

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": images_b64,
        }]
    )

    raw = response.message.content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "is_login_page": False,
            "service_name": "Unknown",
            "legitimacy_score": 5,
            "risk_level": "MEDIUM",
            "red_flags": ["Could not parse model response"],
            "summary": "Analysis inconclusive",
            "is_multistep": False,
        }


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(url: str, vision: dict, memory: dict):
    risk_icons = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}
    final_risk = vision.get("risk_level", "MEDIUM")
    if memory.get("drift_detected"):
        final_risk = "HIGH"
    icon = risk_icons.get(final_risk, "⚠️")

    multistep_tag = "  [multi-step login]" if vision.get("is_multistep") else ""

    print("\n" + "═" * 58)
    print("  SENTINELAI RISK REPORT  (v3 — quality fixes)")
    print("═" * 58)
    print(f"  URL:            {url}")
    print(f"  Service:        {vision.get('service_name', 'Unknown')}{multistep_tag}")
    print(f"  Login page:     {vision.get('is_login_page', False)}")
    print(f"  Risk level:     {icon}  {final_risk}")
    print(f"  Trust score:    {vision.get('legitimacy_score', '?')} / 10")
    print(f"  Summary:        {vision.get('summary', '')}")

    print(f"\n  Memory:")
    if not memory.get("has_baseline"):
        snap_idx = memory.get("snapshot_index", 0)
        total = memory.get("total_snapshots", BASELINE_SNAPSHOTS)
        print(f"    Stored snapshot {snap_idx + 1} of {total} — building baseline")
    else:
        sim = memory.get("similarity", 0) or 0
        filled = max(0, min(20, int(sim * 20)))
        bar = "█" * filled + "░" * (20 - filled)
        snaps = memory.get("snapshot_count", 1)
        print(f"    Similarity:   [{bar}] {sim:.3f}  (best of {snaps} snapshots)")
        print(f"    Baseline set: {memory.get('baseline_stored_at', 'unknown')[:19]}")
        if memory.get("drift_detected"):
            print(f"    ⚠️  VISUAL DRIFT DETECTED — page looks different from all baselines!")

    red_flags = list(vision.get("red_flags", []))
    if memory.get("drift_detected"):
        red_flags.append("Visual drift from known-good baseline")
    if red_flags:
        print(f"\n  Red flags:")
        for flag in red_flags:
            print(f"    • {flag}")

    print("═" * 58 + "\n")


# ─────────────────────────────────────────────
# MAIN AGENT PIPELINES
# ─────────────────────────────────────────────

async def build_baseline(url: str, collection):
    """
    Visit a URL BASELINE_SNAPSHOTS times and store each description.
    This builds a robust multi-snapshot baseline for the URL.
    """
    print(f"\n📸 Building baseline for {url} ({BASELINE_SNAPSHOTS} snapshots)")

    for i in range(BASELINE_SNAPSHOTS):
        print(f"\n  Snapshot {i + 1}/{BASELINE_SNAPSHOTS}")

        # Use multi-step handler for all URLs
        shot1, shot2 = await handle_multistep_login(url)

        if shot2 is not None:
            description = describe_multistep_page(shot1, shot2)
        else:
            description = describe_page(shot1)

        embedding = embed_text(description)
        store_snapshot(url, i, description, embedding, collection)

        # Short pause between snapshots so page state is fresh
        if i < BASELINE_SNAPSHOTS - 1:
            await asyncio.sleep(1)

    print(f"  ✅ Baseline complete — {BASELINE_SNAPSHOTS} snapshots stored")


async def check_url(url: str, collection) -> dict:
    """
    Check a URL against its stored baseline and produce a risk report.
    """
    print(f"\n🔍 Checking: {url}")

    shot1, shot2 = await handle_multistep_login(url)
    screenshots = [shot1] if shot2 is None else [shot1, shot2]

    if shot2 is not None:
        description = describe_multistep_page(shot1, shot2)
    else:
        description = describe_page(shot1)

    embedding = embed_text(description)
    memory_result = compare_to_baselines(url, embedding, collection)

    print(f"  [analyse] Running vision risk analysis...")
    vision_result = analyse_screenshots(screenshots, url)

    print_report(url, vision_result, memory_result)
    return {"url": url, "vision": vision_result, "memory": memory_result}


async def run_phishing_simulation(fake_url: str, real_url: str, collection):
    """Compare a fake page against the baseline of the real page it imitates."""
    print(f"\n🎣 Phishing simulation")
    print(f"   Fake URL:  {fake_url}")
    print(f"   Comparing against baseline of: {real_url}")

    shot1, shot2 = await handle_multistep_login(fake_url)
    screenshots = [shot1] if shot2 is None else [shot1, shot2]

    if shot2 is not None:
        description = describe_multistep_page(shot1, shot2)
    else:
        description = describe_page(shot1)

    embedding = embed_text(description)

    # Compare against REAL page's baseline
    memory_result = compare_to_baselines(real_url, embedding, collection)

    print(f"  [analyse] Running vision risk analysis...")
    vision_result = analyse_screenshots(screenshots, fake_url)

    print_report(fake_url, vision_result, memory_result)
    return {"url": fake_url, "vision": vision_result, "memory": memory_result}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    print("=" * 58)
    print("  SentinelAI — Login Page Detection Agent")
    print("  Week 2 Quality Fixes: v3")
    print(f"  Fix 1: structured descriptions + {BASELINE_SNAPSHOTS}-snapshot baselines")
    print(f"  Fix 2: multi-step login detection and interaction")
    print("=" * 58)

    # Check Ollama
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        print(f"\n✅ Ollama running. Models: {', '.join(model_names)}")
        if not any(EMBED_MODEL in m for m in model_names):
            print(f"\n⚠️  Run: ollama pull {EMBED_MODEL}")
            return
    except Exception:
        print("\n❌ Ollama is not running.")
        return

    collection = get_db()
    print(f"✅ ChromaDB ready (cosine metric). Stored docs: {collection.count()}")

    real_urls = [
        "https://github.com/login",
        "https://login.microsoftonline.com",
    ]

    # ── Pass 1: build robust baselines (3 snapshots each) ──
    print("\n── Pass 1: building baselines ──")
    for url in real_urls:
        await build_baseline(url, collection)

    # ── Pass 2: check real pages — should score HIGH similarity ──
    print("\n── Pass 2: checking real pages against baselines ──")
    for url in real_urls:
        await check_url(url, collection)

    # ── Pass 3: phishing simulation ──
    fake_html_path = os.path.abspath("fake_github.html").replace("\\", "/")
    fake_url = f"file:///{fake_html_path}"

    print("\n── Pass 3: phishing simulation ──")
    await run_phishing_simulation(
        fake_url=fake_url,
        real_url="https://github.com/login",
        collection=collection,
    )

    print(f"\n📦 ChromaDB v3 now has {collection.count()} stored documents")
    print(f"   ({BASELINE_SNAPSHOTS} snapshots × {len(real_urls)} URLs = "
          f"{BASELINE_SNAPSHOTS * len(real_urls)} baseline docs)\n")


if __name__ == "__main__":
    asyncio.run(main())
