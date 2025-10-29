# © 2025 Madhan Mohan | All Rights Reserved.
# Unauthorized copying, modification, or redistribution of this file, via any medium, is strictly prohibited.

import os
import re
import json
import difflib
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from groq import Groq

# Import the execution layer you already have
# (must expose: build_response, get_known_states, get_known_crops)
from chatbot import build_response, get_known_states, get_known_crops

# -------------------- Setup --------------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Put it in your .env")

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=API_KEY)

# -------------------- Prompt --------------------
VALID_INTENTS = [
    "rainfall", "rainfall_data", "avg_rainfall", "compare_rainfall",
    "top_crops", "total_production", "crop_data",
    "correlation", "highest_rainfall_year",
]

def get_system_prompt() -> str:
    """Builds the system prompt, lazy-loading states and crops."""
    # This function now calls your lazy-loaders.
    # It will only run when the Groq API is called.
    
    known_states = get_known_states()
    known_crops = get_known_crops()
    
    # This f-string should ONLY contain the prompt text for the AI
    return f"""You are a precise NLU processor for Indian agriculture questions.
Return ONLY valid JSON with keys:
- intents: array of strings chosen from {VALID_INTENTS}
- states: array of proper-cased Indian state/UT names (pick only from KnownStates)
- years: array of integers (single years)
- crop: string or null (normalize to common names like Rice, Wheat, Maize, Sugarcane)
- top_n: integer (e.g., 5)
- last_n: integer (e.g., 3)

Rules:
- Today is 2024-05-20. "last year" = 2023, "this year" = 2024.
- *PRIORITY 0: Multi-Intent Queries:* If a query has multiple intents (e.g., 'compare_rainfall' AND 'top_crops'), you MUST apply all shared entities (like states and years) to BOTH intents.
- *PRIORITY 1: Be specific.* Only include intents that the user explicitly asks for.
  - If the user asks ONLY for "crop production", do NOT include 'compare_rainfall'.
  - If the user asks ONLY for "rainfall", do NOT include 'total_production'.
  - "correlation" or "relationship" maps ONLY to 'correlation' intent. Do not add 'rainfall_data' or 'total_production' unless explicitly asked for as a separate request.
- *PRIORITY 2: Handle ambiguity:*
  - If the query is only a year (e.g., "2010"), return intents ['rainfall_data', 'total_production'] for All-India for that year.
  - If the user gives two states and asks "which is higher" without specifying a metric, include both 'total_production' and 'compare_rainfall'.
- "top 5 crops" (or "top 3", "top 10", etc.) means intent 'top_crops' and set top_n.
- "production of [crop]" (e.g., "production of Rice") maps to 'crop_data'.
- *"data of crop production" or "all crop production" or "list crop production" should map to 'top_crops' and you MUST set 'top_n' to 100.*
- "total production" (e.g., "what was the total production") maps to 'total_production'.
- if 'rainfall data' or 'show rainfall' is asked for states/years, use 'rainfall_data'. If no states, assume All-India.
- "average rainfall" maps to 'avg_rainfall'.
- "compare rainfall" maps to 'compare_rainfall'.
- "highest rainfall year" maps to 'highest_rainfall_year'.
- KnownStates: {known_states}
- KnownCrops: {known_crops}
"""

# -------------------- Utilities --------------------
def _coerce_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _extract_years_textual(q: str) -> List[int]:
    """Extract explicit years and simple ranges from plain text."""
    years: List[int] = []
    # explicit years
    for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q):
        yi = int(y)
        if 1900 <= yi <= 2100:
            years.append(yi)

    # ranges like 2009-2013 or 2010 to 2014
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\s*(?:-|to|–)\s*(19\d{2}|20\d{2})\b", q, flags=re.IGNORECASE):
        a, b = int(m.group(1)), int(m.group(2))
        if 1900 <= a <= 2100 and 1900 <= b <= 2100:
            lo, hi = (a, b) if a <= b else (b, a)
            years.extend(list(range(lo, hi + 1)))

    # de-dup and sort
    years = sorted(set(years))
    return years

def _normalize_states(cands: List[str]) -> List[str]:
    """Match user states to the known list (FUZZY MATCHING with difflib)."""
    known = get_known_states()
    # Create a map of {lowercase_state: PrettyCaseState}
    norm_map = {s.lower(): s for s in known}
    out: List[str] = []
    
    for raw in cands:
        key = str(raw).strip().lower()
        
        # 1. Try for a perfect lowercase match
        if key in norm_map:
            out.append(norm_map[key])
            continue
        
        # 2. Try for a fuzzy match using difflib
        # This will match 'karnatka' to 'karnataka'
        matches = difflib.get_close_matches(key, norm_map.keys(), n=1, cutoff=0.8)
        if matches:
            out.append(norm_map[matches[0]])
    
    # 3. Make the final list unique but preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def _normalize_crop(c: Optional[str]) -> Optional[str]:
    if not c or not isinstance(c, str):
        return None
    c = c.strip().lower()
    if not c:
        return None
    # map a few common aliases
    aliases = {
        "paddy": "Rice",
        "arhar": "Tur",
        "rapeseed and mustard": "Rapeseed & Mustard",
        "rapeseed & mustard": "Rapeseed & Mustard",
    }
    # Title-case typical crops; handle known aliases
    title = aliases.get(c, None)
    if title:
        return title
    return c.title()

def _expand_last_n_to_years(last_n: int, q: str) -> List[int]:
    """If user said 'last N years', build a conservative window inside data range."""
    if not last_n or last_n <= 0:
        return []
    # Default to the latest crop window end = 2015
    end = 2015
    start = max(2009, end - last_n + 1)
    return list(range(start, end + 1))

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # Try to extract a JSON object from text
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

# -------------------- Main entry --------------------
def run_conversation_groq(user_q: str) -> Dict[str, Any]:
    # 1) Ask Groq to structure the query
    comp = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=400,
        messages=[{"role": "system", "content": get_system_prompt()},
                  {"role": "user", "content": user_q}],
    )
    raw = comp.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    # 2) Sanitize/augment with fallbacks
    # THIS IS WHERE THIS LOGIC BELONGS
    intents = [s for s in _coerce_list(parsed.get("intents")) if s in VALID_INTENTS]
    states_in = _coerce_list(parsed.get("states"))
    states  = _normalize_states(states_in) # Normalize what the NLU found
    years_in = _coerce_list(parsed.get("years"))
    years = [int(y) for y in years_in if isinstance(y, (int, float)) or (isinstance(y, str) and y.isdigit())]
    crop    = _normalize_crop(parsed.get("crop"))
    top_n   = parsed.get("top_n")
    last_n  = parsed.get("last_n")

    # --- START: FALLBACKS ---
    # If the model missed obvious states (or returned unrecognized tokens), try extraction
    # --- START: FALLBACKS ---
    # The NLU (Groq) might miss states. Let's find them manually in the user's query.
    # This block will run even if `states` has items, to catch extras.
    
    known_states = get_known_states()
    q_lower = user_q.lower()
    found_states_from_text: List[str] = []
    
    for state in known_states:
        # Find the state name if it appears as a whole word
        if re.search(r'\b' + re.escape(state.lower()) + r'\b', q_lower):
            found_states_from_text.append(state)

    if found_states_from_text:
        # Add these states to the list NLU gave us
        all_state_candidates = states + found_states_from_text
        # Re-normalize the complete list to be safe and remove duplicates
        states = _normalize_states(all_state_candidates)

    # If the model missed obvious years, try textual extraction
    if not years:
        extracted = _extract_years_textual(user_q)
        if extracted:
            years = extracted
    # --- END: FALLBACKS ---

    # If last_n given and no explicit years, expand window
    if (not years) and isinstance(last_n, (int, float)) and int(last_n) > 0:
        years = _expand_last_n_to_years(int(last_n), user_q)

    # Reasonable defaults to keep answers flowing
    if not intents:
        # Guess intent:
        uq = user_q.lower()
        if "top" in uq and "crop" in uq:
            intents = ["top_crops"]
        elif "total" in uq and ("crop" in uq or "production" in uq):
            intents = ["total_production"]
        elif "correlation" in uq:
            intents = ["correlation"]
        elif "rain" in uq and ("which" in uq or "more" in uq or "higher" in uq):
            intents = ["compare_rainfall"]
        elif "rain" in uq:
            intents = ["rainfall_data"]
        else:
            intents = ["crop_data"]  # safe generic

    # Normalize top_n
    if isinstance(top_n, (int, float)) and int(top_n) > 0:
        top_n = int(top_n)
    else:
        # If the query contains "top N", infer it
        m = re.search(r"\btop\s+(\d{1,2})\b", user_q.lower())
        top_n = int(m.group(1)) if m else None

    forced = {
        "intents": intents,
        "states": states,                 # [] means All-India for totals/top
        "years": years,                   # engine will clip/fallback for crops 2009–2015 if needed
        "crop": crop,                     # may be None
        "top_n": top_n,                   # may be None (engine defaults to 5 for top_crops)
        "last_n": int(last_n) if isinstance(last_n, (int, float)) else None,
    }

    # 3) Execute using your engine
    result = build_response(user_q, forced=forced)  # returns dict with 'tool_blocks', 'answer', 'artifacts'

    # 4) Shape UI payload (tool blocks + Answer)
    blocks = (result.get("tool_blocks") or "").strip()
    answer = (result.get("answer") or "").strip()
    artifacts = result.get("artifacts", {})

    # Final message exactly like your app expects
    if blocks and answer:
        text = f"{blocks}\n\nAnswer: {answer}"
    elif blocks:
        text = blocks
    else:
        text = answer or "No result."

    return {"text": text, "artifacts": artifacts}