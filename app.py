# app.py — Project Samarth Q&A (download button + on-demand chart + google button)
# © 2025 Madhan Mohan | All Rights Reserved.
# Unauthorized copying, modification, or redistribution of this file, via any medium, is strictly prohibited.


import time
import re
import io
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from urllib.parse import quote_plus

# ---- Brain (Groq) ----
try:
    from chatbot_brain_groq import run_conversation_groq
except ImportError:
    st.error("Error: Could not import run_conversation_groq from chatbot_brain_groq.py.")
    st.stop()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Project Samarth Q&A",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME + LAYOUT ----------
st.markdown("""
<style>
:root { --bg:#0b1220; --muted:#93a3b0; }
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1100px 600px at 18% -12%, rgba(124,58,237,.22), transparent 60%),
              radial-gradient(800px 480px at 90% 8%, rgba(16,185,129,.16), transparent 60%),
              var(--bg)!important;
}
.block-container { padding-top: 1.75rem; }

/* HEADER */
.header-container {
  position: sticky; top: 10px; z-index: 999;
  display:flex; justify-content:space-between; align-items:center;
  background:linear-gradient(90deg, rgba(28,25,60,0.93), rgba(17,24,39,0.93));
  border:1px solid rgba(148,163,184,0.25); border-radius:14px;
  padding:1.0rem 1.4rem; margin-bottom:1.0rem; box-shadow:0 6px 16px rgba(0,0,0,0.3);
  backdrop-filter: blur(6px);
}
.header-left { flex:1; }
.header-right { min-width:230px; text-align:right; }
.app-title { font-size:2.05rem; font-weight:900; color:#f8fafc; margin:0; }
.app-subtitle { color:var(--muted); margin-top:.3rem; margin-bottom:.6rem; font-weight:500; }
.badge{display:inline-block;padding:5px 12px;border-radius:999px;font-size:.82rem;font-weight:700;margin-right:8px;margin-bottom:8px;color:#e9e5ff;background:linear-gradient(135deg, rgba(124,58,237,.18), rgba(99,102,241,.18));border:1px solid rgba(180,156,255,.35);backdrop-filter: blur(6px);}
.header-card{background:rgba(16,26,43,0.75);border:1px solid rgba(148,163,184,0.25);border-radius:14px;padding:10px 14px;display:inline-block;text-align:left;box-shadow:0 4px 10px rgba(0,0,0,0.3);}
.header-card b{color:#a78bfa;}

/* CHAT */
.chat-bubble-user,.chat-bubble-bot{
  border-radius:16px;padding:14px 16px;margin:6px 0 10px;border:1px solid rgba(148,163,184,.20);box-shadow:0 6px 14px rgba(0,0,0,.22);
}
.chat-bubble-user{background:linear-gradient(135deg, rgba(124,58,237,.35), rgba(59,130,246,.25));color:#f1f5f9;}
.chat-bubble-bot{background:rgba(16, 26, 43, .72);color:#e5e7eb;}
.answer-block{background:linear-gradient(135deg, rgba(16,185,129,.20), rgba(45,212,191,.15));border:1px solid rgba(45,212,191,.45);padding:14px 16px;border-radius:14px;color:#eafff7;font-weight:700;box-shadow:0 10px 20px rgba(0,0,0,.25);}

/* INPUT with search icon */
[data-testid="stChatInput"]{position:sticky;bottom:0;z-index:1000;}
[data-testid="stChatInput"]>div{position:relative;padding-left:42px;background:rgba(16, 26, 43, .75);border:1px solid rgba(148,163,184,.30);border-radius:16px;box-shadow:0 8px 18px rgba(0,0,0,.25);}
[data-testid="stChatInput"]>div::before{content:"🔎";position:absolute;left:12px;top:50%;transform:translateY(-50%);font-size:1.1rem;opacity:.9;}
[data-testid="stChatInputTextArea"] textarea { color:#f3f4f6!important; }

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#7c3aed,#6366f1)!important;color:#fff!important;border:none!important;border-radius:12px!important;font-weight:700!important;box-shadow:0 8px 20px rgba(124,58,237,.35)!important;}
.stButton>button:hover{filter:brightness(1.04);}
.link-btn{display:inline-block;padding:.55rem .85rem;border:1px solid rgba(148,163,184,.35);border-radius:10px;text-decoration:none;margin-left:.5rem;}
.footer-note{font-size:.9rem;color:#93a3b0;}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def google_url(q: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9\\s]", "", q).strip()
    return f"https://www.google.com/search?q={quote_plus(clean)}"

def render_csv_download(label: str, df: pd.DataFrame, key_base: str):
    """Unique-keyed CSV download button for each artifact instance."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    payload = buf.getvalue()
    digest = hashlib.md5((key_base + "||" + str(df.shape) + "||" + str(hash(payload))).encode()).hexdigest()
    unique_key = f"dl_{digest}"
    st.download_button(
        label=f"⬇️ Download {label}.csv",
        data=payload,
        file_name=f"{label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=unique_key,
    )

def render_bar_chart(label: str, df: pd.DataFrame):
    if {"Crop", "Total_ProductionMT"}.issubset(df.columns):
        x_col, y_col = "Crop", "Total_ProductionMT"
    elif {"Crop", "Total_Production"}.issubset(df.columns):
        x_col, y_col = "Crop", "Total_Production"
    elif {"State", "Total_ProductionMT"}.issubset(df.columns):
        x_col, y_col = "State", "Total_ProductionMT"
    elif {"State", "Annual_Rainfall"}.issubset(df.columns):
        x_col, y_col = "State", "Annual_Rainfall"
    else:
        return
    # Sort by value, descending, so the chart looks organized
    df_sorted = df.sort_values(by=y_col, ascending=True)
    x, y = df_sorted[x_col], df_sorted[y_col]
    num_bars = len(x)
    fig_height = max(6.0, num_bars * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height)) # (width, height)
    ax.barh(x, y)
    
    ax.set_title(label)
    if "Production" in y_col:
         ax.set_xlabel(y_col)
    elif "Rainfall" in y_col:
         ax.set_xlabel(y_col + " (mm)")
    plt.tight_layout()
    st.pyplot(fig)
def render_artifact_actions(idx_prefix: str, origin_q: str, artifacts: dict):
    """Download CSV | Show/Hide chart | Google buttons under a result."""
    if not artifacts:
        return
    for name, df in (artifacts or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        label = name.replace("_", " ").title()
        key_base = f"{idx_prefix}:{name}"

        cols = st.columns([0.34, 0.28, 0.38])
        with cols[0]:
            render_csv_download(label, df, key_base)
        with cols[1]:
            opened = st.session_state.charts_open.get(key_base, False)
            if st.button(("📈 Show chart" if not opened else "❌ Hide chart"), key=f"btn_chart_{key_base}"):
                st.session_state.charts_open[key_base] = not opened
                st.rerun()
        with cols[2]:
            g_url = google_url(origin_q)
            st.markdown(f'<a class="link-btn" href="{g_url}" target="_blank">🔎 Google</a>', unsafe_allow_html=True)

        if st.session_state.charts_open.get(key_base, False):
            render_bar_chart(label, df)

# ---------- HEADER ----------
st.markdown("""
<div class="header-container">
  <div class="header-left">
    <div class="app-title">🌾 Project Samarth — Agriculture × Climate Q&A</div>
    <div class="app-subtitle">Ask questions about IMD rainfall (1901–2015) & MoA crop output (2009–2015).</div>
    <div>
      <span class="badge">Rainfall</span>
      <span class="badge">Top crops</span>
      <span class="badge">Totals</span>
      <span class="badge">Correlation</span>
    </div>
  </div>
  <div class="header-right">
    <div class="header-card">
      <div style="font-weight:700; color:#f3f4f6;">📍 State-level</div>
      <div style="line-height:1.4; font-size:0.9rem; color:#d1d5db; margin-top:4px;">
        <b>Rainfall:</b> 1901–2015<br>
        <b>Crops:</b> 2009–2015
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🔎 About")
    st.write("Local prototype over SQLite built from official CSVs. Citations are printed for traceability.")
    st.write("---")

    st.subheader("⚡ Quick Examples")
    examples = [
    # --- 1. Basic Intents (Single) ---
    "What was the rainfall in Odisha in 2014",
    "Production of Rice in Karnataka in 2012",
    "Show me the top 3 crops in Gujarat in 2011",
    "Which state had the highest rainfall in 2015",

    # --- 2. Testing Specific Fixes ---
    "Compare the average rainfall in Karnataka and Kerala from 2005 to 2008",          # Flaw 2 (Rainfall Range)
    "Give me rainfall data for 2010",                                                 # Flaw 3 (Rainfall All-India)
    "Give me data of crop production in year 2011",                                   # Flaw 4 (All Crops List)

    # --- 3. Multi-Intent & Summary (Challenge) ---
    "Which state, Karnataka or Kerala, had highest rainfall in 2010 and also tell me their top two crops",  # Flaw 5
    "Which state had more total crop production in 2014 — Tamil Nadu or Gujarat",

    # --- 4. Correlation & Trend Analysis (Advanced) ---
    "Analyze the production trend of Rice in Odisha over the last decade and correlate this with rainfall for the same period",
    "Show me the relationship between rainfall and production in Karnataka",
]

    for i, ex in enumerate(examples, 1):
        if st.button(f"➡️ {ex}", use_container_width=True, key=f"ex_{i}"):
            st.session_state._prefill = ex
            st.session_state._submit_trigger = True

    st.write("---")
    st.subheader("🧾 Transcript")
    if "messages" in st.session_state and st.session_state.messages:
        txt = [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
        st.download_button(
            "Download .txt",
            "\n\n".join(txt),
            "samarth_chat.txt",
            "text/plain",
            key="dl_transcript_file",   # fixed unique key
        )

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role","content","artifacts","q"}
if "charts_open" not in st.session_state:
    st.session_state.charts_open = {}  # f"{idx}:{artifact}" -> bool

# ---------- RENDER HISTORY ----------
for idx, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{m["content"]}</div>', unsafe_allow_html=True)
        continue

    content = m["content"]
    origin_q = m.get("q", m["content"])

    if "\n\nAnswer:" in content:
        tools, answer = content.split("\n\nAnswer:", 1)
        if tools.strip():
            with st.expander("📦 Tool outputs (traceable)", expanded=False):
                st.code(tools.strip(), language="markdown")
        st.markdown(f'<div class="answer-block">Answer:{answer}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{content}</div>', unsafe_allow_html=True)

    # --- Per-artifact action row: Download | Show/Hide Chart | Google ---
    arts = m.get("artifacts") or {}
    render_artifact_actions(str(idx), origin_q, arts)

st.write("---")

# ---------- INPUT ----------
# Always render the chat input bar so it's always visible and sticky
chat_entry = st.chat_input(
    placeholder="🔎 Ask your question… e.g., Which were the top 5 crops in Karnataka in 2010",
    key="chat_input",
)

prefill = st.session_state.pop("_prefill", "") if "_prefill" in st.session_state else ""
user_q = None  # This will hold the query to be processed

if prefill and ("_submit_trigger" in st.session_state and st.session_state.pop("_submit_trigger")):
    user_q = prefill  # Process the prefill from the sidebar
elif chat_entry:
    user_q = chat_entry  # Process the user's typed input

# ---------- HANDLE QUERY ----------
if user_q:
    st.markdown(f'<div class="chat-bubble-user">{user_q}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.spinner("Thinking…"):
        res = run_conversation_groq(user_q)  # {"text": str, "artifacts": dict}
        text = (res.get("text") or "").strip()
        artifacts = res.get("artifacts") or {}

    if not text:
        text = f"🌐 Try Google: {google_url(user_q)}"

    with st.empty():
        buf = ""
        for line in text.splitlines():
            buf += line + "\n"
            time.sleep(0.015)
            st.markdown(f'<div class="chat-bubble-bot">{buf}▌</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-bot">{text}</div>', unsafe_allow_html=True)
    render_artifact_actions(f"live-{len(st.session_state.messages)}", user_q, artifacts)

    # now persist in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "artifacts": artifacts,
        "q": user_q
    })

# ---------- FOOTER ----------
left, _, right = st.columns([0.25, 0.5, 0.25])
with left:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.charts_open = {}
        st.rerun()
with right:
    st.markdown('<div class="footer-note">Built for the Bharat Digital Fellowship — Project Samarth.</div>', unsafe_allow_html=True)
