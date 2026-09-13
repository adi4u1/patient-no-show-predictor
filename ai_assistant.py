"""
Gemini AI Assistant
Powered by google-genai SDK (new).
Provides context-aware healthcare Q&A for both patients and admins.
"""

import streamlit as st
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are MediBot, an AI health assistant for a clinic appointment management system.
Your role:
- Help patients understand their appointments, no-show risk scores, and how to prepare for visits.
- Answer general health-related questions (symptoms, specialties, when to seek care).
- Help admins interpret analytics, understand risk factors, and manage appointments.
- Explain the ML no-show predictor: what it does, what features it uses, and what the scores mean.
- Always recommend consulting a qualified doctor for medical diagnoses or treatment decisions.
- Keep responses concise, clear, and professional.
- Do NOT fabricate patient data or medical records.
- If asked about a specific patient's data, explain you can only see what is shared in the conversation.
Tone: warm, professional, trustworthy.
"""


def _get_api_key() -> str:
    # Try env first, then Streamlit secrets, then session state
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    if not key:
        key = st.session_state.get("gemini_api_key", "")
    return key.strip()


def _get_client():
    """Return a cached google-genai Client."""
    if "genai_client" not in st.session_state or st.session_state.genai_client is None:
        api_key = _get_api_key()
        if not api_key:
            return None
        st.session_state.genai_client = genai.Client(api_key=api_key)
    return st.session_state.genai_client


def _build_context() -> str:
    """Build a context snippet from the current patient/admin session."""
    context_lines = []
    patient = st.session_state.get("patient")
    if patient:
        appts   = db.get_patient_appointments(patient["id"])
        history = db.get_patient_history(patient["id"])
        ns_rate = (
            sum(1 for h in history if h["no_show"] == 1) / len(history) * 100
            if history else 0
        )
        upcoming = [a for a in appts if a["status"] == "Scheduled"]
        context_lines.append(f"[Patient: {patient['name']}, Blood Type: {patient.get('blood_type','?')}]")
        context_lines.append(f"[No-show rate: {ns_rate:.0f}%, Past visits: {len(history)}]")
        if upcoming:
            a = upcoming[0]
            context_lines.append(
                f"[Next appointment: {a['appt_date']} {a['appt_time']} with {a['doctor_name']} "
                f"({a['specialty']}), risk: {a.get('no_show_risk', 0):.0%}]"
            )

    if st.session_state.get("admin_logged_in"):
        stats = db.get_dashboard_stats()
        context_lines.append(
            f"[Admin context — Patients: {stats['total_patients']}, "
            f"Appointments today: {stats['today_appts']}, "
            f"No-show rate: {stats['no_show_rate']}%, "
            f"High-risk scheduled: {stats['high_risk_count']}]"
        )

    return "\n".join(context_lines)


def show():
    st.markdown("## 🤖 MediBot — AI Health Assistant")
    st.caption(f"Powered by **Gemini 2.5 Flash**  •  {datetime.now().strftime('%B %d, %Y')}")
    st.divider()

    if not GEMINI_AVAILABLE:
        st.error(
            "The `google-genai` package is not installed. "
            "Run `pip install google-genai` to enable MediBot."
        )
        return

    # ── API Key setup ────────────────────────────
    api_key = _get_api_key()
    if not api_key:
        st.warning("🔑 Gemini API key not configured.")
        with st.expander("🔧 Configure API Key", expanded=True):
            st.markdown(
                "Get a free API key from "
                "[Google AI Studio](https://aistudio.google.com/app/apikey)."
            )
            key_input = st.text_input(
                "Enter your Gemini API Key",
                type="password",
                placeholder="AIza…",
            )
            if st.button("Save Key & Connect", type="primary"):
                if key_input.strip():
                    st.session_state.gemini_api_key = key_input.strip()
                    st.session_state.genai_client   = None  # reset client
                    st.session_state.chat_history   = []    # reset history
                    st.success("API key saved! MediBot is ready.")
                    st.rerun()
                else:
                    st.error("Please enter a valid key.")
        return

    # ── Chat UI ──────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role":    "assistant",
                "content": (
                    "👋 Hello! I'm **MediBot**, your AI health assistant. "
                    "I can help you with:\n"
                    "- 📅 Understanding your appointments and risk scores\n"
                    "- 🏥 Questions about medical specialties\n"
                    "- 📊 Interpreting analytics (for admins)\n"
                    "- 💊 General health guidance\n\n"
                    "How can I help you today?"
                ),
            }
        ]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested prompts (shown only at start)
    if len(st.session_state.chat_messages) == 1:
        st.markdown("**Quick prompts:**")
        suggestions = [
            "What does my no-show risk score mean?",
            "How should I prepare for a cardiology appointment?",
            "What factors affect no-show predictions?",
            "Explain the high-risk patients in the dashboard",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sugg_{i}", use_container_width=True):
                _send_message(s)

    # Chat input
    user_input = st.chat_input("Ask MediBot anything…")
    if user_input:
        _send_message(user_input)

    # Clear conversation button
    if len(st.session_state.chat_messages) > 1:
        if st.button("🗑️ Clear Conversation", use_container_width=False):
            st.session_state.chat_messages = [st.session_state.chat_messages[0]]
            st.session_state.chat_history  = []
            st.rerun()


def _send_message(user_text: str):
    """Send a message to Gemini and render the response."""
    st.session_state.chat_messages.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.markdown(user_text)

    client = _get_client()
    if client is None:
        with st.chat_message("assistant"):
            st.error("Could not initialise Gemini. Please check your API key.")
        return

    # Augment with live context
    context  = _build_context()
    full_msg = f"{context}\n\nUser: {user_text}" if context else user_text

    # Build contents list from history + new message
    contents = list(st.session_state.chat_history) + [
        types.Content(role="user", parts=[types.Part(text=full_msg)])
    ]

    with st.chat_message("assistant"):
        with st.spinner("MediBot is thinking…"):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )
                reply = response.text
                # Append both turns to history for multi-turn context
                st.session_state.chat_history.append(
                    types.Content(role="user", parts=[types.Part(text=full_msg)])
                )
                st.session_state.chat_history.append(
                    types.Content(role="model", parts=[types.Part(text=reply)])
                )
            except Exception as e:
                reply = f"⚠️ Error communicating with Gemini: {e}"
        st.markdown(reply)

    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
    st.rerun()
