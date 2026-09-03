"""
Healthcare Appointment Management System
Main Streamlit Entry Point
Powered by Gemini 2.5 Flash + Scikit-learn ML
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Make local packages importable
sys.path.insert(0, os.path.dirname(__file__))

# ── Page configuration (MUST be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="MediCare — Healthcare Appointment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "## MediCare\nAI-Powered Healthcare Appointment System\nBuilt with Streamlit + Gemini 2.5 Flash",
    },
)

from database import db
from modules  import booking, admin, ai_assistant
from utils.helpers import local_css


# ── Bootstrap (runs once per server process) ────────────────────────────────────
@st.cache_resource
def _bootstrap():
    db.init_db()
    from ml import predictor
    predictor.train_model()
    return True

_bootstrap()

# ── Global styles ───────────────────────────────────────────────────────────────
local_css()


# ── Home page renderer ─────────────────────────────────────────────────────────
def _render_home():
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <h1 style="font-size:2.8rem; font-weight:800; color:#0f3460;">
                🏥 MediCare
            </h1>
            <p style="font-size:1.2rem; color:#57606a; max-width:600px; margin:auto;">
                AI-Powered Healthcare Appointment Management System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 📅")
        st.markdown("**Smart Booking**")
        st.caption("Book across 8 specialties with real-time slot availability and conflict detection.")
    with c2:
        st.markdown("### 🤖")
        st.markdown("**No-Show Predictor**")
        st.caption("Gradient Boosting ML model analyses 9 patient features to predict attendance risk.")
    with c3:
        st.markdown("### 🛡️")
        st.markdown("**Admin Dashboard**")
        st.caption("Full analytics, risk management, status updates, and on-demand model retraining.")
    with c4:
        st.markdown("### 💬")
        st.markdown("**AI Assistant**")
        st.caption("Gemini 2.5 Flash chat for context-aware healthcare Q&A for patients and admins.")

    st.divider()

    # Live stats
    st.subheader("📊 Live System Stats")
    try:
        stats = db.get_dashboard_stats()
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Registered Patients",   stats["total_patients"])
        sc2.metric("Total Appointments",    stats["total_appts"])
        sc3.metric("Today's Appointments",  stats["today_appts"])
        sc4.metric("Overall No-Show Rate",  f"{stats['no_show_rate']}%")
    except Exception as e:
        st.error(f"Could not load stats: {e}")

    st.divider()

    # Quick navigation guidance
    st.subheader("🚀 Quick Start")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info(
            "**Patient?**\n\n"
            "Go to **Book Appointment** to log in / register, then book, view, "
            "or cancel your appointments.\n\n"
            "Demo: `alice@demo.com` / `demo123`"
        )
    with q2:
        st.warning(
            "**Admin?**\n\n"
            "Go to **Admin Dashboard** and log in with the admin password to manage "
            "all appointments and run analytics.\n\n"
            "Default password: `admin123`"
        )
    with q3:
        st.success(
            "**Need help?**\n\n"
            "Chat with **MediBot** (AI Assistant) for health guidance, appointment tips, "
            "or help interpreting risk scores."
        )

    st.divider()

    # ML explainer
    with st.expander("🧠 How does the No-Show Risk Predictor work?"):
        st.markdown("""
The **Gradient Boosting Classifier** analyses the following features when predicting
whether a patient is likely to miss their appointment:

| Feature | Description |
|---|---|
| Historical No-Show Rate | % of past appointments the patient missed |
| Lead Time (days) | How far in advance the booking was made |
| Total Past Appointments | Number of previous clinic visits |
| Past Cancellations | Count of previously cancelled appointments |
| Day of Week | Weekend bookings show higher no-show rates |
| Hour of Day | Early/late slots show elevated risk |
| Patient Age | Age-based patterns in attendance |
| Days Since Last Visit | Inactive patients are higher risk |
| Gender | Statistical adjustment factor |

**Risk Levels:**
- 🟢 **Low** (< 35%) — Standard scheduling
- 🟡 **Medium** (35–59%) — Email/SMS reminder recommended
- 🔴 **High** (≥ 60%) — Phone call + overbooking buffer advised

The model is trained on real appointment history stored in the local SQLite database
and can be **retrained any time** from the Admin Dashboard → ML Model tab.
        """)

    st.markdown(
        "<p style='text-align:center; color:#9ba3af; font-size:12px; margin-top:2rem;'>"
        "Made with IBM Bob  •  MediCare v1.0  •  Streamlit + Gemini 2.5 Flash + Scikit-learn"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;font-size:3rem;'>🏥</div>"
        "<h2 style='text-align:center;margin:0;'>MediCare</h2>"
        "<p style='text-align:center;font-size:0.8rem;color:#9ba3af;margin:0;'>AI-Powered Healthcare</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Home",
            "📅 Book Appointment",
            "🛡️ Admin Dashboard",
            "🤖 AI Assistant",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Tech Stack**")
    st.caption("🗄️ SQLite  •  Local persistent DB")
    st.caption("🤖 Gemini 2.5 Flash  •  Chat AI")
    st.caption("🧠 Gradient Boosting  •  Risk ML")
    st.caption("📊 Plotly  •  Analytics charts")
    st.divider()
    st.markdown(
        "<p style='font-size:11px;color:#9ba3af;text-align:center;'>"
        "MediCare v1.0<br>Made with IBM Bob</p>",
        unsafe_allow_html=True,
    )


# ── Page Router ──────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    _render_home()

elif page == "📅 Book Appointment":
    booking.show()

elif page == "🛡️ Admin Dashboard":
    admin.show()

elif page == "🤖 AI Assistant":
    ai_assistant.show()
