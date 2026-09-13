"""
Shared utility helpers for the Healthcare App.
"""

import streamlit as st
from datetime import date, time, timedelta, datetime


def generate_time_slots(start_hour=9, end_hour=17, interval_minutes=30) -> list:
    """Generate a list of appointment time strings."""
    slots = []
    current = datetime(2000, 1, 1, start_hour, 0)
    end     = datetime(2000, 1, 1, end_hour, 0)
    while current < end:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=interval_minutes)
    return slots


def available_dates(num_days: int = 30) -> list:
    """Return a list of upcoming weekday dates (as date objects)."""
    dates  = []
    cursor = date.today() + timedelta(days=1)   # start from tomorrow
    while len(dates) < num_days:
        if cursor.weekday() < 5:  # Mon–Fri
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def risk_badge(score: float) -> str:
    if score >= 0.60:
        return "🔴 High"
    elif score >= 0.35:
        return "🟡 Medium"
    return "🟢 Low"


def status_emoji(status: str) -> str:
    mapping = {
        "Scheduled": "📅",
        "Completed":  "✅",
        "Cancelled":  "❌",
        "No-Show":    "⚠️",
    }
    return mapping.get(status, "❓")


def format_date(d: str) -> str:
    try:
        return datetime.fromisoformat(d).strftime("%b %d, %Y")
    except Exception:
        return d


def local_css():
    """Inject minimal Streamlit-compatible CSS theming."""
    st.markdown(
        """
        <style>
        /* Sidebar branding */
        [data-testid="stSidebar"] { background: #0f3460; }
        [data-testid="stSidebar"] * { color: #e8eaf6 !important; }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label { color: #ffffff !important; }

        /* Metric card polish */
        [data-testid="metric-container"] {
            background: #f0f4ff;
            border: 1px solid #c7d2fe;
            border-radius: 10px;
            padding: 8px 12px;
        }
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        /* Dividers */
        hr { border-color: #e2e8f0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
