"""
Patient Booking Module
Handles: registration, login, appointment booking, appointment management.
"""

import streamlit as st
from datetime import date, datetime
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db
from ml import predictor
from utils.helpers import (
    generate_time_slots, available_dates,
    risk_badge, status_emoji, format_date,
)


# ──────────────────────────────────────────────
# Auth helpers (session state)
# ──────────────────────────────────────────────

def _is_logged_in() -> bool:
    return st.session_state.get("patient") is not None


def _current_patient() -> dict:
    return st.session_state.get("patient", {})


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def show():
    """Render the full patient-facing booking portal."""
    if not _is_logged_in():
        _render_auth()
    else:
        _render_portal()


# ──────────────────────────────────────────────
# Auth — Login / Register
# ──────────────────────────────────────────────

def _render_auth():
    st.markdown("## 🏥 Patient Portal")
    st.markdown("Book and manage your healthcare appointments seamlessly.")
    st.divider()

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        _login_form()

    with tab_register:
        _register_form()


def _login_form():
    st.subheader("Welcome back!")
    with st.form("login_form"):
        email    = st.text_input("Email address", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submit   = st.form_submit_button("Login", use_container_width=True)

    if submit:
        if not email or not password:
            st.error("Please fill in all fields.")
            return
        patient = db.login_patient(email.strip().lower(), password)
        if patient:
            st.session_state.patient = patient
            st.success(f"Welcome back, {patient['name']}! 👋")
            st.rerun()
        else:
            st.error("Invalid email or password.")

    st.info("💡 **Demo credentials** — Email: `alice@demo.com`  Password: `demo123`")


def _register_form():
    st.subheader("Create your account")
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            name       = st.text_input("Full Name *", placeholder="Jane Doe")
            email      = st.text_input("Email *", placeholder="jane@example.com")
            phone      = st.text_input("Phone", placeholder="+1-555-0100")
            password   = st.text_input("Password *", type="password")
        with col2:
            dob        = st.date_input("Date of Birth", min_value=date(1920, 1, 1), max_value=date.today())
            gender     = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Prefer not to say"])
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-", "Unknown"])
            address    = st.text_input("Address", placeholder="123 Main St, City")
        submit = st.form_submit_button("Create Account", use_container_width=True)

    if submit:
        if not name or not email or not password:
            st.error("Name, email, and password are required.")
            return
        result = db.register_patient(
            name, email.strip().lower(), phone, dob.isoformat(),
            gender, blood_type, address, password,
        )
        if result["success"]:
            st.success(result["message"])
            st.info("Please go to the Login tab to sign in.")
        else:
            st.error(result["message"])


# ──────────────────────────────────────────────
# Patient Portal (post-login)
# ──────────────────────────────────────────────

def _render_portal():
    patient = _current_patient()

    # Sidebar welcome
    st.sidebar.markdown(f"### 👤 {patient['name']}")
    st.sidebar.caption(patient.get("email", ""))
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.patient = None
        st.rerun()

    st.markdown(f"## 👋 Hello, {patient['name'].split()[0]}!")
    st.divider()

    tab_book, tab_appts, tab_profile = st.tabs([
        "📅 Book Appointment",
        "📋 My Appointments",
        "👤 My Profile",
    ])

    with tab_book:
        _booking_form(patient)

    with tab_appts:
        _my_appointments(patient)

    with tab_profile:
        _patient_profile(patient)


# ──────────────────────────────────────────────
# Booking Form
# ──────────────────────────────────────────────

def _booking_form(patient: dict):
    st.subheader("📅 Book a New Appointment")

    doctors  = db.get_all_doctors()
    if not doctors:
        st.warning("No doctors available at the moment.")
        return

    # Doctor selection
    specialties = sorted(set(d["specialty"] for d in doctors))
    specialty   = st.selectbox("Filter by Specialty", ["All"] + specialties)

    filtered_docs = doctors if specialty == "All" else [
        d for d in doctors if d["specialty"] == specialty
    ]
    doc_labels    = {f"{d['name']} — {d['specialty']}": d for d in filtered_docs}
    selected_label = st.selectbox("Select Doctor", list(doc_labels.keys()))
    doctor         = doc_labels[selected_label]

    col1, col2 = st.columns(2)
    with col1:
        future_dates = available_dates(30)
        appt_date    = st.selectbox(
            "Appointment Date",
            options=future_dates,
            format_func=lambda d: d.strftime("%A, %b %d %Y"),
        )
    with col2:
        all_slots    = generate_time_slots()
        booked       = db.get_booked_slots(doctor["id"], appt_date.isoformat())
        free_slots   = [s for s in all_slots if s not in booked]
        if not free_slots:
            st.warning("No available slots for this date. Please choose another.")
            return
        appt_time = st.selectbox("Time Slot", free_slots)

    reason = st.text_area("Reason for Visit", placeholder="Brief description of your symptoms or reason…", height=100)

    # ── Live Risk Preview ──────────────────────
    st.divider()
    st.markdown("#### 🤖 AI No-Show Risk Assessment")

    with st.spinner("Calculating risk…"):
        risk = predictor.predict_no_show_risk(
            patient["id"], appt_date.isoformat(), appt_time, doctor["id"]
        )

    risk_col1, risk_col2, risk_col3 = st.columns(3)
    risk_col1.metric("Risk Score",  f"{risk['risk_score']:.1%}")
    risk_col2.metric("Risk Level",  f"{risk['risk_color']} {risk['risk_label']}")
    risk_col3.metric("Past No-Shows",
                     f"{risk['features']['no_show_rate']:.0%} rate")

    if risk["risk_label"] in ("Medium", "High"):
        with st.expander("⚠️ Risk Factors Detected", expanded=True):
            for factor in risk["factors"]:
                st.markdown(f"- {factor}")
        if risk["risk_label"] == "High":
            st.warning(
                "📞 **Reminder:** Given the high risk, a confirmation call will be scheduled "
                "24 hours before your appointment."
            )

    st.divider()

    if st.button("✅ Confirm Booking", use_container_width=True, type="primary"):
        result = db.book_appointment(
            patient_id    = patient["id"],
            doctor_id     = doctor["id"],
            appt_date     = appt_date.isoformat(),
            appt_time     = appt_time,
            reason        = reason,
            no_show_risk  = risk["risk_score"],
        )
        if result["success"]:
            st.success(f"🎉 {result['message']}")
            st.balloons()
            st.info(
                f"**Summary:**\n"
                f"- Doctor: **{doctor['name']}** ({doctor['specialty']})\n"
                f"- Date: **{appt_date.strftime('%A, %B %d %Y')}** at **{appt_time}**\n"
                f"- No-show Risk: **{risk['risk_color']} {risk['risk_label']}** ({risk['risk_score']:.1%})"
            )
        else:
            st.error(result["message"])


# ──────────────────────────────────────────────
# My Appointments
# ──────────────────────────────────────────────

def _my_appointments(patient: dict):
    st.subheader("📋 Your Appointments")

    appointments = db.get_patient_appointments(patient["id"])
    if not appointments:
        st.info("You have no appointments yet. Book one from the 'Book Appointment' tab!")
        return

    # Filter
    status_filter = st.multiselect(
        "Filter by status",
        ["Scheduled", "Completed", "Cancelled", "No-Show"],
        default=["Scheduled"],
    )
    filtered = [a for a in appointments if a["status"] in status_filter] if status_filter else appointments

    st.markdown(f"Showing **{len(filtered)}** appointment(s)")

    for appt in filtered:
        risk_score = appt.get("no_show_risk", 0)
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.markdown(f"**{appt['doctor_name']}**  —  _{appt['specialty']}_")
                st.caption(f"📅 {format_date(appt['appt_date'])} at {appt['appt_time']}")
                if appt.get("reason"):
                    st.caption(f"📝 {appt['reason']}")
            with col2:
                st.markdown(f"{status_emoji(appt['status'])} **{appt['status']}**")
                if appt["status"] == "Scheduled":
                    st.caption(f"No-Show Risk: {risk_badge(risk_score)}")
            with col3:
                if appt["status"] == "Scheduled":
                    if st.button("❌ Cancel", key=f"cancel_{appt['id']}"):
                        db.cancel_appointment(appt["id"])
                        st.success("Appointment cancelled.")
                        st.rerun()


# ──────────────────────────────────────────────
# Patient Profile
# ──────────────────────────────────────────────

def _patient_profile(patient: dict):
    st.subheader("👤 Your Profile")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {patient['name']}")
        st.markdown(f"**Email:** {patient['email']}")
        st.markdown(f"**Phone:** {patient.get('phone', '—')}")
        st.markdown(f"**Date of Birth:** {format_date(patient.get('dob', '—'))}")

    with col2:
        st.markdown(f"**Gender:** {patient.get('gender', '—')}")
        st.markdown(f"**Blood Type:** {patient.get('blood_type', '—')}")
        st.markdown(f"**Address:** {patient.get('address', '—')}")
        st.markdown(f"**Member Since:** {format_date(patient.get('created_at', '—'))}")

    # History summary
    st.divider()
    st.subheader("📊 Your Appointment History")
    history = db.get_patient_history(patient["id"])
    if history:
        total  = len(history)
        ns     = sum(1 for h in history if h["no_show"] == 1)
        compl  = sum(1 for h in history if h["status"] == "Completed")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Past Visits", total)
        mc2.metric("Completed",         compl)
        mc3.metric("No-Shows",          ns)
    else:
        st.info("No appointment history yet.")
