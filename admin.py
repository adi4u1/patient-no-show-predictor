"""
Admin Dashboard Module
Full clinic management: appointment oversight, patient records,
analytics charts, ML risk management, and model retraining.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db
from ml import predictor
from utils.helpers import risk_badge, status_emoji, format_date

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def show():
    if not st.session_state.get("admin_logged_in"):
        _admin_login()
    else:
        _render_dashboard()


def _admin_login():
    st.markdown("## 🔐 Admin Login")
    st.divider()
    with st.form("admin_login"):
        pwd    = st.text_input("Admin Password", type="password")
        submit = st.form_submit_button("Login as Admin", use_container_width=True)
    if submit:
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_logged_in = True
            st.success("Access granted!")
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.info("💡 Default admin password: `admin123`")


def _render_dashboard():
    # Sidebar logout
    st.sidebar.markdown("### 🛡️ Admin Panel")
    if st.sidebar.button("🚪 Logout Admin", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.rerun()

    st.markdown("## 🏥 Admin Dashboard")
    st.divider()

    tab_overview, tab_appts, tab_patients, tab_risk, tab_model = st.tabs([
        "📊 Overview",
        "📋 Appointments",
        "👥 Patients",
        "⚠️ Risk Management",
        "🤖 ML Model",
    ])

    with tab_overview:
        _render_overview()
    with tab_appts:
        _render_appointments()
    with tab_patients:
        _render_patients()
    with tab_risk:
        _render_risk_management()
    with tab_model:
        _render_model_tab()


# ──────────────────────────────────────────────
# Overview Tab
# ──────────────────────────────────────────────

def _render_overview():
    stats = db.get_dashboard_stats()

    st.subheader("📈 Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients",       stats["total_patients"])
    c2.metric("Total Appointments",   stats["total_appts"])
    c3.metric("Today's Appointments", stats["today_appts"])
    c4.metric("No-Show Rate",         f"{stats['no_show_rate']}%")
    c5.metric("High-Risk Scheduled",  stats["high_risk_count"])

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Appointments by Status")
        by_status = stats["by_status"]
        if by_status:
            df_status = pd.DataFrame(by_status)
            fig = px.pie(
                df_status,
                names="status",
                values="cnt",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No appointment data yet.")

    with col_right:
        st.subheader("Appointments by Specialty")
        by_spec = stats["by_specialty"]
        if by_spec:
            df_spec = pd.DataFrame(by_spec)
            fig = px.bar(
                df_spec,
                x="cnt",
                y="specialty",
                orientation="h",
                color="cnt",
                color_continuous_scale="Blues",
                labels={"cnt": "Count", "specialty": ""},
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No specialty data yet.")

    # Daily trend
    st.subheader("📅 Daily Appointment Trend (Last 30 Days)")
    trend = stats["daily_trend"]
    if trend:
        df_trend = pd.DataFrame(trend)
        fig = px.area(
            df_trend,
            x="appt_date",
            y="cnt",
            labels={"appt_date": "Date", "cnt": "Appointments"},
            color_discrete_sequence=["#3b82d4"],
        )
        fig.update_layout(margin=dict(t=10, b=0), height=250)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data for the past 30 days.")


# ──────────────────────────────────────────────
# Appointments Tab
# ──────────────────────────────────────────────

def _render_appointments():
    st.subheader("📋 All Appointments")

    appts = db.get_all_appointments()
    if not appts:
        st.info("No appointments found.")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.multiselect(
            "Status", ["Scheduled", "Completed", "Cancelled", "No-Show"],
            default=["Scheduled"],
        )
    with col_f2:
        all_docs    = sorted(set(a["doctor_name"] for a in appts))
        doc_filter  = st.multiselect("Doctor", all_docs)
    with col_f3:
        risk_filter = st.selectbox("Risk Level", ["All", "High (≥60%)", "Medium (35–59%)", "Low (<35%)"])

    filtered = appts
    if status_filter:
        filtered = [a for a in filtered if a["status"] in status_filter]
    if doc_filter:
        filtered = [a for a in filtered if a["doctor_name"] in doc_filter]
    if risk_filter == "High (≥60%)":
        filtered = [a for a in filtered if a.get("no_show_risk", 0) >= 0.60]
    elif risk_filter == "Medium (35–59%)":
        filtered = [a for a in filtered if 0.35 <= a.get("no_show_risk", 0) < 0.60]
    elif risk_filter == "Low (<35%)":
        filtered = [a for a in filtered if a.get("no_show_risk", 0) < 0.35]

    st.markdown(f"Showing **{len(filtered)}** appointment(s)")

    if not filtered:
        st.info("No appointments match the selected filters.")
        return

    # Build display dataframe
    rows = []
    for a in filtered:
        rows.append({
            "ID":         a["id"],
            "Patient":    a["patient_name"],
            "Doctor":     a["doctor_name"],
            "Specialty":  a["specialty"],
            "Date":       format_date(a["appt_date"]),
            "Time":       a["appt_time"],
            "Status":     f"{status_emoji(a['status'])} {a['status']}",
            "Risk":       risk_badge(a.get("no_show_risk", 0)),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Update Appointment Status")
    appt_ids   = [a["id"] for a in filtered if a["status"] == "Scheduled"]
    if not appt_ids:
        st.info("No scheduled appointments to update.")
        return

    sel_id     = st.selectbox("Select Appointment ID", appt_ids)
    new_status = st.selectbox("New Status", ["Completed", "No-Show", "Cancelled"])
    notes      = st.text_input("Admin Notes (optional)")

    if st.button("Update Status", type="primary"):
        db.update_appointment_status(sel_id, new_status, notes)
        st.success(f"Appointment #{sel_id} marked as **{new_status}**.")
        st.rerun()


# ──────────────────────────────────────────────
# Patients Tab
# ──────────────────────────────────────────────

def _render_patients():
    st.subheader("👥 Registered Patients")

    patients = db.get_all_patients()
    if not patients:
        st.info("No patients registered yet.")
        return

    rows = []
    for p in patients:
        history = db.get_patient_history(p["id"])
        ns_rate = (
            sum(1 for h in history if h["no_show"] == 1) / len(history) * 100
            if history else 0
        )
        rows.append({
            "ID":             p["id"],
            "Name":           p["name"],
            "Email":          p["email"],
            "Phone":          p.get("phone", "—"),
            "Blood Type":     p.get("blood_type", "—"),
            "Total Visits":   len(history),
            "No-Show Rate":   f"{ns_rate:.0f}%",
            "Member Since":   format_date(p.get("created_at", "")),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Patient Deep-Dive")
    selected_id = st.selectbox(
        "Select Patient ID",
        [p["id"] for p in patients],
        format_func=lambda i: next(
            (f"#{i} — {p['name']}" for p in patients if p["id"] == i), str(i)
        ),
    )
    if selected_id:
        _patient_detail(selected_id)


def _patient_detail(patient_id: int):
    patient = db.get_patient(patient_id)
    appts   = db.get_patient_appointments(patient_id)
    history = db.get_patient_history(patient_id)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {patient['name']}")
        st.markdown(f"**Email:** {patient['email']}")
        st.markdown(f"**Phone:** {patient.get('phone', '—')}")
    with col2:
        st.markdown(f"**DOB:** {format_date(patient.get('dob', '—'))}")
        st.markdown(f"**Gender:** {patient.get('gender', '—')}")
        st.markdown(f"**Blood Type:** {patient.get('blood_type', '—')}")

    if history:
        ns   = sum(1 for h in history if h["no_show"] == 1)
        comp = sum(1 for h in history if h["status"] == "Completed")
        c1, c2, c3 = st.columns(3)
        c1.metric("Past Appointments", len(history))
        c2.metric("Completed",         comp)
        c3.metric("No-Shows",          ns)

    if appts:
        st.markdown("**Upcoming / Recent Appointments**")
        for a in appts[:5]:
            st.caption(
                f"{status_emoji(a['status'])} {format_date(a['appt_date'])} {a['appt_time']} "
                f"— {a['doctor_name']} ({a['specialty']}) — {a['status']}"
            )


# ──────────────────────────────────────────────
# Risk Management Tab
# ──────────────────────────────────────────────

def _render_risk_management():
    st.subheader("⚠️ No-Show Risk Management")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Recalculate All Risk Scores", use_container_width=True, type="primary"):
            with st.spinner("Recalculating…"):
                predictor.update_all_risk_scores()
            st.success("All risk scores updated!")
            st.rerun()

    appts = db.get_all_appointments()
    scheduled = [a for a in appts if a["status"] == "Scheduled"]

    if not scheduled:
        st.info("No scheduled appointments.")
        return

    # Risk distribution chart
    risk_scores = [a.get("no_show_risk", 0) for a in scheduled]
    high   = sum(1 for r in risk_scores if r >= 0.60)
    medium = sum(1 for r in risk_scores if 0.35 <= r < 0.60)
    low    = sum(1 for r in risk_scores if r < 0.35)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Risk",   high)
    c2.metric("🟡 Medium Risk", medium)
    c3.metric("🟢 Low Risk",    low)

    fig = go.Figure(go.Histogram(
        x=risk_scores,
        nbinsx=20,
        marker_color="#3b82d4",
        opacity=0.8,
    ))
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="No-Show Risk Score",
        yaxis_title="Count",
        height=300,
        margin=dict(t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # High-risk table
    st.subheader("🔴 High-Risk Appointments Requiring Attention")
    high_risk = [a for a in scheduled if a.get("no_show_risk", 0) >= 0.50]
    high_risk.sort(key=lambda a: a.get("no_show_risk", 0), reverse=True)

    if not high_risk:
        st.success("No high-risk appointments at the moment.")
        return

    rows = []
    for a in high_risk:
        rows.append({
            "Patient":   a["patient_name"],
            "Doctor":    a["doctor_name"],
            "Date":      format_date(a["appt_date"]),
            "Time":      a["appt_time"],
            "Risk Score": f"{a['no_show_risk']:.1%}",
            "Risk Level": risk_badge(a["no_show_risk"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📋 Intervention Recommendations"):
        st.markdown("""
        **For High-Risk Patients (≥60%):**
        - 📞 Schedule a confirmation phone call 48 hours before appointment
        - 📱 Send automated SMS/email reminder 24 hours before
        - 🎫 Consider overbooking buffer for this slot

        **For Medium-Risk Patients (35–59%):**
        - 📧 Send email reminder 2 days before
        - 💬 Include directions and parking info to reduce friction

        **General Best Practices:**
        - Track patterns monthly and retrain model quarterly
        - Offer rescheduling options proactively to high-risk patients
        """)


# ──────────────────────────────────────────────
# ML Model Tab
# ──────────────────────────────────────────────

def _render_model_tab():
    st.subheader("🤖 ML No-Show Predictor — Model Management")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🔁 Retrain Model on Latest Data", use_container_width=True, type="primary"):
            with st.spinner("Training…"):
                msg = predictor.train_model(force=True)
            st.success(msg)

    with col_t2:
        if st.button("📊 Evaluate Model Performance", use_container_width=True):
            with st.spinner("Evaluating…"):
                perf = predictor.get_model_performance()
            if perf.get("available"):
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",     f"{perf['accuracy']}%")
                mc2.metric("Precision",    f"{perf['precision_ns']}%")
                mc3.metric("Recall",       f"{perf['recall_ns']}%")
                mc4.metric("F1-Score",     f"{perf['f1_ns']}%")
                st.caption(f"Evaluated on {perf['training_size']} records "
                           f"({perf['support']} no-show samples)")
            else:
                st.warning("Not enough history data to evaluate. Need at least 20 records.")

    st.divider()

    st.subheader("Feature Importance Guide")
    features = [
        ("Historical No-Show Rate",    "% of past appointments the patient missed"),
        ("Lead Time (days)",           "How far in advance the appointment was booked"),
        ("Total Past Appointments",    "Number of previous clinic visits"),
        ("Cancellations",              "Number of previously cancelled appointments"),
        ("Day of Week",                "Weekend appointments show higher no-show rates"),
        ("Hour of Day",                "Very early/late slots show higher risk"),
        ("Patient Age",                "Younger patients tend to have higher no-show rates"),
        ("Days Since Last Visit",      "Inactive patients are higher risk"),
    ]
    df_feat = pd.DataFrame(features, columns=["Feature", "Description"])
    st.dataframe(df_feat, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔬 Manual Risk Predictor")
    st.caption("Test the model with a specific patient and appointment.")

    patients  = db.get_all_patients()
    doctors   = db.get_all_doctors()
    if not patients or not doctors:
        st.info("Need at least one patient and doctor.")
        return

    mp_col1, mp_col2 = st.columns(2)
    with mp_col1:
        sel_patient = st.selectbox(
            "Patient",
            patients,
            format_func=lambda p: f"{p['name']} (#{p['id']})",
        )
        test_date = st.date_input("Test Date", min_value=date.today())
    with mp_col2:
        sel_doctor  = st.selectbox(
            "Doctor",
            doctors,
            format_func=lambda d: f"{d['name']} — {d['specialty']}",
        )
        test_time   = st.selectbox("Test Time", ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"])

    if st.button("🔮 Predict Risk", use_container_width=True):
        with st.spinner("Running prediction…"):
            result = predictor.predict_no_show_risk(
                sel_patient["id"], test_date.isoformat(), test_time, sel_doctor["id"]
            )
        st.markdown(f"### {result['risk_color']} {result['risk_label']} Risk — {result['risk_score']:.1%}")
        for factor in result["factors"]:
            st.markdown(f"- {factor}")

        feat = result["features"]
        feat_df = pd.DataFrame([{
            "Feature":  k.replace("_", " ").title(),
            "Value":    round(v, 3) if isinstance(v, float) else v,
        } for k, v in feat.items()])
        with st.expander("Raw Feature Values"):
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
