"""
ML No-Show Risk Predictor
Uses a Random Forest Classifier trained on patient appointment history.
Features engineered from past behavior, demographics, and scheduling patterns.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime, date
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db

MODEL_PATH = os.path.join(os.path.dirname(__file__), "noshow_model.pkl")


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def _extract_features(patient_id: int, appt_date: str, appt_time: str, doctor_id: int) -> dict:
    """Extract ML features for a given patient + appointment context."""
    history = db.get_patient_history(patient_id)
    patient = db.get_patient(patient_id)

    total_appts    = len(history)
    no_shows       = sum(1 for h in history if h["no_show"] == 1)
    no_show_rate   = no_shows / total_appts if total_appts > 0 else 0.0
    cancellations  = sum(1 for h in history if h["status"] == "Cancelled")

    # Recency: days since last appointment
    days_since_last = 365
    if history:
        last_date = max(datetime.fromisoformat(h["appt_date"]) for h in history)
        days_since_last = (date.today() - last_date.date()).days

    # Appointment lead time (days until appointment)
    try:
        appt_dt   = datetime.fromisoformat(appt_date)
        lead_days = (appt_dt.date() - date.today()).days
    except Exception:
        lead_days = 0

    # Day of week (0=Mon … 6=Sun)
    try:
        dow = datetime.fromisoformat(appt_date).weekday()
    except Exception:
        dow = 0

    # Hour of day
    try:
        hour = int(appt_time.split(":")[0])
    except Exception:
        hour = 9

    # Patient age
    age = 30
    if patient and patient.get("dob"):
        try:
            birth = datetime.fromisoformat(patient["dob"])
            age   = (date.today() - birth.date()).days // 365
        except Exception:
            pass

    # Gender encoding
    gender_code = 0
    if patient:
        gender_code = 1 if patient.get("gender", "").lower() == "male" else 0

    return {
        "total_past_appts":  total_appts,
        "no_show_rate":      no_show_rate,
        "cancellations":     cancellations,
        "days_since_last":   days_since_last,
        "lead_days":         lead_days,
        "day_of_week":       dow,
        "hour_of_day":       hour,
        "age":               age,
        "gender_code":       gender_code,
    }


# ──────────────────────────────────────────────
# Training Data Builder
# ──────────────────────────────────────────────

def _build_training_data() -> tuple:
    """Build training dataset from appointment_history table."""
    history = db.get_all_history()
    if len(history) < 10:
        return None, None

    records = []
    for h in history:
        pid    = h["patient_id"]
        doctor = h["doctor_id"]
        feat   = _extract_features(pid, h["appt_date"], h["appt_time"], doctor)
        feat["label"] = h["no_show"]
        records.append(feat)

    df = pd.DataFrame(records)
    feature_cols = [
        "total_past_appts", "no_show_rate", "cancellations",
        "days_since_last", "lead_days", "day_of_week",
        "hour_of_day", "age", "gender_code",
    ]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y


# ──────────────────────────────────────────────
# Model Training & Persistence
# ──────────────────────────────────────────────

_pipeline: Pipeline = None


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        )),
    ])


def train_model(force: bool = False) -> str:
    """Train (or retrain) and persist the model. Returns a status message."""
    global _pipeline

    if os.path.exists(MODEL_PATH) and not force:
        _load_model()
        return "Model loaded from disk."

    X, y = _build_training_data()
    if X is None:
        _pipeline = _build_pipeline()
        # Synthetic fallback so predict() always works
        X_syn, y_syn = _make_synthetic_data()
        _pipeline.fit(X_syn, y_syn)
        _save_model()
        return "Trained on synthetic data (not enough real history)."

    _pipeline = _build_pipeline()
    _pipeline.fit(X, y)
    _save_model()
    return f"Model trained on {len(X)} historical records."


def _make_synthetic_data(n=500):
    """Generate synthetic training data that captures intuitive no-show patterns."""
    rng = np.random.RandomState(0)
    X, y = [], []
    for _ in range(n):
        total     = rng.randint(0, 20)
        ns_rate   = rng.beta(1, 4)          # mostly low
        canc      = rng.randint(0, 5)
        since     = rng.randint(0, 400)
        lead      = rng.randint(0, 60)
        dow       = rng.randint(0, 7)
        hour      = rng.randint(8, 18)
        age       = rng.randint(18, 80)
        gender    = rng.randint(0, 2)

        # Rule-based label with noise
        risk = (
            ns_rate * 3.0
            + (lead > 30) * 0.4
            + (dow >= 5)  * 0.3
            + (hour < 9 or hour > 16) * 0.2
            + (total < 2) * 0.3
            + rng.normal(0, 0.1)
        )
        label = 1 if risk > 0.9 else 0
        X.append([total, ns_rate, canc, since, lead, dow, hour, age, gender])
        y.append(label)

    return np.array(X), np.array(y)


def _save_model():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_pipeline, f)


def _load_model():
    global _pipeline
    with open(MODEL_PATH, "rb") as f:
        _pipeline = pickle.load(f)


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict_no_show_risk(patient_id: int, appt_date: str, appt_time: str,
                          doctor_id: int) -> dict:
    """
    Returns:
        risk_score  : float 0–1
        risk_label  : "Low" | "Medium" | "High"
        risk_color  : Streamlit-compatible colour string
        factors     : list[str] – human-readable contributing factors
    """
    global _pipeline
    if _pipeline is None:
        train_model()

    feat    = _extract_features(patient_id, appt_date, appt_time, doctor_id)
    X       = np.array([[
        feat["total_past_appts"],
        feat["no_show_rate"],
        feat["cancellations"],
        feat["days_since_last"],
        feat["lead_days"],
        feat["day_of_week"],
        feat["hour_of_day"],
        feat["age"],
        feat["gender_code"],
    ]])
    proba   = _pipeline.predict_proba(X)[0]
    # proba[1] = probability of no-show
    score   = float(proba[1]) if len(proba) > 1 else 0.0

    if score >= 0.60:
        label, color = "High",   "🔴"
    elif score >= 0.35:
        label, color = "Medium", "🟡"
    else:
        label, color = "Low",    "🟢"

    factors = _explain_risk(feat, score)

    return {
        "risk_score": round(score, 3),
        "risk_label": label,
        "risk_color": color,
        "factors":    factors,
        "features":   feat,
    }


def _explain_risk(feat: dict, score: float) -> list:
    """Generate human-readable explanation of key risk factors."""
    reasons = []
    if feat["no_show_rate"] > 0.4:
        pct = round(feat["no_show_rate"] * 100)
        reasons.append(f"High historical no-show rate ({pct}% of past appointments)")
    if feat["lead_days"] > 30:
        reasons.append(f"Long lead time ({feat['lead_days']} days until appointment)")
    if feat["day_of_week"] >= 5:
        reasons.append("Appointment falls on a weekend")
    if feat["hour_of_day"] < 9 or feat["hour_of_day"] > 16:
        reasons.append("Appointment scheduled outside peak hours")
    if feat["cancellations"] >= 3:
        reasons.append(f"Multiple past cancellations ({feat['cancellations']})")
    if feat["total_past_appts"] < 2:
        reasons.append("New patient — limited appointment history")
    if feat["days_since_last"] > 180:
        reasons.append(f"Inactive patient (last visit {feat['days_since_last']} days ago)")
    if not reasons:
        reasons.append("No significant risk factors identified")
    return reasons


# ──────────────────────────────────────────────
# Batch Risk Update (called after admin triggers)
# ──────────────────────────────────────────────

def update_all_risk_scores():
    """Recalculate and persist risk scores for all scheduled appointments."""
    appts = db.get_all_appointments()
    conn  = db.get_connection()
    for a in appts:
        if a["status"] != "Scheduled":
            continue
        result = predict_no_show_risk(
            a["patient_id"], a["appt_date"], a["appt_time"], a["doctor_id"]
        )
        conn.execute(
            "UPDATE appointments SET no_show_risk=? WHERE id=?",
            (result["risk_score"], a["id"]),
        )
    conn.commit()
    conn.close()


def get_model_performance() -> dict:
    """Evaluate model on available history and return metrics."""
    X, y = _build_training_data()
    if X is None or len(X) < 20:
        return {"available": False}

    global _pipeline
    if _pipeline is None:
        train_model()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    _pipeline.fit(X_tr, y_tr)
    preds  = _pipeline.predict(X_te)
    report = classification_report(y_te, preds, output_dict=True, zero_division=0)

    return {
        "available":      True,
        "accuracy":       round(report["accuracy"] * 100, 1),
        "precision_ns":   round(report.get("1", {}).get("precision", 0) * 100, 1),
        "recall_ns":      round(report.get("1", {}).get("recall", 0) * 100, 1),
        "f1_ns":          round(report.get("1", {}).get("f1-score", 0) * 100, 1),
        "support":        int(report.get("1", {}).get("support", 0)),
        "training_size":  len(X),
    }
