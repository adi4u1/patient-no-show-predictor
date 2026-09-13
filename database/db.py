"""
SQLite database layer for the Healthcare Appointment Management System.
Handles schema creation, seeding, and all CRUD operations.
"""

import sqlite3
import os
import hashlib
from datetime import datetime, date, timedelta
import random

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "healthcare.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    email           TEXT    UNIQUE NOT NULL,
    phone           TEXT,
    dob             TEXT,
    gender          TEXT,
    blood_type      TEXT,
    address         TEXT,
    password_hash   TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS doctors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    specialty       TEXT    NOT NULL,
    email           TEXT    UNIQUE NOT NULL,
    phone           TEXT,
    available_days  TEXT    DEFAULT 'Mon,Tue,Wed,Thu,Fri',
    slot_duration   INTEGER DEFAULT 30
);

CREATE TABLE IF NOT EXISTS appointments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    doctor_id       INTEGER NOT NULL REFERENCES doctors(id),
    appt_date       TEXT    NOT NULL,
    appt_time       TEXT    NOT NULL,
    reason          TEXT,
    status          TEXT    DEFAULT 'Scheduled',
    no_show_risk    REAL    DEFAULT 0.0,
    notes           TEXT,
    created_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(doctor_id, appt_date, appt_time)
);

CREATE TABLE IF NOT EXISTS appointment_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    doctor_id       INTEGER NOT NULL REFERENCES doctors(id),
    appt_date       TEXT    NOT NULL,
    appt_time       TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    no_show         INTEGER DEFAULT 0,
    created_at      TEXT    DEFAULT (datetime('now'))
);
"""


def init_db():
    """Create tables and seed initial data if empty."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        _seed_doctors(conn)
        _seed_demo_patients(conn)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ──────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────

DOCTORS_SEED = [
    ("Dr. Arjun Sharma",    "Cardiology",       "arjun.sharma@clinic.com",   "+91-9000000001"),
    ("Dr. Priya Mehta",     "Dermatology",      "priya.mehta@clinic.com",    "+91-9000000002"),
    ("Dr. Rohan Verma",     "Orthopedics",      "rohan.verma@clinic.com",    "+91-9000000003"),
    ("Dr. Sunita Rao",      "Pediatrics",       "sunita.rao@clinic.com",     "+91-9000000004"),
    ("Dr. Kiran Nair",      "Neurology",        "kiran.nair@clinic.com",     "+91-9000000005"),
    ("Dr. Meena Iyer",      "Gynecology",       "meena.iyer@clinic.com",     "+91-9000000006"),
    ("Dr. Suresh Pillai",   "General Medicine", "suresh.pillai@clinic.com",  "+91-9000000007"),
    ("Dr. Divya Krishnan",  "ENT",              "divya.krishnan@clinic.com", "+91-9000000008"),
]

DEMO_PATIENTS = [
    ("Alice Johnson",  "alice@demo.com",   "+1-555-0101", "1990-04-15", "Female", "A+",  "12 Oak St"),
    ("Bob Williams",   "bob@demo.com",     "+1-555-0102", "1985-08-22", "Male",   "B+",  "45 Pine Ave"),
    ("Carol Davis",    "carol@demo.com",   "+1-555-0103", "1978-11-30", "Female", "O-",  "78 Maple Dr"),
    ("David Martinez", "david@demo.com",   "+1-555-0104", "2000-01-05", "Male",   "AB+", "99 Elm Blvd"),
    ("Eva Brown",      "eva@demo.com",     "+1-555-0105", "1995-06-18", "Female", "A-",  "33 Cedar Ln"),
]


def _seed_doctors(conn: sqlite3.Connection):
    existing = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    if existing > 0:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO doctors (name, specialty, email, phone) VALUES (?,?,?,?)",
        DOCTORS_SEED,
    )


def _seed_demo_patients(conn: sqlite3.Connection):
    existing = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if existing > 0:
        return

    pw_hash = _hash_password("demo123")
    for name, email, phone, dob, gender, blood, addr in DEMO_PATIENTS:
        conn.execute(
            """INSERT OR IGNORE INTO patients
               (name,email,phone,dob,gender,blood_type,address,password_hash)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, email, phone, dob, gender, blood, addr, pw_hash),
        )

    # Seed appointment history for ML training
    random.seed(42)
    doctors = conn.execute("SELECT id FROM doctors").fetchall()
    patients = conn.execute("SELECT id FROM patients").fetchall()
    statuses = ["Completed", "No-Show", "Completed", "Completed", "No-Show", "Completed"]
    for pid in patients:
        for i in range(random.randint(5, 15)):
            past_date = (date.today() - timedelta(days=random.randint(10, 365))).isoformat()
            doctor = random.choice(doctors)
            status = random.choice(statuses)
            no_show = 1 if status == "No-Show" else 0
            conn.execute(
                """INSERT INTO appointment_history
                   (patient_id, doctor_id, appt_date, appt_time, status, no_show)
                   VALUES (?,?,?,?,?,?)""",
                (pid["id"], doctor["id"], past_date, "10:00", status, no_show),
            )


# ──────────────────────────────────────────────
# Patient Operations
# ──────────────────────────────────────────────

def register_patient(name, email, phone, dob, gender, blood_type, address, password) -> dict:
    pw_hash = _hash_password(password)
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO patients
                   (name,email,phone,dob,gender,blood_type,address,password_hash)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (name, email, phone, dob, gender, blood_type, address, pw_hash),
            )
        return {"success": True, "message": "Registration successful!"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Email already registered."}


def login_patient(email: str, password: str):
    pw_hash = _hash_password(password)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE email=? AND password_hash=?",
            (email, pw_hash),
        ).fetchone()
    return dict(row) if row else None


def get_patient(patient_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
    return dict(row) if row else None


def get_all_patients():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY name").fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# Doctor Operations
# ──────────────────────────────────────────────

def get_all_doctors():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM doctors ORDER BY specialty, name").fetchall()
    return [dict(r) for r in rows]


def get_doctor(doctor_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM doctors WHERE id=?", (doctor_id,)).fetchone()
    return dict(row) if row else None


def get_booked_slots(doctor_id: int, appt_date: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT appt_time FROM appointments WHERE doctor_id=? AND appt_date=? AND status!='Cancelled'",
            (doctor_id, appt_date),
        ).fetchall()
    return [r["appt_time"] for r in rows]


# ──────────────────────────────────────────────
# Appointment Operations
# ──────────────────────────────────────────────

def book_appointment(patient_id, doctor_id, appt_date, appt_time, reason, no_show_risk=0.0) -> dict:
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO appointments
                   (patient_id, doctor_id, appt_date, appt_time, reason, no_show_risk)
                   VALUES (?,?,?,?,?,?)""",
                (patient_id, doctor_id, appt_date, appt_time, reason, no_show_risk),
            )
        return {"success": True, "message": "Appointment booked successfully!"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "That slot is already taken. Please choose another."}


def get_patient_appointments(patient_id: int):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT a.*, d.name as doctor_name, d.specialty
               FROM appointments a
               JOIN doctors d ON a.doctor_id = d.id
               WHERE a.patient_id=?
               ORDER BY a.appt_date DESC, a.appt_time DESC""",
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_appointments():
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT a.*, p.name as patient_name, p.email as patient_email,
                      d.name as doctor_name, d.specialty
               FROM appointments a
               JOIN patients p ON a.patient_id = p.id
               JOIN doctors d ON a.doctor_id = d.id
               ORDER BY a.appt_date DESC, a.appt_time DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_appointment_status(appt_id: int, status: str, notes: str = ""):
    with get_connection() as conn:
        conn.execute(
            "UPDATE appointments SET status=?, notes=? WHERE id=?",
            (status, notes, appt_id),
        )
        # Mirror to history
        row = conn.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
        if row:
            no_show = 1 if status == "No-Show" else 0
            conn.execute(
                """INSERT INTO appointment_history
                   (patient_id, doctor_id, appt_date, appt_time, status, no_show)
                   VALUES (?,?,?,?,?,?)""",
                (row["patient_id"], row["doctor_id"], row["appt_date"],
                 row["appt_time"], status, no_show),
            )


def cancel_appointment(appt_id: int):
    update_appointment_status(appt_id, "Cancelled")


def get_patient_history(patient_id: int):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM appointment_history WHERE patient_id=? ORDER BY appt_date DESC",
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_history():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM appointment_history").fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# Dashboard Stats
# ──────────────────────────────────────────────

def get_dashboard_stats() -> dict:
    with get_connection() as conn:
        total_patients   = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        total_appts      = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
        today_appts      = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE appt_date=?",
            (date.today().isoformat(),),
        ).fetchone()[0]
        no_shows         = conn.execute(
            "SELECT COUNT(*) FROM appointment_history WHERE no_show=1",
        ).fetchone()[0]
        total_history    = conn.execute(
            "SELECT COUNT(*) FROM appointment_history",
        ).fetchone()[0]
        high_risk_count  = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE no_show_risk >= 0.6 AND status='Scheduled'",
        ).fetchone()[0]
        by_status = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM appointments GROUP BY status"
        ).fetchall()
        by_specialty = conn.execute(
            """SELECT d.specialty, COUNT(*) as cnt
               FROM appointments a JOIN doctors d ON a.doctor_id=d.id
               GROUP BY d.specialty ORDER BY cnt DESC"""
        ).fetchall()
        daily_trend = conn.execute(
            """SELECT appt_date, COUNT(*) as cnt
               FROM appointments
               WHERE appt_date >= date('now', '-30 days')
               GROUP BY appt_date ORDER BY appt_date"""
        ).fetchall()

    no_show_rate = round((no_shows / total_history * 100), 1) if total_history else 0
    return {
        "total_patients":  total_patients,
        "total_appts":     total_appts,
        "today_appts":     today_appts,
        "no_show_rate":    no_show_rate,
        "high_risk_count": high_risk_count,
        "by_status":       [dict(r) for r in by_status],
        "by_specialty":    [dict(r) for r in by_specialty],
        "daily_trend":     [dict(r) for r in daily_trend],
    }
