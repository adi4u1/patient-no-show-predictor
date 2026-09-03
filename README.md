# 🏥 MediCare — AI-Powered Healthcare Appointment System

A fully deployable **Streamlit** application for managing patient appointments,
featuring an **ML-based no-show risk predictor** and a **Gemini 2.5 Flash AI assistant**.

---

## ✨ Features

| Module | Description |
|---|---|
| 📅 **Patient Booking Portal** | Register / login, filter by specialty, pick doctor + date + slot, instant ML risk preview |
| 🛡️ **Admin Dashboard** | Overview analytics, appointment management, patient records, risk heat-map |
| 🤖 **No-Show Risk Predictor** | Gradient Boosting ML model — 9 engineered features, 3 risk tiers (Low / Medium / High) |
| 💬 **MediBot AI Assistant** | Gemini 2.5 Flash chat for patients & admins with live session context injection |
| 📊 **Analytics Charts** | Plotly pie, bar, histogram, area charts — all rendered natively in Streamlit |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
cd healthcare_app
pip install -r requirements.txt
```

### 2. Configure your Gemini API key

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your key
# GEMINI_API_KEY=AIza...
```

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

> **Alternatively**, you can paste the key directly in the AI Assistant page — no `.env` needed.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🔐 Demo Credentials

| Role | Credential |
|---|---|
| **Patient (demo)** | Email: `alice@demo.com`  Password: `demo123` |
| **Admin** | Password: `admin123` |

Five demo patients are auto-seeded on first launch with synthetic appointment history
so the ML model trains immediately.

---

## 🧠 ML No-Show Predictor

The **GradientBoostingClassifier** (`sklearn`) is trained on appointment history with 9 features:

1. Historical no-show rate
2. Lead time (days until appointment)
3. Total past appointments
4. Past cancellations
5. Day of week
6. Hour of day
7. Patient age
8. Days since last visit
9. Gender code

**Risk tiers:**
- 🟢 **Low** < 35% — standard scheduling
- 🟡 **Medium** 35–59% — send email/SMS reminder
- 🔴 **High** ≥ 60% — phone call + overbooking buffer

The model auto-retrains on synthetic data if real history is insufficient,
and can be force-retrained anytime from **Admin Dashboard → ML Model**.

---

## 🗂️ Project Structure

```
healthcare_app/
├── app.py                    ← Streamlit entry point + home page
├── requirements.txt
├── .env.example
├── database/
│   └── db.py                 ← SQLite schema, CRUD, seeding, dashboard stats
├── ml/
│   └── predictor.py          ← ML model training, prediction, feature engineering
├── modules/
│   ├── booking.py            ← Patient portal (auth + booking + appointment management)
│   ├── admin.py              ← Admin dashboard (overview, appointments, patients, risk, ML)
│   └── ai_assistant.py       ← Gemini 2.5 Flash chat assistant (MediBot)
└── utils/
    └── helpers.py            ← Shared utilities (time slots, date helpers, CSS)
```

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push the `healthcare_app/` folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Set **Main file path** to `app.py`.
4. Under **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ADMIN_PASSWORD = "your_password"
   ```
5. Deploy — the SQLite database is created automatically on first launch.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (pure Python — no HTML/CSS/JS written by developer)
- **Database:** SQLite (via `sqlite3` stdlib)
- **ML:** `scikit-learn` — GradientBoostingClassifier
- **Charts:** `plotly`
- **AI:** Google Gemini 2.5 Flash (`google-generativeai`)
- **Data:** `pandas`, `numpy`

---

*Made with IBM Bob*
