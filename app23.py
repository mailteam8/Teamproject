import os
import pickle
import base64
import requests
from datetime import datetime
from fastapi import FastAPI
from supabase import create_client, Client
import uvicorn
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# إعدادات البيئة
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# تدريب نموذج Random Forest
def train_random_forest(patient_id, readings):
    X = []
    y = []
    for r in readings:
        features = [
            r["oxygen_saturation"],
            r["pulse_rate"],
            r["temperature"],
            r.get("activity_level", 0),
            int(r.get("fall_detected", False)),
            int(r.get("sos_triggered", False))
        ]
        X.append(features)
        y.append(r.get("diagnosis", "Normal"))  # التشخيص السابق كـ label

    if not X or not y:
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    filename = f"heart_guard_rf_{patient_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return filename

# التنبؤ باستخدام Random Forest
def predict_with_rf(patient_id, readings):
    filename = f"heart_guard_rf_{patient_id}.pkl"
    if not os.path.exists(filename):
        filename = train_random_forest(patient_id, readings)

    if not filename or not os.path.exists(filename):
        return []

    with open(filename, "rb") as f:
        model = pickle.load(f)

    predictions = []
    for r in readings:
        features = [
            r["oxygen_saturation"],
            r["pulse_rate"],
            r["temperature"],
            r.get("activity_level", 0),
            int(r.get("fall_detected", False)),
            int(r.get("sos_triggered", False))
        ]
        diagnosis = model.predict([features])[0]
        predictions.append({
            "read_id": r.get("read_id"),
            "created_at": r.get("created_at"),
            "features": features,
            "diagnosis": diagnosis,
            "geo_location": r.get("geo_location")
        })
    return predictions

@app.get("/predict/{pat_id}")
def predict_patient(pat_id: str):
    try:
        # جلب آخر 10 قراءات
        readings_response = supabase.table("tbl_reading")\
            .select("*")\
            .eq("pat_id", pat_id)\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        readings = readings_response.data

        if not readings:
            return {"pat_id": pat_id, "status": "⚠️ لا توجد قراءات"}

        # التنبؤ باستخدام Random Forest
        predictions = predict_with_rf(pat_id, readings)

        if not predictions:
            return {"pat_id": pat_id, "status": "⚠️ لا توجد بيانات كافية لتدريب النموذج"}

        # أخذ آخر تشخيص
        diagnosis = predictions[-1]["diagnosis"]

        # توصية عامة (يمكنك تخصيصها أكثر)
        recommendation = "متابعة الحالة بشكل مستمر"
        if diagnosis in ["Respiratory risk", "Cardiac stress", "Unstable angina",
                         "Hypertension crisis", "Sepsis suspected", "Hypothermia risk",
                         "Fall detected", "SOS triggered", "Overexertion risk"]:
            recommendation = "⚠️ حالة حرجة، يجب التدخل الطبي فورًا"

        # ✅ التقرير دائمًا
        report_data = {
            "rep_date": datetime.utcnow().isoformat(),
            "rep_diagnosis": diagnosis,
            "rep_recommendation": recommendation,
            "pat_id": pat_id,
            "geo_location": predictions[-1].get("geo_location")
        }
        report_response = supabase.table("tbl_report").insert(report_data).execute()

        # ✅ التنبيه عند الحالات الخطيرة
        alert_response = None
        if diagnosis in ["Respiratory risk", "Cardiac stress", "Unstable angina",
                         "Hypertension crisis", "Sepsis suspected", "Hypothermia risk",
                         "Fall detected", "SOS triggered", "Overexertion risk"]:
            alert_data = {
                "alert_type": diagnosis,
                "alert_message": recommendation,
                "alert_timestamp": datetime.utcnow().isoformat(),
                "is_seen": False,
                "pat_id": pat_id,
                "geo_location": predictions[-1].get("geo_location")
            }
            alert_response = supabase.table("tb_alert").insert(alert_data).execute()

        return {
            "pat_id": pat_id,
            "count_readings": len(predictions),
            "predictions": predictions,
            "report": report_response.data,
            "alert": alert_response.data if alert_response else None
        }

    except Exception as e:
        return {"pat_id": pat_id, "status": "❌ خطأ أثناء التنبؤ", "message": str(e)}

# ✅ تشغيل التطبيق محليًا وعلى Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app23:app", host="0.0.0.0", port=port, reload=True)
