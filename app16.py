from fastapi import FastAPI
import os, pandas as pd, joblib
from supabase import create_client
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# تحميل متغيرات البيئة
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🟢 تدريب النموذج
@app.get("/train/model")
def train_model():
    readings = supabase.table("tbl_reading").select("*").execute().data
    df = pd.DataFrame(readings)
    if df.empty:
        return {"error": "لا توجد بيانات للتدريب"}

    df["is_emergency"] = ((df["temperature"] > 38) | (df["oxygen_saturation"] < 90)).astype(int)
    X = df[["temperature", "oxygen_saturation", "pulse_rate"]]
    y = df["is_emergency"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "models/health_model.pkl")
    return {"message": "تم تدريب النموذج وحفظه بنجاح", "accuracy": model.score(X, y)}

# 🟢 توليد تقرير (بدون rep_id)
def generate_report(pat_id, diagnosis, recommendation):
    return {
        "pat_id": pat_id,
        "rep_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }

# 🟢 توليد تنبيه (بدون alert_id)
def generate_alert(pat_id, emergency_flag, prob):
    if emergency_flag == 1:
        if prob > 0.8:
            alert_type = "Critical Condition"
            alert_message = "الحالة حرجة جدًا، يجب مراجعة الطبيب فورًا."
        else:
            alert_type = "Moderate Risk"
            alert_message = "هناك احتمال خطر، يفضل مراجعة الطبيب."
        is_seen = False
    else:
        alert_type = "Healthy"
        alert_message = "الوضع مستقر، لا ترهق نفسك ولا تهمل أدويتك."
        is_seen = True

    return {
        "pat_id": pat_id,
        "alert_type": alert_type,
        "alert_message": alert_message,
        "alert_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "is_seen": is_seen
    }

# 🟢 حفظ مع إعادة المحاولة
def save_with_retry(table, data, retries=3):
    error_msg = None
    for attempt in range(retries):
        try:
            response = supabase.table(table).insert(data).execute()
            if response.data:
                return {"status": "success", "message": f"تم حفظ {table} بنجاح"}
        except Exception as e:
            error_msg = str(e)
    return {"status": "error", "message": f"فشل حفظ {table} بعد عدة محاولات", "details": error_msg, "json": data}

# 🟢 التنبؤ باستخدام النموذج
def predict_ai(reading):
    model = joblib.load("models/health_model.pkl")
    X = [[reading["temperature"], reading["oxygen_saturation"], reading["pulse_rate"]]]
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return prediction, prob

# 🟢 توليد تقرير مجمع
def generate_summary_report(report, alert, prob):
    if report["rep_diagnosis"] == "خطر عالي" and "Critical" in alert["alert_type"]:
        overall_status = "خطر عالي"
        risk_level = 3
    elif "Moderate" in alert["alert_type"] or (0.4 <= prob <= 0.7):
        overall_status = "خطر متوسط"
        risk_level = 2
    else:
        overall_status = "الوضع مستقر"
        risk_level = 1

    return {
        "overall_status": overall_status,
        "risk_level": risk_level,
        "diagnosis": report["rep_diagnosis"],
        "recommendation": report["rep_recommendation"],
        "alert_type": alert["alert_type"],
        "alert_message": alert["alert_message"],
        "probability": prob,
        "signature": "مع تحيات تطبيق أمان قلب"
    }

# 🟢 التنبؤ بحسب رقم القراءة
@app.get("/predict/reading/{read_id}")
def predict_by_reading(read_id: int):
    reading = supabase.table("tbl_reading").select("*").eq("read_id", read_id).execute()
    if not reading.data:
        return {"error": f"لا توجد قراءة بالرقم {read_id}"}
    selected_reading = reading.data[0]
    pat_id = selected_reading.get("pat_id", 0)

    prediction, prob = predict_ai(selected_reading)
    diagnosis = "خطر عالي" if prediction == 1 else "الوضع مستقر"
    recommendation = "يجب مراجعة الطبيب فورًا" if prediction == 1 else "استمر في المتابعة الدورية فقط"

    report = generate_report(pat_id, diagnosis, recommendation)
    alert = generate_alert(pat_id, prediction, prob)

    report_status = save_with_retry("tbl_report", report)
    alert_status = save_with_retry("tbl_alert", alert)

    summary = generate_summary_report(report, alert, prob)

    return {
        "read_id": read_id,
        "report": report,
        "report_status": report_status,
        "alert": alert,
        "alert_status": alert_status,
        "summary": summary
    }

# 🟢 التنبؤ بحسب المريض (آخر قراءة له)
@app.get("/predict/patient/{pat_id}")
def predict_by_patient(pat_id: int):
    reading = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).order("created_at", desc=True).limit(1).execute()
    if not reading.data:
        return {"error": f"لا توجد قراءات للمريض {pat_id}"}
    selected_reading = reading.data[0]

    prediction, prob = predict_ai(selected_reading)
    diagnosis = "خطر عالي" if prediction == 1 else "الوضع مستقر"
    recommendation = "يجب مراجعة الطبيب فورًا" if prediction == 1 else "استمر في المتابعة الدورية فقط"

    report = generate_report(pat_id, diagnosis, recommendation)
    alert = generate_alert(pat_id, prediction, prob)

    report_status = save_with_retry("tbl_report", report)
    alert_status = save_with_retry("tbl_alert", alert)

    summary = generate_summary_report(report, alert, prob)

    return {
        "pat_id": pat_id,
        "report": report,
        "report_status": report_status,
        "alert": alert,
        "alert_status": alert_status,
        "summary": summary
    }

# 🟢 إرجاع آخر تنبيهين
@app.get("/alerts/latest")
def get_latest_alerts():
    response = supabase.table("tbl_alert").select("*").order("alert_timestamp", desc=True).limit(2).execute()
    if response.data:
        return {"status": "success", "latest_alerts": response.data}
    else:
        return {"status": "error", "message": "لا توجد تنبيهات"}
