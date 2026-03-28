import os
import pickle
import random
from datetime import datetime
from fastapi import FastAPI
from supabase import create_client, Client
import uvicorn
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# إعدادات البيئة
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# قائمة رسائل عامة للتشخيص الطبيعي
general_recommendations = [
    "Your vital signs are within normal ranges. Keep maintaining a healthy lifestyle.",
    "Everything looks stable. Stay active and eat balanced meals.",
    "Your health indicators are fine. Remember to rest well and stay hydrated.",
    "Normal readings detected. Keep up the good work with your daily routine.",
    "Your vitals are stable. Continue with regular exercise and healthy habits.",
    "Your body is showing healthy signs. Keep smiling and stay positive.",
    "Stable readings today. A short walk or light activity will keep you energized.",
    "Your health looks good. Make sure to drink enough water throughout the day.",
    "Normal results. Keep a balanced diet and avoid unnecessary stress.",
    "Your vital signs are fine. A good night’s sleep will help maintain this stability.",
    "Everything is within safe limits. Keep enjoying your daily activities.",
    "Your health indicators are normal. Don’t forget to take breaks and relax.",
    "Stable condition detected. Keep following your doctor’s advice and healthy routines.",
    "Your vitals are good. A little stretching or breathing exercise can boost your energy.",
    "Normal readings. Stay consistent with your healthy habits.",
    "Your health looks stable. Keep connecting with loved ones and enjoy life.",
    "Your body is balanced. Maintain your routine and avoid overexertion.",
    "Normal signs detected. Keep focusing on good nutrition and mental well-being.",
    "Your vitals are fine. A calm mind and active body keep you strong.",
    "Stable readings. Keep up your healthy lifestyle and regular checkups."
]

# رسائل التنبيه حسب نوع التشخيص
alert_messages = {
    "Respiratory risk": "⚠️ انخفاض خطير في نسبة الأكسجين، يجب التدخل الطبي فورًا",
    "Cardiac stress": "⚠️ معدل نبض مرتفع جدًا، خطر على القلب، يجب مراجعة الطبيب فورًا",
    "Unstable angina": "⚠️ أعراض ذبحة صدرية غير مستقرة، حالة حرجة تستدعي تدخل عاجل",
    "Hypertension crisis": "⚠️ ضغط دم مرتفع جدًا، خطر نزيف أو جلطة، تدخل عاجل مطلوب",
    "Sepsis suspected": "⚠️ حرارة مرتفعة مع مؤشرات عدوى، احتمال تسمم دم، يجب الإسعاف فورًا",
    "Hypothermia risk": "⚠️ انخفاض شديد في درجة الحرارة، خطر على الحياة، تدخل عاجل مطلوب",
    "Arrhythmia detected": "⚠️ اضطراب في ضربات القلب، قد يشير إلى خلل خطير، راجع الطبيب فورًا",
    "Shock suspected": "⚠️ مؤشرات صدمة جسدية، حالة طارئة تستدعي تدخل طبي فوري",
    "Critical condition": "⚠️ حالة حرجة جدًا، يجب النقل إلى الطوارئ فورًا"
}

# توصيات التقارير حسب نوع التشخيص
report_recommendations = {
    "Respiratory risk": "Your oxygen level is critically low. Seek immediate medical attention and avoid exertion.",
    "Cardiac stress": "Your heart rate is dangerously high. Rest immediately and consult a cardiologist.",
    "Unstable angina": "Signs of unstable angina detected. Emergency care is required to prevent complications.",
    "Hypertension crisis": "Blood pressure is critically elevated. Emergency intervention is necessary to reduce risks.",
    "Sepsis suspected": "High temperature and infection markers detected. Immediate hospital care is strongly advised.",
    "Hypothermia risk": "Body temperature is dangerously low. Warm up immediately and seek urgent medical help.",
    "Arrhythmia detected": "Irregular heartbeat detected. Please consult a cardiologist as soon as possible.",
    "Shock suspected": "Indicators of shock detected. Emergency medical intervention is required.",
    "Critical condition": "Overall condition is critical. Immediate transfer to emergency care is necessary."
}

# تدريب نموذج Random Forest
def train_random_forest(patient_id, readings):
    X, y = [], []
    for r in readings:
        features = [
            r["oxygen_saturation"],
            r["pulse_rate"],
            r["temperature"]
        ]
        X.append(features)
        y.append(r.get("diagnosis", "Normal"))

    if not X or not y:
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    filename = f"heart_guard_rf_{patient_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return filename

# إنشاء نموذج عام
def create_general_model():
    general_data = [
        {"oxygen_saturation": 97, "pulse_rate": 78, "temperature": 37.0, "diagnosis": "Normal"},
        {"oxygen_saturation": 95, "pulse_rate": 75, "temperature": 36.8, "diagnosis": "Normal"},
        {"oxygen_saturation": 96, "pulse_rate": 80, "temperature": 37.1, "diagnosis": "Normal"}
    ]
    filename = "heart_guard_general.pkl"
    train_random_forest("general", general_data)
    return filename

# التنبؤ باستخدام Random Forest
def predict_with_rf(patient_id, readings):
    filename = f"heart_guard_rf_{patient_id}.pkl"
    if not os.path.exists(filename):
        filename = train_random_forest(patient_id, readings)

    if not filename or not os.path.exists(filename):
        filename = "heart_guard_general.pkl"
        if not os.path.exists(filename):
            filename = create_general_model()

    with open(filename, "rb") as f:
        model = pickle.load(f)

    predictions = []
    for r in readings:
        features = [
            r["oxygen_saturation"],
            r["pulse_rate"],
            r["temperature"]
        ]
        diagnosis = model.predict([features])[0]
        predictions.append({
            "read_id": r.get("read_id"),
            "created_at": r.get("created_at"),
            "dev_id": r.get("dev_id"),
            "location": r.get("location"),
            "pat_id": r.get("pat_id"),
            "diagnosis": diagnosis
        })
    return predictions

@app.get("/predict/{pat_id}")
def predict_patient(pat_id: str):
    try:
        # ✅ جلب آخر قراءة فقط
        last_reading_response = supabase.table("tbl_reading")\
            .select("*")\
            .eq("pat_id", pat_id)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        last_reading = last_reading_response.data

        if not last_reading:
            return {"pat_id": pat_id, "status": "⚠️ لا توجد قراءات"}

        # ✅ جلب كل القراءات لتحديث PKL
        all_readings_response = supabase.table("tbl_reading")\
            .select("*")\
            .eq("pat_id", pat_id)\
            .execute()
        all_readings = all_readings_response.data

        if all_readings:
            train_random_forest(pat_id, all_readings)

        # ✅ التنبؤ باستخدام آخر قراءة فقط
        predictions = predict_with_rf(pat_id, last_reading)

        if not predictions:
            return {"pat_id": pat_id, "status": "⚠️ لا توجد بيانات كافية لتدريب النموذج"}

        diagnosis = predictions[-1]["diagnosis"]

        # ✅ توصية التقرير
        if diagnosis == "Normal":
            recommendation = random.choice(general_recommendations)
        else:
            recommendation = report_recommendations.get(diagnosis, "⚠️ حالة حرجة، يجب التدخل الطبي فورًا")

        # ✅ إنشاء التقرير
        report_data = {
            "rep_date": datetime.utcnow().isoformat(),
            "rep_diagnosis": diagnosis,
            "rep_recommendation": recommendation,
            "pat_id": pat_id
        }
        report_response = supabase.table("tbl_report").insert(report_data).execute()

        # ✅ إنشاء التنبيه إذا الحالة خطيرة
        alert_response = None
        if diagnosis != "Normal":
            alert_message = alert_messages.get(diagnosis, "⚠️ حالة حرجة، يجب التدخل الطبي فورًا")
            alert_data = {
                "alert_type": diagnosis,
                "alert_message": alert_message,
                "alert_timestamp": datetime.utcnow().isoformat(),
                "is_seen": False,
                "pat_id": pat_id
            }
            alert_response = supabase.table("tbl_alert").insert(alert_data).execute()

        return {
            "pat_id": pat_id,
            "report": report_response.data,
            "alert": alert_response.data if alert_response else None
        }

    except Exception as e:
        return {"pat_id": pat_id, "status": "❌ خطأ أثناء التنبؤ", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app25:app", host="0.0.0.0", port=port, reload=True)  
