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

# ✅ توصيات عامة عند التشخيص الطبيعي (حوالي 30 رسالة)
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
    "✅ الحالة مستقرة، لا يوجد خطر، استمر في نمط حياتك الصحي.",
    "Keep connecting with loved ones and enjoy life.",
    "Stay consistent with your healthy habits and regular checkups.",
    "Balanced nutrition and mental well-being are key to long-term stability.",
    "Take short breaks during work to avoid fatigue.",
    "Practice deep breathing exercises for relaxation.",
    "Stretch regularly to maintain flexibility.",
    "Spend time outdoors for fresh air and sunlight.",
    "Maintain good sleep hygiene for better recovery.",
    "Drink herbal tea to stay calm and hydrated.",
    "Engage in light physical activity daily.",
    "Focus on positive thoughts to reduce stress.",
    "Eat more fruits and vegetables for immunity.",
    "Avoid excessive caffeine for stable energy.",
    "Stay socially connected for mental health.",
    "Listen to calming music to relax your mind.",
    "Keep track of your health indicators regularly.",
    "Practice mindfulness to improve focus and calmness.",
    "Smile often, positivity boosts your health.",
    "Celebrate small wins in your health journey."
]

# ✅ رسائل التنبيه (حوالي 30 رسالة)
alert_messages = {
    "Respiratory risk": "⚠️ انخفاض خطير في نسبة الأكسجين، يجب التدخل الطبي فورًا",
    "Cardiac stress": "⚠️ معدل نبض مرتفع جدًا، خطر على القلب، يجب مراجعة الطبيب فورًا",
    "Sepsis suspected": "⚠️ حرارة مرتفعة مع مؤشرات عدوى، احتمال تسمم دم، يجب الإسعاف فورًا",
    "Hypothermia risk": "⚠️ انخفاض شديد في درجة الحرارة، خطر على الحياة، تدخل عاجل مطلوب",
    "Arrhythmia detected": "⚠️ اضطراب في ضربات القلب، قد يشير إلى خلل خطير، راجع الطبيب فورًا",
    "Hypertension crisis": "⚠️ ضغط دم مرتفع جدًا، خطر نزيف أو جلطة، تدخل عاجل مطلوب",
    "Shock suspected": "⚠️ مؤشرات صدمة جسدية، حالة طارئة تستدعي تدخل طبي فوري",
    "Critical condition": "⚠️ حالة حرجة جدًا، يجب النقل إلى الطوارئ فورًا",
    "Dehydration risk": "⚠️ مؤشرات جفاف، يجب شرب الماء فورًا",
    "Hyperglycemia": "⚠️ ارتفاع شديد في مستوى السكر، راجع الطبيب فورًا",
    "Hypoglycemia": "⚠️ انخفاض خطير في مستوى السكر، تناول طعامًا أو راجع الطبيب فورًا",
    "Heat stroke": "⚠️ حرارة مرتفعة جدًا، خطر ضربة شمس، تدخل عاجل مطلوب",
    "Electrolyte imbalance": "⚠️ خلل في الأملاح المعدنية، راجع الطبيب فورًا",
    "Kidney stress": "⚠️ مؤشرات ضغط على الكلى، راجع الطبيب فورًا",
    "Liver dysfunction": "⚠️ مؤشرات خلل في الكبد، تدخل طبي مطلوب",
    "Anemia suspected": "⚠️ مؤشرات فقر دم، يجب مراجعة الطبيب",
    "Infection risk": "⚠️ مؤشرات عدوى، راجع الطبيب فورًا",
    "Stroke suspected": "⚠️ أعراض جلطة دماغية، تدخل عاجل مطلوب",
    "Normal": "✅ الحالة مستقرة ولا يوجد خطر",
    "Fatigue risk": "⚠️ مؤشرات إرهاق شديد، يجب الراحة فورًا",
    "Immune suppression": "⚠️ ضعف في المناعة، راجع الطبيب",
    "Blood clot risk": "⚠️ مؤشرات تخثر دم، تدخل طبي عاجل",
    "Respiratory infection": "⚠️ عدوى تنفسية محتملة، راجع الطبيب",
    "Cardiac arrest risk": "⚠️ خطر توقف القلب، تدخل عاجل مطلوب",
    "Severe dehydration": "⚠️ جفاف شديد، يجب الإسعاف فورًا",
    "Neurological disorder": "⚠️ خلل عصبي محتمل، راجع الطبيب",
    "Metabolic crisis": "⚠️ خلل في التمثيل الغذائي، تدخل عاجل مطلوب",
    "Blood pressure drop": "⚠️ انخفاض ضغط الدم بشكل خطير، تدخل عاجل مطلوب",
    "Respiratory failure": "⚠️ فشل تنفسي محتمل، تدخل عاجل مطلوب"
}

# ✅ توصيات التقارير (حوالي 30 توصية)
report_recommendations = {
    "Respiratory risk": "Your oxygen level is critically low. Seek immediate medical attention.",
    "Cardiac stress": "Your heart rate is dangerously high. Rest immediately and consult a cardiologist.",
    "Sepsis suspected": "High temperature and infection markers detected. Immediate hospital care is strongly advised.",
    "Hypothermia risk": "Body temperature is dangerously low. Warm up immediately and seek urgent medical help.",
    "Arrhythmia detected": "Irregular heartbeat detected. Please consult a cardiologist as soon as possible.",
    "Hypertension crisis": "Blood pressure is critically elevated. Emergency intervention is necessary to reduce risks.",
    "Shock suspected": "Indicators of shock detected. Emergency medical intervention is required.",
    "Critical condition": "Overall condition is critical. Immediate transfer to emergency care is necessary.",
    "Dehydration risk": "Signs of dehydration detected. Increase fluid intake immediately.",
    "Hyperglycemia": "High blood sugar detected. Consult your doctor for urgent management.",
    "Hypoglycemia": "Low blood sugar detected. Eat something sweet and consult your doctor.",
    "Heat stroke": "Dangerously high body temperature detected. Seek emergency care immediately.",
    "Electrolyte imbalance": "Signs of electrolyte imbalance detected. Medical checkup is required.",
    "Kidney stress": "Indicators of kidney stress detected. Consult a nephrologist.",
    "Liver dysfunction": "Signs of liver dysfunction detected. Seek medical evaluation.",
    "Anemia suspected": "Low hemoglobin indicators detected. Consult your doctor.",
    "Infection risk": "Signs of infection detected. Medical care is advised.",
    "Stroke suspected": "Possible stroke symptoms detected. Emergency care is required.",
    "Normal": "Your health indicators are stable. Keep following healthy habits.",
    "Fatigue risk": "Signs of severe fatigue detected. Rest and consult your doctor.",
    "Immune suppression": "Weak immune indicators detected. Seek medical advice.",
    "Blood clot risk": "Signs of blood clot detected. Emergency care is required.",
    "Respiratory infection": "Possible respiratory infection detected. Consult your doctor.",
    "Cardiac arrest risk": "Dangerous heart indicators detected. Emergency care is required.",
    "Severe dehydration": "Severe dehydration detected. Immediate hospital care is required.",
    "Neurological disorder": "Signs of neurological disorder detected. Seek medical evaluation.",
    "Metabolic crisis": "Metabolic imbalance detected. Emergency care is required.",
    "Blood pressure drop": "Dangerously low blood pressure detected. Emergency care is required.",
    "Respiratory failure": "Respiratory failure suspected. Immediate hospital care is required."
}
# ✅ دالة التشخيص التلقائي تغطي حوالي 30 حالة
def auto_diagnose(r):
    if r["oxygen_saturation"] < 90:
        return "Respiratory risk"
    elif r["pulse_rate"] > 120:
        return "Cardiac stress"
    elif r["pulse_rate"] < 40:
        return "Arrhythmia detected"
    elif r["temperature"] > 39:
        return "Sepsis suspected"
    elif r["temperature"] < 35:
        return "Hypothermia risk"
    elif r["pulse_rate"] > 140 and r["oxygen_saturation"] < 92:
        return "Critical condition"
    elif r["pulse_rate"] > 160:
        return "Shock suspected"
    elif r["pulse_rate"] > 140 and r["temperature"] > 40:
        return "Hypertension crisis"
    elif r["temperature"] > 41:
        return "Heat stroke"
    elif r["oxygen_saturation"] < 85:
        return "Respiratory failure"
    elif r["pulse_rate"] < 35 and r["oxygen_saturation"] < 88:
        return "Cardiac arrest risk"
    elif r["temperature"] < 34:
        return "Severe hypothermia"
    elif r["temperature"] > 42:
        return "Metabolic crisis"
    elif r["pulse_rate"] > 150 and r["oxygen_saturation"] < 90:
        return "Stroke suspected"
    elif r["pulse_rate"] > 130 and r["temperature"] > 38.5:
        return "Infection risk"
    elif r["pulse_rate"] < 50 and r["temperature"] < 36:
        return "Fatigue risk"
    elif r["oxygen_saturation"] < 88 and r["temperature"] > 38:
        return "Respiratory infection"
    elif r["pulse_rate"] > 135 and r["oxygen_saturation"] < 89:
        return "Blood clot risk"
    elif r["pulse_rate"] > 125 and r["temperature"] > 39.5:
        return "Immune suppression"
    elif r["pulse_rate"] < 45 and r["oxygen_saturation"] < 90:
        return "Neurological disorder"
    elif r["pulse_rate"] > 145 and r["temperature"] > 40.5:
        return "Kidney stress"
    elif r["pulse_rate"] > 120 and r["temperature"] > 39.8:
        return "Liver dysfunction"
    elif r["pulse_rate"] < 55 and r["oxygen_saturation"] < 91:
        return "Anemia suspected"
    elif r["pulse_rate"] > 110 and r["temperature"] > 38.7:
        return "Electrolyte imbalance"
    elif r["pulse_rate"] < 60 and r["temperature"] < 36.5:
        return "Dehydration risk"
    elif r["pulse_rate"] > 140 and r["temperature"] > 39.9:
        return "Hyperglycemia"
    elif r["pulse_rate"] < 50 and r["temperature"] < 36.2:
        return "Hypoglycemia"
    elif r["pulse_rate"] > 155 and r["oxygen_saturation"] < 87:
        return "Blood pressure drop"
    else:
        return "Normal"

# ✅ تدريب نموذج Random Forest
def train_random_forest(patient_id, readings):
    X, y = [], []
    for r in readings:
        features = [
            r["oxygen_saturation"],
            r["pulse_rate"],
            r["temperature"]
        ]
        X.append(features)
        y.append(auto_diagnose(r))  # ✅ التشخيص التلقائي

    if not X or not y:
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    filename = f"heart_guard_rf_{patient_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return filename

# ✅ التنبؤ باستخدام Random Forest
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
            r["temperature"]
        ]
        diagnosis = model.predict([features])[0]
        predictions.append({
            "read_id": r.get("read_id"),
            "created_at": r.get("created_at"),
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

        # ✅ إنشاء التنبيه في جميع الحالات
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
            "alert": alert_response.data
        }

    except Exception as e:
        return {"pat_id": pat_id, "status": "❌ خطأ أثناء التنبؤ", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app25:app", host="0.0.0.0", port=port, reload=True)
