import os
import pickle
import base64
import requests
from datetime import datetime
from fastapi import FastAPI
from supabase import create_client, Client
import uvicorn

app = FastAPI()

# إعدادات البيئة
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# دالة تدريب النموذج
def train_patient_model(patient_id, data_batch):
    stats = []
    for col in zip(*data_batch):
        avg = sum(col)/len(col)
        min_val = min(col)
        max_val = max(col)
        stats.append({"avg": avg, "min": min_val, "max": max_val})

    filename = f"heart_guard_{patient_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(stats, f)
    return filename

# رفع الملف إلى GitHub
def upload_file_to_github(filepath, repo=GITHUB_REPO, branch=GITHUB_BRANCH):
    try:
        filename = os.path.basename(filepath)
        url = f"https://api.github.com/repos/{repo}/contents/models/{filename}"

        with open(filepath, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json"
        }

        get_response = requests.get(url, headers=headers, params={"ref": branch})
        sha = None
        if get_response.status_code == 200:
            sha = get_response.json()["sha"]

        data = {
            "message": f"إضافة/تحديث نموذج {filename}",
            "content": content,
            "branch": branch
        }
        if sha:
            data["sha"] = sha

        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        return {"status": "success", "response": response.json()}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# دالة إنشاء النموذج العام مرة واحدة فقط
def create_general_model():
    try:
        filename = "heart_guard_general.pkl"

        if os.path.exists(filename):
            return {"status": "exists", "filename": filename}

        general_data = [
            [95, 75, 36.8],
            [96, 80, 37.0],
            [97, 78, 36.9],
            [98, 76, 36.7],
            [97, 79, 37.1]
        ]

        filename = train_patient_model("general", general_data)
        upload_result = upload_file_to_github(filename)

        return {"status": "created", "filename": filename, "upload_result": upload_result}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/predict/{pat_id}")
def predict_patient(pat_id: str):
    try:
        filename = f"heart_guard_{pat_id}.pkl"

        # ✅ إذا الملف غير موجود → حاول إنشاؤه
        if not os.path.exists(filename):
            readings_response = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).execute()
            readings = readings_response.data

            data_batch = []
            if readings:
                for r in readings:
                    if all(k in r for k in ["oxygen_saturation", "pulse_rate", "temperature"]):
                        data_batch.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])

            if data_batch:
                filename = train_patient_model(pat_id, data_batch)
                upload_file_to_github(filename)
            else:
                general_result = create_general_model()
                filename = general_result.get("filename", "heart_guard_general.pkl")

        # ✅ بعد التأكد أن هناك ملف PKL (خاص أو عام)، نكمل التنبؤ
        with open(filename, "rb") as f:
            stats = pickle.load(f)

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

        predictions = []
        abnormal_count = 0
        oxygen_abnormal = 0
        pulse_abnormal = 0
        temp_abnormal = 0

        for r in readings:
            values = [r["oxygen_saturation"], r["pulse_rate"], r["temperature"]]
            result = []
            for idx, val in enumerate(values):
                avg = stats[idx]["avg"]
                min_val = stats[idx]["min"]
                max_val = stats[idx]["max"]

                if val < min_val or val > max_val:
                    status = "❌ غير طبيعي"
                    abnormal_count += 1
                    if idx == 0: oxygen_abnormal += 1
                    if idx == 1: pulse_abnormal += 1
                    if idx == 2: temp_abnormal += 1
                else:
                    status = "✅ طبيعي"

                result.append({
                    "value": val,
                    "avg": avg,
                    "min": min_val,
                    "max": max_val,
                    "status": status
                })

            predictions.append({
                "read_id": r.get("read_id"),
                "created_at": r.get("created_at"),
                "oxygen_saturation": r["oxygen_saturation"],
                "pulse_rate": r["pulse_rate"],
                "temperature": r["temperature"],
                "prediction": result
            })

        # التشخيص والتوصية (كل الاحتمالات)
        if abnormal_count == 0:
            diagnosis = "Normal"
            recommendation = "Take care of your health and don't overexert yourself"
        elif oxygen_abnormal > 3:
            diagnosis = "Respiratory risk"
            recommendation = "انخفاض الأكسجين بشكل متكرر، يجب مراجعة طبيب صدرية فورًا."
        elif pulse_abnormal > 3 and max([r["pulse_rate"] for r in readings]) > 140:
            diagnosis = "Unstable angina"
            recommendation = "معدل نبض مرتفع جدًا مع أعراض قلبية، يجب مراجعة الطوارئ فورًا."
        elif pulse_abnormal > 3:
            diagnosis = "Cardiac stress"
            recommendation = "معدل نبض غير طبيعي بشكل متكرر، يُفضل مراجعة طبيب قلب."
        elif temp_abnormal > 3 and max([r["temperature"] for r in readings]) > 39.5:
            diagnosis = "Sepsis suspected"
            recommendation = "ارتفاع حرارة شديد جدًا مع احتمال عدوى خطيرة، يجب مراجعة المستشفى."
        elif temp_abnormal > 3 and min([r["temperature"] for r in readings]) < 35.0:
            diagnosis = "Hypothermia risk"
            recommendation = "انخفاض حرارة الجسم بشكل خطير، يجب تدفئة المريض فورًا ومراجعة الطوارئ."
        elif temp_abnormal > 3:
            diagnosis = "Fever detected"
            recommendation = "ارتفاع الحرارة بشكل متكرر، احتمال وجود عدوى أو التهاب."
        elif max([r["pulse_rate"] for r in readings]) > 180:
            diagnosis = "Hypertension crisis"
            recommendation = "ضغط دم أو نبض مرتفع جدًا، يجب مراجعة الطوارئ فورًا."
        else:
            diagnosis = "Minor irregularities"
            recommendation = "بعض القراءات غير طبيعية، يُفضل متابعة الحالة وتجنب الإجهاد."

        # ✅ التقرير دائمًا
        report_data = {
            "rep_date": datetime.utcnow().isoformat(),
            "rep_diagnosis": diagnosis,
            "rep_recommendation": recommendation,
            "pat_id": pat_id
        }
        report_response = supabase.table("tbl_report").insert(report_data).execute()

        # ✅ التنبيه فقط عند الحالات الخطيرة
        alert_response = None
        if diagnosis in ["Respiratory risk", "Cardiac stress", "Critical condition", 
                         "Unstable angina", "Hypertension crisis", "Sepsis suspected", "Hypothermia risk"]:
            alert_data = {
                "alert_type": diagnosis,
                "alert_message": recommendation,
                "alert_timestamp": datetime.utcnow().isoformat(),
                "is_seen": False,
                "pat_id": pat_id
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
    uvicorn.run("app20:app", host="0.0.0.0", port=port, reload=True)
