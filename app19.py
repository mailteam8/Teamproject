import os
import pickle
import base64
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from supabase import create_client, Client

app = FastAPI()

# قراءة متغيرات البيئة
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

# دالة رفع الملف إلى GitHub
def upload_file_to_github(filepath, repo=GITHUB_REPO, branch=GITHUB_BRANCH):
    filename = os.path.basename(filepath)
    url = f"https://api.github.com/repos/{repo}/contents/models/{filename}"

    with open(filepath, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {
        "message": f"إضافة نموذج {filename}",
        "content": content,
        "branch": branch
    }

    response = requests.put(url, headers=headers, json=data)
    if response.status_code in [200, 201]:
        return {"status": "success", "response": response.json()}
    else:
        return {"status": "error", "response": response.json()}

# ✅ تدريب جميع المرضى
@app.get("/train")
def train_all_patients():
    results = []
    patients_response = supabase.table("tbl_patient").select("pat_id").execute()
    patients = patients_response.data

    for patient in patients:
        pat_id = patient["pat_id"]
        readings_response = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).execute()
        readings = readings_response.data

        data_batch = []
        for r in readings:
            if all(k in r for k in ["oxygen_saturation", "pulse_rate", "temperature"]):
                data_batch.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])

        if data_batch:
            filename = train_patient_model(pat_id, data_batch)
            upload_result = upload_file_to_github(filename)
            results.append({"pat_id": pat_id, "upload_result": upload_result})
        else:
            results.append({"pat_id": pat_id, "status": "⚠️ البيانات غير كافية"})

    return results

# ✅ فحص مريض محدد
@app.get("/check/{pat_id}")
def check_patient(pat_id: str):
    readings_response = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).execute()
    readings = readings_response.data

    if not readings:
        return {"pat_id": pat_id, "status": "⚠️ لا توجد قراءات"}

    data_batch = []
    details = []
    for r in readings:
        if all(k in r for k in ["oxygen_saturation", "pulse_rate", "temperature"]):
            data_batch.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])
            details.append({
                "read_id": r.get("read_id"),
                "created_at": r.get("created_at"),
                "oxygen_saturation": r["oxygen_saturation"],
                "pulse_rate": r["pulse_rate"],
                "temperature": r["temperature"],
                "location": r.get("location")
            })

    if data_batch:
        filename = train_patient_model(pat_id, data_batch)
        upload_result = upload_file_to_github(filename)
        return {
            "pat_id": pat_id,
            "count_readings": len(data_batch),
            "upload_result": upload_result,
            "readings": details
        }
    else:
        return {"pat_id": pat_id, "status": "⚠️ البيانات غير كافية"}

# ✅ التنبؤ باستخدام آخر 10 قراءات
@app.get("/predict/{pat_id}")
def predict_patient(pat_id: str):
    filename = f"heart_guard_{pat_id}.pkl"
    if not os.path.exists(filename):
        return {"pat_id": pat_id, "status": "⚠️ النموذج غير موجود، درّب أولاً عبر /check أو /train"}

    with open(filename, "rb") as f:
        stats = pickle.load(f)

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
    for r in readings:
        values = [r["oxygen_saturation"], r["pulse_rate"], r["temperature"]]
        result = []
        for idx, val in enumerate(values):
            avg = stats[idx]["avg"]
            min_val = stats[idx]["min"]
            max_val = stats[idx]["max"]

            if val < min_val or val > max_val:
                status = "❌ غير طبيعي"
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

    return {
        "pat_id": pat_id,
        "count_readings": len(predictions),
        "predictions": predictions
    }

# ✅ تنزيل النموذج مباشرة من الخدمة
@app.get("/download/{pat_id}")
def download_model(pat_id: str):
    filename = f"heart_guard_{pat_id}.pkl"
    if os.path.exists(filename):
        return FileResponse(
            filename,
            media_type="application/octet-stream",
            filename=filename
        )
    else:
        return {"error": "⚠️ الملف غير موجود، درّب أولاً عبر /check أو /train"}
