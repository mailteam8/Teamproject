from fastapi import FastAPI
import os
import pickle
import base64
import requests
from supabase import create_client, Client

app = FastAPI()

# الاتصال بـ Supabase عبر متغيرات البيئة
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# إعدادات GitHub عبر متغيرات البيئة
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPO")            # مثال: "username/repo"
BRANCH = os.environ.get("GITHUB_BRANCH", "main")

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

def upload_file_to_github(filepath, repo=REPO, branch=BRANCH):
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
    return response.status_code in [200, 201]

@app.get("/train")
def train_all_patients():
    results = []

    # قراءة المرضى من جدول tbl_patient
    patients_response = supabase.table("tbl_patient").select("pat_id").execute()
    patients = patients_response.data

    for patient in patients:
        pat_id = patient["pat_id"]

        # قراءة القراءات الخاصة بالمريض من جدول tbl_reading
        readings_response = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).execute()
        readings = readings_response.data

        data_batch = []
        for r in readings:
            if all(k in r for k in ["oxygen_saturation", "pulse_rate", "temperature"]):
                data_batch.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])

        if data_batch:
            filename = train_patient_model(pat_id, data_batch)
            uploaded = upload_file_to_github(filename)
            status = "تم التدريب والرفع" if uploaded else "تم التدريب لكن فشل الرفع"
        else:
            status = "بيانات غير كافية"

        results.append({"pat_id": pat_id, "status": status})

    return results
