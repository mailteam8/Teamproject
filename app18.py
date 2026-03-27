from fastapi import FastAPI
import os, pickle, base64, requests
from supabase import create_client, Client

app = FastAPI()

# الاتصال بـ Supabase عبر متغيرات البيئة
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# إعدادات GitHub عبر متغيرات البيئة
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPO")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")

def train_patient_model(patient_id, data_batch):
    stats = []
    for col in zip(*data_batch):
        avg = sum(col)/len(col)
        stats.append({"avg": avg, "min": min(col), "max": max(col)})
    filename = f"heart_guard_{patient_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(stats, f)
    return filename

def upload_file_to_github(filepath, repo=REPO, branch=BRANCH):
    filename = os.path.basename(filepath)
    url = f"https://api.github.com/repos/{repo}/contents/models/{filename}"
    with open(filepath, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    data = {"message": f"إضافة نموذج {filename}", "content": content, "branch": branch}
    requests.put(url, headers=headers, json=data)

@app.get("/train")
def train_all_patients():
    patients_response = supabase.table("tbl_patient").select("*").execute()
    patients = patients_response.data
    results = []
    for patient in patients:
        pat_id = patient["pat_id"]
        if all(k in patient for k in ["ecg","bp","temp","oxygen","movement"]):
            data_batch = [[patient["ecg"], patient["bp"], patient["temp"], patient["oxygen"], patient["movement"]]]
            filename = train_patient_model(pat_id, data_batch)
            upload_file_to_github(filename)
            results.append({"pat_id": pat_id, "status": "تم التدريب والرفع"})
        else:
            results.append({"pat_id": pat_id, "status": "بيانات غير كافية"})
    return results
