import os
import pickle
import base64
import requests
from supabase import create_client, Client

# الاتصال بـ Supabase عبر متغيرات البيئة
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# إعدادات GitHub عبر متغيرات البيئة
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")   # ضع التوكن في Render كمتغير بيئة
REPO = os.environ.get("GITHUB_REPO")            # مثال: "username/repo"
BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# دوال النموذج
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
    print(f"✅ تم تحديث النموذج الشخصي للمريض {patient_id}")
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
    if response.status_code in [200, 201]:
        print(f"✅ تم رفع الملف {filename} إلى GitHub")
    else:
        print(f"⚠️ فشل رفع الملف {filename}: {response.json()}")

# قراءة المرضى من جدول tbl_patient
patients_response = supabase.table("tbl_patient").select("*").execute()
patients = patients_response.data

for patient in patients:
    pat_id = patient["pat_id"]
    print(f"📌 تدريب النموذج للمريض {pat_id}")

    # استخدام بيانات القياسات من نفس الجدول
    if all(k in patient for k in ["ecg", "bp", "temp", "oxygen", "movement"]):
        data_batch = [[patient["ecg"], patient["bp"], patient["temp"], patient["oxygen"], patient["movement"]]]
        filename = train_patient_model(pat_id, data_batch)
        upload_file_to_github(filename)
    else:
        print(f"⚠️ لا توجد بيانات كافية للمريض {pat_id}")
