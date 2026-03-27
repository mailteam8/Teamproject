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
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPO")
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

def load_patient_model(patient_id):
    filename = f"heart_guard_{patient_id}.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def predict(patient_id, new_reading):
    stats = load_patient_model(patient_id)
    if not stats:
        return f"⚠️ لا يوجد نموذج للمريض {patient_id}"
    results = []
    for i, val in enumerate(new_reading):
        avg, min_val, max_val = stats[i]["avg"], stats[i]["min"], stats[i]["max"]
        status = "طبيعي" if min_val <= val <= max_val else "خارج النطاق"
        results.append({"value": val, "avg": avg, "range": (min_val, max_val), "status": status})
    return results

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

# قراءة القراءات من جدول tbl_reading
readings_response = supabase.table("tbl_reading").select("*").execute()
readings = readings_response.data

# تجميع القراءات لكل مريض
patients_data = {}
for r in readings:
    pat_id = r["pat_id"]
    if pat_id not in patients_data:
        patients_data[pat_id] = []
    if all(k in r for k in ["oxygen_saturation", "pulse_rate", "temperature"]):
        patients_data[pat_id].append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])

# تدريب النماذج ورفعها
for pat_id, data_batch in patients_data.items():
    if data_batch:
        filename = train_patient_model(pat_id, data_batch)
        upload_file_to_github(filename)
    else:
        print(f"⚠️ لا توجد بيانات كافية للمريض {pat_id}")

# مثال للتنبؤ باستخدام قراءة جديدة
example_reading = [95.0, 72, 36.8]  # oxygen_saturation, pulse_rate, temperature
print("🔮 نتيجة التنبؤ:", predict("10ac3acc-e296-4579-a749-7d61ad54ee5d", example_reading))
