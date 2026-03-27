from fastapi import FastAPI
import os, pandas as pd, joblib, base64, requests, datetime
from supabase import create_client
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# تحميل متغيرات البيئة
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")   # ضع الـ Token في إعدادات Render
GITHUB_REPO = os.getenv("GITHUB_REPO")     # مثال: "username/repo"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🟢 رفع الملف إلى GitHub
def upload_to_github(file_path, repo, token, branch="main"):
    with open(file_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode("utf-8")

    # اسم الملف مع التاريخ لتخزين نسخ متعددة
    filename = f"health_model_{datetime.date.today()}.pkl"
    url = f"https://api.github.com/repos/{repo}/contents/models/{filename}"

    headers = {"Authorization": f"token {token}"}
    data = {
        "message": f"Upload trained health model {filename}",
        "content": encoded,
        "branch": branch
    }
    response = requests.put(url, headers=headers, json=data)
    return response.json()

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

    os.makedirs("models", exist_ok=True)
    model_path = "models/health_model.pkl"
    joblib.dump(model, model_path)

    # رفع الملف إلى GitHub
    github_status = upload_to_github(model_path, GITHUB_REPO, GITHUB_TOKEN)

    return {
        "message": "تم تدريب النموذج وحفظه بنجاح",
        "accuracy": model.score(X, y),
        "github_status": github_status
    }@app.get("/train/model")
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

    os.makedirs("models", exist_ok=True)
    model_path = "models/health_model.pkl"
    joblib.dump(model, model_path)

    # رفع الملف إلى GitHub
    github_status = upload_to_github(model_path, GITHUB_REPO, GITHUB_TOKEN)

    return {
        "message": "تم تدريب النموذج وحفظه بنجاح",
        "accuracy": model.score(X, y),
        "github_status": github_status
    }    readings = supabase.table("tbl_reading").select("*").execute().data
    df = pd.DataFrame(readings)
    if df.empty:
        return {"error": "لا توجد بيانات للتدريب"}

    df["is_emergency"] = ((df["temperature"] > 38) | (df["oxygen_saturation"] < 90)).astype(int)
    X = df[["temperature", "oxygen_saturation", "pulse_rate"]]
    y = df["is_emergency"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model_path = "models/health_model.pkl"
    joblib.dump(model, model_path)

    # رفع الملف إلى GitHub
    github_status = upload_to_github(model_path, GITHUB_REPO, GITHUB_TOKEN)

    return {
        "message": "تم تدريب النموذج وحفظه بنجاح",
        "accuracy": model.score(X, y),
        "github_status": github_status
    }
