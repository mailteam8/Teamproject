import pickle
from supabase import create_client, Client

# الاتصال بـ Supabase
url = "https://kzqcznveyxallyonedls.supabase.co"
key = "sb_publishable_rUhjaGNhHHlkwHis22Fqkg_mG2Fswbz"
supabase: Client = create_client(url, key)

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

def load_patient_model(patient_id):
    filename = f"heart_guard_{patient_id}.pkl"
    with open(filename, "rb") as f:
        stats = pickle.load(f)
    return stats

# قراءة المرضى من جدول tbl_patient
patients_response = supabase.table("tbl_patient").select("*").execute()
patients = patients_response.data

for patient in patients:
    pat_id = patient["pat_id"]
    print(f"📌 تدريب النموذج للمريض {pat_id}")

    # جلب بيانات المريض من جدول patient_data (افترض أن هذا الجدول موجود)
    data_response = supabase.table("patient_data").select("*").eq("pat_id", pat_id).execute()
    data_batch = [
        [row["ecg"], row["bp"], row["temp"], row["oxygen"], row["movement"]]
        for row in data_response.data
    ]

    # تدريب النموذج
    if data_batch:
        train_patient_model(pat_id, data_batch)
    else:
        print(f"⚠️ لا توجد بيانات للمريض {pat_id}")
