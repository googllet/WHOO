import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# โหลดข้อมูล
df = pd.read_csv("sampled_malicious_urls_50k.csv")

# ตรวจสอบข้อมูลก่อนแปลง
print("ข้อมูลก่อนแปลง:\n", df.head())

# ตรวจสอบว่าไม่มีค่า NaN ในข้อมูล
df = df.dropna(subset=['cleaned_url', 'type'])

# แปลงข้อความ URL ด้วย TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)  # จำกัดฟีเจอร์เพื่อประสิทธิภาพ
X = vectorizer.fit_transform(df['cleaned_url'])
y = df['type']

# ใช้ SMOTE เพื่อปรับสมดุลข้อมูล
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# สร้าง DataFrame ใหม่หลังปรับสมดุล
df_resampled = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
df_resampled['type'] = y_resampled.reset_index(drop=True)

# บันทึกข้อมูลที่ปรับสมดุล
df_resampled.to_csv("balanced50k_sampled_urls.csv", index=False)
print("ปรับสมดุลและบันทึกข้อมูลสำเร็จ!")
# แสดงจำนวนข้อมูลแต่ละประเภท
type_counts = df_resampled['type'].value_counts()
print("ข้อมูลแต่ละประเภทหลังปรับสมดุล:\n", type_counts)