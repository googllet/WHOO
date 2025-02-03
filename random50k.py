import pandas as pd

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("cleaned_unique_malicious_urls.csv")

# ตรวจสอบขนาดข้อมูล
print("จำนวนข้อมูลก่อนสุ่ม:", len(df))

# สุ่มเลือกข้อมูล 2000 แถว
df_sampled = df.sample(n=50000, random_state=42)

# บันทึกข้อมูลที่สุ่มเลือกไว้ในไฟล์ใหม่
df_sampled.to_csv("sampled_malicious_urls_50k.csv", index=False)

print("จำนวนข้อมูลหลังสุ่ม:", len(df_sampled))
print(df_sampled['type'].value_counts())