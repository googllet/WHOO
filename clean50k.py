import pandas as pd
import re

# โหลดชุดข้อมูล
file_path = "sampled_malicious_urls_50k.csv"  # ปรับเป็น path ที่ถูกต้อง
df = pd.read_csv(file_path)

# ดูข้อมูลเบื้องต้น
print(df.head())
print(df.info())

# ลบค่า Missing Values หากมี
df.dropna(inplace=True)

# แปลง URL ทั้งหมดให้เป็นตัวพิมพ์เล็ก
df['url'] = df['url'].str.lower()

# ลบ query parameters และ fragment identifiers ใน URL
df['cleaned_url'] = df['url'].apply(lambda x: re.sub(r'\?.*|#.*', '', x))

# ลบโปรโตคอล (http, https) และ www.
df['cleaned_url'] = df['cleaned_url'].apply(lambda x: re.sub(r'(https?://)?(www\.)?', '', x))

# ลบเครื่องหมายพิเศษที่ไม่จำเป็น
df['cleaned_url'] = df['cleaned_url'].apply(lambda x: re.sub(r'[^\w./-]', '', x))

# แสดงตัวอย่างข้อมูลที่สะอาดแล้ว
print(df[['url', 'cleaned_url']].head())

# บันทึกข้อมูลใหม่เป็นไฟล์ CSV
df.to_csv("cleaned50k_urls.csv", index=False)