import pandas as pd

df = pd.read_csv("cleaned_malicious_urls.csv")

print("จำนวน URL ที่ซ้ำกัน:", df['cleaned_url'].duplicated().sum())

if 'type' in df.columns:
    print(df['type'].value_counts(normalize=True))

df['cleaned_url'] = df['cleaned_url'].astype(str)  # แปลงข้อมูลให้เป็น string
df['domain'] = df['cleaned_url'].apply(lambda x: x.split('/')[0] if x != 'nan' else '')
print("ค่าที่เป็น NaN:", df['cleaned_url'].isna().sum())
print(df['cleaned_url'].head(10))  # ดูตัวอย่างข้อมูล
df.dropna(subset=['cleaned_url'], inplace=True)

from collections import Counter
df['domain'] = df['cleaned_url'].apply(lambda x: x.split('/')[0])
common_domains = Counter(df['domain']).most_common(10)
print("โดเมนยอดนิยม:", common_domains)

# ลบ URL ที่ซ้ำกันและบันทึกไฟล์ใหม่
df.drop_duplicates(subset='cleaned_url', inplace=True)
df.to_csv("cleaned_unique_malicious_urls.csv", index=False)
