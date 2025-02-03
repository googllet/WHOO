import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

# อ่านข้อมูลจากไฟล์
df = pd.read_csv("balanced50k_sampled_urls.csv")

# ตั้งชื่อคอลัมน์เป้าหมายให้ตรงกับข้อมูล
target_column = 'type'

# ตรวจสอบว่าคอลัมน์เป้าหมายอยู่ใน DataFrame
if target_column not in df.columns:
    raise ValueError(f"ไม่พบคอลัมน์ '{target_column}' ในข้อมูล กรุณาระบุชื่อคอลัมน์ที่ถูกต้อง")

# แยกข้อมูลเป็น X และ y
X = df.drop(target_column, axis=1)
y = df[target_column]

# แปลงข้อมูลเป้าหมายให้เป็นค่าเชิงตัวเลขหากยังไม่ใช่
if y.dtype == 'object':
    y = y.astype('category').cat.codes

# แบ่งข้อมูลสำหรับ train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset indices to prevent index issues
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# ใช้ KFold เพื่อแบ่งข้อมูล train เป็น training และ validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_accuracy = 0

# ปรับพารามิเตอร์ที่สำคัญ
params = {
    'objective': 'multiclass' if y.nunique() > 2 else 'binary',
    'boosting_type': 'gbdt',
    'metric': 'multi_error' if y.nunique() > 2 else 'binary_error',
    'num_class': y.nunique() if y.nunique() > 2 else 1,
    'verbosity': -1,
    'num_leaves': 1500,
    'max_depth': 40,
    'min_data_in_leaf': 20,  # จำนวนข้อมูลในแต่ละ leaf
    'subsample': 1.0,
    'force_col_wise': True  # ลด overhead
}

for train_idx, val_idx in kf.split(X_train):
    # แบ่งข้อมูลสำหรับ training และ validation
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # สร้างโมเดล LightGBM
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # เทรนโมเดลพร้อม callback สำหรับ Early Stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,  # ลดจำนวน boost round
        callbacks=[lgb.early_stopping(stopping_rounds=20)]  # ใช้ early stopping
    )

    # ทำนายผล
    y_pred = model.predict(X_val)
    y_pred_binary = [pred.argmax() if params['objective'] == 'multiclass' else int(pred > 0.5) for pred in y_pred]

    # ประเมินความแม่นยำ
    accuracy = accuracy_score(y_val, y_pred_binary)
    print(f"Accuracy: {accuracy:.2f}")
    
    # เก็บค่าความแม่นยำที่ดีที่สุด
    best_accuracy = max(best_accuracy, accuracy)

print(f"Best Accuracy: {best_accuracy:.2f}")

# บันทึกโมเดลที่ดีที่สุด
model.save_model("lightgbm_model.txt")
print("โมเดลถูกบันทึกลงไฟล์ 'lightgbm_model.txt'")
