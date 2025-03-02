from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Tên file dữ liệu
DATA_FILE = "game_data.csv"

# Đọc dữ liệu từ file (nếu có)
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["feature1", "feature2", "result", "timestamp"])

# Trang chủ
@app.route('/')
def home():
    data = load_data().tail(10)  # Hiển thị 10 dòng cuối
    return render_template("index.html", data=data.to_dict(orient="records"))

# Nhập dữ liệu mới
@app.route('/add', methods=['POST'])
def add_data():
    result = request.form.get("result")
    if result not in ["0", "1"]:
        return redirect(url_for('home'))  # Tránh lỗi nhập sai
    
    df = load_data()
    new_row = {"feature1": 1, "feature2": 0, "result": int(result), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    df = df.append(new_row, ignore_index=True)
    df.to_csv(DATA_FILE, index=False)  # Lưu dữ liệu

    return redirect(url_for('home'))

# Huấn luyện mô hình
@app.route('/train')
def train_model():
    df = load_data()
    if df.empty:
        return "Không có dữ liệu để huấn luyện. Hãy nhập dữ liệu trước!"

    X = df[["feature1", "feature2"]]
    y = df["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return f"Huấn luyện thành công! Độ chính xác: {accuracy * 100:.2f}%"

# Dự đoán kết quả
@app.route('/predict')
def predict():
    df = load_data()
    if df.empty:
        return "Chưa có dữ liệu, không thể dự đoán!"

    X = df[["feature1", "feature2"]]
    y = df["result"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    prediction = model.predict([[1, 0]])[0]
    return f"Kết quả dự đoán: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)
