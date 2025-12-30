import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import os
from sklearn.dummy import DummyClassifier


tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(tracking_uri)

model = None

try:
    print("Model yüklenmeye başladı")
    model = mlflow.pyfunc.load_model(
        model_uri="models:/CourseCompletionModel/Production"
    )
    print("Model yüklendi")
except Exception as e:
    print(f"Model yüklenemedi: {e}")


# Fallback model
fallback_model = DummyClassifier(strategy="most_frequent")
fallback_model.fit([[0]], [0])


# Modelin beklediği KESİN sütun listesi (Hata mesajından alındı)
EXPECTED_COLUMNS = [
    'Name', 'Gender', 'Age', 'Education_Level', 'Employment_Status', 'City', 
    'Device_Type', 'Internet_Connection_Quality', 'Course_ID', 'Course_Name', 
    'Category', 'Course_Level', 'Course_Duration_Days', 'Instructor_Rating', 
    'Login_Frequency', 'Average_Session_Duration_Min', 'Video_Completion_Rate', 
    'Discussion_Participation', 'Time_Spent_Hours', 'Days_Since_Last_Login', 
    'Notifications_Checked', 'Peer_Interaction_Score', 'Assignments_Submitted', 
    'Assignments_Missed', 'Quiz_Attempts', 'Quiz_Score_Avg', 'Project_Grade', 
    'Rewatch_Count', 'Payment_Mode', 'Fee_Paid', 'Discount_Used', 
    'Payment_Amount', 'App_Usage_Percentage', 'Reminder_Emails_Clicked', 
    'Support_Tickets_Raised', 'Satisfaction_Rating', 'Enrollment_Month', 
    'Category_Level_Cross'
]
# Hashed ID sütunlarını ekle (0'dan 49'a kadar)
for i in range(50):
    EXPECTED_COLUMNS.append(f'hashed_Student_ID_{i}')

class PredictionInput(BaseModel):
    # Kullanıcıdan sadece basit featurelar alıyoruz, gerisini biz dolduracağız
    features: list 

@app.get("/")
def home():
    return {"status": "Active", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PredictionInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model servisi çalışmıyor.")
    
    try:
        # 1. Boş bir DataFrame oluştur (Tüm sütunlar 0 olsun)
        input_df = pd.DataFrame(0, index=[0], columns=EXPECTED_COLUMNS)
        
        # 2. Kullanıcıdan gelen verileri ilk sütunlara yerleştir (Demo amaçlı)
        # Gelen veriyi güvenli bir şekilde yerleştirelim
        user_data = data.features[0] if isinstance(data.features[0], list) else data.features
        
        # Kullanıcı verisi kadar sütunu doldur, kalanı 0 kalsın
        limit = min(len(user_data), len(EXPECTED_COLUMNS))
        for i in range(limit):
            col_name = EXPECTED_COLUMNS[i]
            input_df[col_name] = user_data[i]
            
        # 3. Tahmin Yap (Artık sütun isimleri ve sayısı tutuyor!)
        prediction = model.predict(input_df)
        
        try:
    prediction = model.predict(input_df)
    return {
        "prediction": prediction.tolist(),
        "model": "production"
    }
except Exception:
    prediction = fallback_model.predict(input_df)
    return {
        "prediction": prediction.tolist(),
        "model": "fallback"
    }

