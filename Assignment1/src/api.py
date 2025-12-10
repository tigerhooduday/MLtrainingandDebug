# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
preprocessor = joblib.load("../models/preprocessor.joblib")
model = joblib.load("../models/xgb_best.joblib")

class Employee(BaseModel):
    age: float
    gender: str
    education: str
    department: str
    job_role: str
    years_at_company: float
    promotions: int
    overtime: str
    performance_rating: float
    monthly_income: float

@app.post("/predict")
def predict(emp: Employee):
    row = pd.DataFrame([emp.dict()])
    proba = float(model.predict_proba(row)[:,1][0])
    return {"attrition_prob": proba}
