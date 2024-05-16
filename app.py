# main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import warnings
import holidays

warnings.filterwarnings('ignore')

model = joblib.load('crime_prediction.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

br_holidays = holidays.Brazil()

def process_datetime(dt: datetime):
    features = [
        dt.year, dt.month, dt.day, dt.weekday(), dt.hour, dt.minute
    ]
    return features

@app.get('/predict/')
def predict_periods():
    predictions = []
    current_datetime = datetime.now()
    
    periods = [
        ('morning', current_datetime.replace(hour=5, minute=0, second=0, microsecond=0),
         current_datetime.replace(hour=12, minute=0, second=0, microsecond=0)),
        
        ('afternoon', current_datetime.replace(hour=12, minute=0, second=0, microsecond=0),
         current_datetime.replace(hour=19, minute=0, second=0, microsecond=0)),
        
        ('night', current_datetime.replace(hour=19, minute=0, second=0, microsecond=0),
         (current_datetime + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0))
    ]
    
    for period_name, start_time, end_time in periods:
        if end_time < start_time:
            end_time += timedelta(days=1)
        
        time_point = start_time
        while time_point < end_time:
            features = process_datetime(time_point)
            predicted_coords = model.predict([features])[0]
            
            predictions.append({
                'period': period_name,
                'datetime': time_point,
                'latitude': predicted_coords[0],
                'longitude': predicted_coords[1]
            })
            
            time_point += timedelta(minutes=15)
            
    return predictions

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
