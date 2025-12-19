from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
import torch
from PIL import Image
import torchvision.transforms as transforms

# 🔹 Correct relative imports (VERY IMPORTANT)
from models.haze_cnn import HazeCNN
from backend.models.aqi_lstm import AQILSTM
from backend.models.aqi_transformer import AQITransformer
from backend.services.weather import fetch_weather
from backend.services.openaq import fetch_no2
from backend.blockchain.audit_chain import create_audit_hash

# ---------------------------------------------
# FastAPI App
# ---------------------------------------------
app = FastAPI(
    title="Air Quality Monitoring",
    description="AI Driven Real Time Air Quality Monitoring and Predictive Mitigation using Multi Modal Sensor Data and Machine Learning Models",
    version="1.0.0"
)

# ---------------------------------------------
# Root health check (prevents 404 confusion)
# ---------------------------------------------
@app.get("/")
def root():
    return {
        "status": "System running",
        "message": "Use /docs for API interface"
    }

# ---------------------------------------------
# Input Schema
# ---------------------------------------------
class AQIInput(BaseModel):
    latitude: float
    longitude: float
    city: str

# ---------------------------------------------
# Image preprocessing
# ---------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# ---------------------------------------------
# Model Initialization
# ---------------------------------------------
device = torch.device("cpu")

# Model initialization (FIXED)
haze_model = HazeCNN().to(device)
lstm_model = AQILSTM(input_size=6).to(device)      # ✅ FIX
transformer_model = AQITransformer(input_size=6).to(device)


haze_model.eval()
lstm_model.eval()
transformer_model.eval()

# ---------------------------------------------
# Prediction Endpoint
# ---------------------------------------------
@app.post("/predict")
async def predict(data: AQIInput, image: UploadFile = File(...)):
    """
    Multi-modal AQI prediction:
    - Image-based haze estimation (CNN)
    - Weather data (OpenWeatherMap)
    - Satellite NO2 (OpenAQ)
    - Temporal learning (LSTM)
    - Spatio-temporal fusion (Transformer)
    - Blockchain audit hash
    """

    # --- Image → Haze Index ---
    img = Image.open(image.file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        haze_index = haze_model(img_tensor).item()

    # --- External real-time data ---
    weather = fetch_weather(data.latitude, data.longitude)
    no2 = fetch_no2(data.city) or 0.0

    # --- Feature tensor [batch, seq, features] ---
    features = torch.tensor(
        [[[
            haze_index,
            no2,
            weather["temperature"],
            weather["humidity"],
            weather["wind"],
            weather["pressure"]
        ]]],
        dtype=torch.float32
    ).to(device)

    # --- AI Predictions ---
    with torch.no_grad():
        aqi_lstm = lstm_model(features).item()
        aqi_final = transformer_model(features).item()

    # --- Blockchain Audit Payload ---
    payload = {
        "city": data.city,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "aqi": round(aqi_final, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

    audit_hash = create_audit_hash(payload)

    # --- Final Response ---
    return {
        "Final_AQI": round(aqi_final, 2),
        "LSTM_Output": round(aqi_lstm, 2),
        "Satellite_NO2": round(no2, 2),
        "Haze_Index": round(haze_index, 3),
        "Weather": weather,
        "Blockchain_Audit_Hash": audit_hash,
        "Timestamp": payload["timestamp"]
    }
