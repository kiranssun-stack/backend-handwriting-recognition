import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import model_train

app = FastAPI(title="MNIST Predictor API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model check/load
model_path = "model.pkl"
model = None

def load_or_train_model():
    global model
    if not os.path.exists(model_path):
        print("Model file not found. Training model...")
        model_train.train_model()
    
    print("Loading model...")
    model = joblib.load(model_path)

@app.on_event("startup")
async def startup_event():
    load_or_train_model()

class PredictionRequest(BaseModel):
    # Expecting a list of 784 normalized pixel values (0.0 to 1.0)
    image_data: List[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    if len(request.image_data) != 784:
        raise HTTPException(status_code=400, detail="Invalid image data size. Expected 784 pixels.")

    # Reshape and predict
    data = np.array(request.image_data).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(data)[0]
    
    # Get probability (confidence)
    probabilities = model.predict_proba(data)[0]
    confidence = float(np.max(probabilities))
    
    return {
        "prediction": str(prediction),
        "confidence": confidence
    }

# Serving static files (Vite build output)
dist_path = "dist"

# Health check or Root message if dist doesn't exist
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# Mount static files if they exist
if os.path.exists(dist_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Prevent intercepting API calls if they start with 'predict'
        if full_path.startswith("predict"):
             return # Let the POST handler take it
             
        index_file = os.path.join(dist_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"error": "Index file not found"}
else:
    @app.get("/")
    def root_no_dist():
        return {"message": "MNIST Digit Recognition API. Frontend not found. Build with 'npm run build'."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
