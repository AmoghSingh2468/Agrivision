# AgriVision 🌿

AgriVision is a plant disease detection web app. Upload a photo of a plant leaf and it predicts the disease (or confirms the plant is healthy), scores severity, and shows a Grad-CAM heatmap so you can see exactly which part of the leaf the model is focusing on.

## How It Works

1. You upload a leaf image through the web UI.
2. A TensorFlow/Keras CNN (trained on 38 plant/disease classes across crops like apple, blueberry, corn, tomato, etc.) classifies the image.
3. Grad-CAM generates a heatmap over the last convolutional layer to visualize which regions influenced the prediction.
4. The app returns the predicted disease, a confidence score, a severity level (Healthy → Mild → Moderate → Severe → Critical), and treatment/prevention tips pulled from a built-in knowledge base.

## Tech Stack

- **Backend:** Flask + Flask-CORS
- **Model:** TensorFlow/Keras (`.keras` model, 128x128 input)
- **Explainability:** Grad-CAM (OpenCV for heatmap overlay)
- **Image handling:** Pillow, OpenCV
- **Deployment:** Docker (Gunicorn) — configured for Railway/Vercel

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/api/health` | Health check, confirms model is loaded |
| `POST` | `/api/predict` | Accepts an image file (`multipart/form-data`, field name `file`), returns prediction + Grad-CAM visualization |
| `GET` | `/api/classes` | Lists all supported disease classes |

### Example: `/api/predict` response

```json
{
  "success": true,
  "prediction": {
    "disease": "Apple - Apple scab",
    "confidence": 94.32,
    "raw_class": "Apple___Apple_scab"
  },
  "severity": {
    "level": 70,
    "label": "Severe"
  },
  "treatment": ["..."],
  "prevention": ["..."],
  "images": {
    "original": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "gradcam": "data:image/png;base64,..."
  }
}
```

## Project Structure

```
backend/
├── app.py                  # Flask app & API routes
├── models/
│   └── trained_plant_disease_model.keras
├── utils/
│   ├── gradcam.py           # Grad-CAM heatmap generation
│   └── disease_info.py      # Class names + treatment/prevention data
├── templates/
│   └── index.html
├── static/
├── Dockerfile
├── Procfile
├── vercel.json
└── requirements.txt
```

## Running Locally

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The app runs on `http://localhost:7860` by default (set `PORT` to override).

## Running with Docker

```bash
cd backend
docker build -t agrivision .
docker run -p 7860:7860 agrivision
```

## Supported Plants

Includes disease detection for Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato — 38 classes total, including healthy-leaf classes for each crop.
