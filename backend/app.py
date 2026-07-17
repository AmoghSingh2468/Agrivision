

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename

from utils.disease_info import DISEASE_INFO, CLASS_NAMES
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam, get_last_conv_layer_name

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_plant_disease_model.keras")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG"}), 400
        
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        
        img_array, original_img = preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        
        disease_info = DISEASE_INFO.get(predicted_class, {
            'severity': 50,
            'treatment': ['Consult agricultural expert', 'Monitor plant closely', 'Apply general care'],
            'prevention': ['Maintain plant health', 'Regular monitoring', 'Good sanitation']
        })
        
        
        last_conv_layer = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, predicted_class_idx)
        
        
        original_img_array = np.array(original_img)
        gradcam_overlay = overlay_gradcam(original_img_array, heatmap)
        
        
        gradcam_pil = Image.fromarray(gradcam_overlay)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_colored, (128, 128))
        heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB))
        
        
        response = {
            "success": True,
            "prediction": {
                "disease": predicted_class.replace('___', ' - ').replace('_', ' '),
                "confidence": round(confidence, 2),
                "raw_class": predicted_class
            },
            "severity": {
                "level": disease_info['severity'],
                "label": get_severity_label(disease_info['severity'])
            },
            "treatment": disease_info['treatment'],
            "prevention": disease_info['prevention'],
            "images": {
                "original": image_to_base64(original_img),
                "heatmap": image_to_base64(heatmap_pil),
                "gradcam": image_to_base64(gradcam_pil)
            }
        }
        
        
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_severity_label(severity):
    """Get severity label based on percentage"""
    if severity == 0:
        return "Healthy"
    elif severity < 50:
        return "Mild"
    elif severity < 70:
        return "Moderate"
    elif severity < 85:
        return "Severe"
    else:
        return "Critical"

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available disease classes"""
    return jsonify({
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)