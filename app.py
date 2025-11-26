from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model - model file will be in the same folder as app.py
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = YOLO(MODEL_PATH)

# Store the latest prediction results
latest_results = None

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    global latest_results
    
    if 'file' not in request.files:
        return redirect(url_for('input_page'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('input_page'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = model(filepath)
        
        output_filename = f"output_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save(output_path)
            
            latest_results = {
                'original_image': os.path.join('static', 'uploads', filename),
                'output_image': os.path.join('static', 'outputs', output_filename),
                'detections': []
            }
            
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    latest_results['detections'].append({
                        'class': class_name,
                        'confidence': f"{conf:.2%}"
                    })
        
        return redirect(url_for('output_page'))

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    global latest_results
    
    image_data = request.files.get('image')
    
    if image_data:
        img_bytes = image_data.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        filename = "webcam_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        
        results = model(filepath)
        
        output_filename = f"output_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save(output_path)
            
            latest_results = {
                'original_image': os.path.join('static', 'uploads', filename),
                'output_image': os.path.join('static', 'outputs', output_filename),
                'detections': []
            }
            
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    latest_results['detections'].append({
                        'class': class_name,
                        'confidence': f"{conf:.2%}"
                    })
        
        return {'success': True}
    
    return {'success': False}

@app.route('/output')
def output_page():
    global latest_results
    return render_template('output.html', results=latest_results)

# DO NOT run app.run() in Docker; Gunicorn will start it.
# You can keep this for local testing only:
if __name__ == '__main__':
    app.run(debug=True, port=5000)
