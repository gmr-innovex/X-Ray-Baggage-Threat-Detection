from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['YOLO_MODEL_PATH'] = 'best.pt'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = YOLO(app.config['YOLO_MODEL_PATH'])

# Setup logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            logging.debug(f"File saved to {filepath}")
            results = model(filepath)
            output_image = results[0].plot()  # Get the result image as a numpy array
            output_image = Image.fromarray(output_image)  # Convert to PIL Image
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + file.filename)
            output_image.save(output_image_path)  # Save the predicted image
            logging.debug(f"Predicted image saved to {output_image_path}")
            return render_template('index.html', uploaded_image=file.filename, predicted_image='predicted_' + file.filename)
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return f"Error processing file: {e}"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
