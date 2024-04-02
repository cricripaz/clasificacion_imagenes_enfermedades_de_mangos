import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64

from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap
import cv2

# Initialize Flask app
app = Flask(__name__)

# Initialize Bootstrap
Bootstrap(app)

# Load the model
model = load_model('model_rnn/mangos_model.h5')

# Function for preprocessing the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def predict_class(image):
    if isinstance(image, str):  # Si es una ruta de archivo (imagen subida)
        img = Image.open(image)
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
    else:  # Si es una matriz de imagen (imagen capturada desde la cámara)
        img_array = cv2.resize(image, (150, 150))
        img_array = img_array / 255.0

    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions)
    predicted_class_name = {
        0: 'Alternaria',
        1: 'Anthracnose - Antracnosis',
        2: 'Black Mould Rot - Moho Negro',
        3: 'Healthy - Sano',
        4: 'STEM END ROT - Podredumbre del Tallo'
    }[predicted_class]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' in request.files:
            # Si se está subiendo una imagen
            file = request.files['image']
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                predicted_class_name = predict_class(file_path)
                os.remove(file_path)
                return jsonify({'class': predicted_class_name})
        elif 'image_data' in request.form:
            # Si se está capturando una imagen desde la cámara
            image_data = request.form['image_data']
            decoded_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(decoded_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            predicted_class_name = predict_class(img)
            return jsonify({'class': predicted_class_name})

    return render_template('upload.html')

if __name__ == '__main__':
    app.run()
