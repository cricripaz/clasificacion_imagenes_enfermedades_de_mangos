import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64

from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap

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

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if request.files:
            file = request.files['image']
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                file.save(file_path)

                img_batch = preprocess_image(file_path)

                predictions = model.predict(img_batch)

                predicted_class = np.argmax(predictions)

                predicted_class_name = {
                    0: 'Alternaria',
                    1: 'Anthracnose',
                    2: 'Black Mould Rot',
                    3: 'HEALTHY',
                    4: 'STEM END ROT'
                }[predicted_class]

                os.remove(file_path)

                return jsonify({'class': predicted_class_name})

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)