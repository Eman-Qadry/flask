import os
import gdown
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)


FILE_ID = '1vvnxauxIE5UdtsX-9ho_CX8EHU-8hDGA'
LOCAL_MODEL_PATH = 'model.h5'

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, LOCAL_MODEL_PATH, quiet=False)

download_model()
model = load_model(LOCAL_MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    print(predictions)
    return predictions

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Flask API"


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
