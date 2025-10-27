from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load trained MNIST CNN model
model = load_model("model/mnist_cnn.h5")


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Open image and convert to RGB
    img = Image.open(filepath).convert('RGB')

    # Convert to grayscale
    img = ImageOps.grayscale(img)

    # Optional: invert colors for MNIST style (white digit on black)
    img = ImageOps.invert(img)

    # Resize to 28x28 pixels
    img = img.resize((28, 28))

    # Convert to array and normalize
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1) / 255.0

    # Predict digit
    pred = model.predict(img)
    digit = np.argmax(pred, axis=1)[0]

    # Pass prediction and image path to template
    return render_template('index.html', prediction=digit, filepath=filepath)


if __name__ == '__main__':
    app.run(port=5000, debug=True)