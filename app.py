from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
from keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained model
model = load_model(r'C:\Users\mohan\Downloads\mini-project-itc-9\tomato plant disease detection\tomato_disease.h5')

# Class labels for the tomato diseases
class_labels = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Disease prevention tips
disease_prevention = {
    "Tomato_Bacterial_spot": [
        "Prevent bacterial spot by using disease-free seeds.",
        "Implement crop rotation to reduce the disease's prevalence.",
        "Apply copper-based fungicides to control the disease."
    ],
    "Tomato_Early_blight": [
        "Prevent early blight by practicing good garden hygiene.",
        "Ensure proper watering to avoid splashing soil onto the leaves.",
        "Apply fungicides as needed to control the disease."
    ],
    "Tomato_Late_blight": [
        "Prevent late blight by providing good air circulation in your garden or greenhouse.",
        "Avoid overhead watering, as wet leaves can encourage the disease.",
        "Apply fungicides when necessary to manage the disease."
    ],
    "Tomato_Leaf_Mold": [
        "Prevent leaf mold by ensuring good air circulation and spacing between plants.",
        "Avoid wetting the leaves when watering, and water the soil instead.",
        "Apply fungicides if the disease is present and worsening."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Prevent Septoria leaf spot by maintaining good garden hygiene.",
        "Avoid overhead watering to keep the leaves dry.",
        "Apply fungicides if the disease becomes a problem."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Prevent spider mite infestations by regularly inspecting your plants for signs of infestation.",
        "Increase humidity in the growing area to discourage mites.",
        "Use insecticidal soap or neem oil to control mites if necessary."
    ],
    "Tomato__Target_Spot": [
        "Prevent target spot by ensuring good air circulation and avoiding overcrowding of plants.",
        "Water at the base of the plants, keeping the leaves dry.",
        "Apply fungicides as needed to control the disease."
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "Prevent Tomato Yellow Leaf Curl Virus by using virus-free tomato plants.",
        "Control whiteflies, which transmit the virus, with insecticides.",
        "Remove and destroy infected plants to prevent the spread of the disease."
    ],
    "Tomato__Tomato_mosaic_virus": [
        "Prevent tomato mosaic virus by using virus-free seeds and disease-resistant tomato varieties.",
        "Control aphids, which transmit the virus, with insecticides.",
        "Remove and destroy infected plants to prevent further spread."
    ],
    "Tomato_healthy": [
        "If your tomato plant is healthy, continue to monitor for pests and diseases regularly.",
        "Follow good gardening practices, including proper watering, fertilization, and maintenance."
    ]
}

def preprocess_image(img_path):

    
    # load image
    img = load_img(img_path, target_size=(256, 256))

    # convert image into areay
    img_array = img_to_array(img)

    # Normalize pixel values to [0, 1]
    img = img_array/255

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_labels[predicted_class_index]
            prevention_tips = disease_prevention.get(predicted_class, [])
            return render_template('result.html', prediction=predicted_class, prevention_tips=prevention_tips)
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
