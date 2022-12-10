from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow.keras.models import save_model,load_model
import numpy as np
from predict import Attribute_classify
from utils import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

"""class ClientApp:
    def __init__(self):
        self.filename = "image_974.jpg"
        self.classifier = Attribute_classify(self.filename)"""

model = load_model('model.h5')

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.files['image']
    test_image = image.load_img(image, target_size=(224, 224,3))
    test_image = image.img_to_array(test_image)
    print(test_image)
    result = model.predict(test_image)
    result = np.where(result > 0.5, 1, 0)
    print(result)
    #test_image = np.expand_dims(test_image, axis = 0)
    #decodeImage(image, ClientApp().filename)
    #result = ClientApp().classifier.prediction()
    return jsonify(result)

if __name__ == "__main__":
    #clApp = ClientApp()
    #app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=5000)