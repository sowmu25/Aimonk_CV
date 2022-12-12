from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import save_model,load_model
from tensorflow.keras.preprocessing import image
import numpy as np



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)



model = load_model('model.h5')

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    print("URL HITTED")
    if request.method == 'POST':
        upload_file = request.files['fileup']
        filename = upload_file.filename
        print(filename)
        upload_img_path = os.path.join(os.getcwd(),filename)
        upload_file.save(upload_img_path)
        test_image = image.load_img(filename, target_size=(224, 224,3))
        test_image = image.img_to_array(test_image)
        print(test_image)
        test_image = test_image/255.
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)[0]
        result = np.where(result >= 0.5, 1, 0)
        print(result)
      
        return render_template("index.html",predicted_result="Attribute prediction of {} ".format(filename) +"  is {}".format(result))

if __name__ == "__main__":
   
    app.run(host='0.0.0.0', port=5050)