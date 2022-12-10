

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Attribute_classify:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size =  (224, 224,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        result = np.where(result > 0.5, 1, 0)
        print(result)
        return(result)

        

