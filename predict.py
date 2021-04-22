import numpy as np
from itertools import chain
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import os
import cv2
import tensorflow as tf
from tensorflow.python.util import deprecation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class LogoClassification:
    def __init__(self):
        self.class_names = ['Goggles', 'Hat', 'Jacket', 'Shirt', 'Shoes', 'Shorts', 'T-Shirt', 'Trouser', 'Wallet', 'Watch']
        self.model = load_model("models/fashion.h5")

    def getPrediction(self, img):
        img = cv2.imread(img)
        dim = (224, 224)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        print(preds)
        preds_unlist = list(chain(*preds))
        print(preds_unlist)
        preds_int = [int((round(i, 2))) for i in preds_unlist]
        print(preds_int)
        final_pred = dict(zip(self.class_names, preds_int))
        print(100 * '-')
        print(final_pred)
        return final_pred
