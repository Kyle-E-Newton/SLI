import tensorflow.keras as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from PIL import Image
from collections import Counter

import multiprocessing

class_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
print(class_list)

model1 = K.models.load_model('SLI_Model1.h5')
model2 = K.models.load_model('SLI_Model2.h5')
model3 = K.models.load_model('SLI_Model3.h5')
model4 = K.models.load_model('SLI_Model4.h5')
model5 = K.models.load_model('SLI_Model3.h5')

models = [model1, model2, model3, model4, model5]

def make_feature_extractor():
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_layer = K.layers.GlobalAveragePooling2D()(feature_extractor.output)
    convolutional_base = K.models.Model(feature_extractor.input, last_layer)
    return convolutional_base

feature_extractor = make_feature_extractor()

def classify(model):
    retval = class_list[np.argmax(model.predict(feature_extractor.predict(np.expand_dims(np.asarray(Image.open("./img.jpg").resize((224,224))) / 255., axis=0))))] # absolute beauty #
    return retval

def classify_ensemble():
    predictions = list()
    for model in models:
        predictions.append(classify(model))

    most_common, num_most_common = Counter(predictions).most_common(1)[0]
    return most_common

if __name__ == "__main__":
    print(classify_ensemble())

    
    
    # processes = list()
    # for model in models:
    #     x = multiprocessing.Process(target=classify, args=(model,))
    #     processes.append(x)
    #     x.start()
    
    # for process in processes:
    #     process.join()

    # print("Classification")