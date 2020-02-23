from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow.keras as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from PIL import Image
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)



base_url = '/api/'
model = K.models.load_model('SLI_Model2.h5')

class_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
print(class_list)

def make_feature_extractor():
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_layer = K.layers.GlobalAveragePooling2D()(feature_extractor.output)
    convolutional_base = K.models.Model(feature_extractor.input, last_layer)
    return convolutional_base

feature_extractor = make_feature_extractor()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route(base_url + 'image', methods=['POST'])
def classify():             # i cry ery time
    d = request.get_json()
    decodeTxt = base64.b64decode(d["data"].split(',', 1)[1])
    with open('./img.jpg', 'wb') as f:
        f.write(decodeTxt)
    
    retval = class_list[np.argmax(model.predict(feature_extractor.predict(np.expand_dims(np.asarray(Image.open("./img.jpg").resize((224,224))) / 255., axis=0))))] # absolute beauty #
    return jsonify({"letter": str(retval), "status": 1})


def jsonify_return_data(row):
    myrow = {
        "letter": row.letter
    }
    return myrow

def main():
    app.run()

if __name__ == "__main__":
    app.debug = False
    main()