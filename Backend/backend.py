from flask import Flask, request
from flask_cors import CORS
import Classifier
from PIL import Image

app = Flask(__name__)
CORS(app)

base_url = '/api/'
classifier = Classifier.Classifier

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route(base_url + 'image', methods=['POST'])
def imageProcess():
    file = request.files['file']
    ret_data = classify(file)
    return jsonify_return_data(ret_data)

def classify(image):
    prediction = classifier.classify(image)
    return prediction

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