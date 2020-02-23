from flask import Flask
import Classifier
from PIL import Image
app = Flask(__name__)

base_url = '/api/'
Classifier = Classifier.Classifier

UPLOAD_FOLDER = 'uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route(base_url + 'image', methods=['POST'])
def imageProcess():
    file = request.files['file']
    ret_data = classify(file)
    return jsonify_return_data(ret_data)

def classify(image):
    prediction = Classifier.classify(image)
    return prediction

def jsonify_return_data(row):
    myrow = {
        "letter": row.letter
    }
    return myrow

def main():
    app.run()

if if __name__ == "__main__":
    app.debug = False
    main()