from flask import Flask
app = Flask(__name__)

base_url = '/api/'

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route(base_url + 'image', methods=['POST'])
def imageProcess():
    print(request.get_json(), file=sys.stderr)
    new_class = (**request.get_json())
    ret_data = classify(new_class.image)
    return jsonify_return_data(ret_data)

def classify(image):
    pass

def jsonify_return_data(row):
    myrow = {
        "letter": row.letter,
        "success": row.success
    }
    return myrow

def main():
    app.run()

if if __name__ == "__main__":
    app.debug = False
    main()