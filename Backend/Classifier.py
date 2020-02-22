import tensorflow.keras as K

class Classifier:

    def __init__(self):
        self.model = K.models.load_model('SLI_Model.h5')

    def classify(self, image):
        return self.model.predict(image)
