from __future__ import print_function, division

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from PIL import Image
from collections import Counter

generator = tf.keras.models.load_model('generator.h5')

model0 = tf.keras.models.load_model('model0.h5')
model1 = tf.keras.models.load_model('model1.h5')
model2 = tf.keras.models.load_model('model2.h5')
model3 = tf.keras.models.load_model('model3.h5')
model4 = tf.keras.models.load_model('model4.h5')
model5 = tf.keras.models.load_model('model5.h5')
model6 = tf.keras.models.load_model('model6.h5')
model7 = tf.keras.models.load_model('model7.h5')
model8 = tf.keras.models.load_model('model8.h5')
model9 = tf.keras.models.load_model('model9.h5')
model10 = tf.keras.models.load_model('model10.h5')
model11 = tf.keras.models.load_model('model11.h5')
model12 = tf.keras.models.load_model('model12.h5')
model13 = tf.keras.models.load_model('model13.h5')
model14 = tf.keras.models.load_model('model14.h5')
model15 = tf.keras.models.load_model('model15.h5')

models = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]


def generate_data(batch_size=1000, latent_dim=256):

    noise = np.random.normal(0, 1, (batch_size, latent_dim))            # input noise
    sampled_labels = np.random.randint(0, 24, (batch_size, 1))          # input labels

    return generator.predict([noise, sampled_labels]), sampled_labels

def test_models():
    for model in models:
        x_train, y_train = generate_data()
        model.fit(x_train, y_train)

    for model in models:
        x_test, y_test = generate_data(batch_size=100)
        model.evaluate(x_test, y_test)

def test_ensemble():
    for model in models:
        x_train, y_train = generate_data()
        model.fit(x_train, y_train)
        
def classify_ensemble():
    predictions = list()
    images = generate_data(100)
    for model in models:
        model_predictions = model.predict(images[0])
        model_predictions = [np.argmax(i) for i in model_predictions]
        predictions.append(model_predictions)

    preds = list()
    final = list()
    for i in range(0, len(predictions[0])):
        for j in range(0, len(predictions)):
            preds.append(predictions[j][i])
        most, num = Counter(preds).most_common(1)[0]
        final.append(most)
        preds = list()
    real = np.array(images[1]).ravel()

    return real, final

if __name__ == "__main__":
    real, pred = classify_ensemble()
    correct = 0
    incorrect = 0
    for i in range(0, len(pred)):
        if real[i] == pred[i]:
            correct += 1
        else:
            incorrect += 1
        
    print("Accuracy: ", (correct / (incorrect+correct)))