from __future__ import print_function, division

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

generator = tf.keras.load_model('generator')

model1 = tf.keras.load_model('model1.h5')
model2 = tf.keras.load_model('model1.h5')
model3 = tf.keras.load_model('model1.h5')
model4 = tf.keras.load_model('model1.h5')
model5 = tf.keras.load_model('model1.h5')
model6 = tf.keras.load_model('model1.h5')
model7 = tf.keras.load_model('model1.h5')
model8 = tf.keras.load_model('model1.h5')
model9 = tf.keras.load_model('model1.h5')
model10 = tf.keras.load_model('model1.h5')
model11 = tf.keras.load_model('model1.h5')
model12 = tf.keras.load_model('model1.h5')
model13 = tf.keras.load_model('model1.h5')
model14 = tf.keras.load_model('model1.h5')
model15 = tf.keras.load_model('model1.h5')
model16 = tf.keras.load_model('model1.h5')

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16]
class_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def generate_data(batch_size=1000, latent_dim=256):

    noise = np.random.normal(0, 1, (batch_size, latent_dim))            # input noise
    sampled_labels = np.random.randint(0, 24, (batch_size, 1))          # input labels

    return generator.predict([noise, sampled_labels]), sampled_labels

def make_feature_extractor():
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_layer = K.layers.GlobalAveragePooling2D()(feature_extractor.output)
    convolutional_base = K.models.Model(feature_extractor.input, last_layer)
    return convolutional_base

def test_models():
    for model in models:
        x_train, y_train = generate_data()
        model.fit(x_train, y_train)

    for model in models:
        x_test, y_test = generate_data(batch_size=100)
        model.evaluate(x_test, y_test)

def classify():
    feature_extractor = make_feature_extractor()
    return class_list[np.argmax(model.predict(feature_extractor.predict(np.expand_dims(np.asarray(Image.open("./img.jpg").resize((224,224))) / 255., axis=0))))] # absolute beauty #

def classify_ensemble():
    predictions = list()
    for model in models:
        predictions.append(classify(model))

    common, num_common = Counter(predictions).most_common(1)[0]
    return most

if __name__ == "__main__":
    print(classify_ensemble)