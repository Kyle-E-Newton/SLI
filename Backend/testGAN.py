from __future__ import print_function, division

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as K
import numpy as np
import pandas as pd
import random

def make_model():
    model = K.models.Sequential()
    model.add(K.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(K.layers.MaxPooling2D((2, 2)))
    model.add(K.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(K.layers.MaxPooling2D((2, 2)))
    model.add(K.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dense(25, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

generator = tf.keras.models.load_model('generator.h5')


def generate_data(batch_size=1000, latent_dim=256):

    noise = np.random.normal(0, 1, (batch_size, latent_dim))            # input noise
    sampled_labels = np.random.randint(0, 24, (batch_size, 1))          # input labels

    return generator.predict([noise, sampled_labels]), sampled_labels


if __name__ == "__main__":
    
    models = [make_model() for _ in range(16)]    # list of classifiers

    for model in models:
        x_train, y_train = generate_data(1000)
        model.fit(x_train, y_train, epochs=random.randint(1,8))

    for i, model in enumerate(models):
        x_test, y_test = generate_data(batch_size=100)
        model.evaluate(x_test, y_test)
        model.save('model{}.h5'.format(i))
