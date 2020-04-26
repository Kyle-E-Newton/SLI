from __future__ import print_function, division

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

classifier = tf.keras.load_model('SLI_Model.h5')
generator = tf.keras.load_model('generator')


def generate_data(batch_size=1000, latent_dim=256):

    noise = np.random.normal(0, 1, (batch_size, latent_dim))            # input noise
    sampled_labels = np.random.randint(0, 24, (batch_size, 1))          # input labels

    return generator.predict([noise, sampled_labels]), sampled_labels


if __name__ == "__main__":
    
    models = [classifier for _ in range(10)]    # list of classifiers

    for model in models:
        x_train, y_train = generate_data()
        model.fit(x_train, y_train)

    for model in models:
        x_test, y_tset = generate_data(batch_size=100)
        model.evaluate(x_train, y_train)
