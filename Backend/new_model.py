import tensorflow.keras as K
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

batch_size = 32

path = 'sign-language-mnist/'
train_dir = path + 'sign_mnist_train/sign_mnist_train.csv'
test_dir = path + 'sign_mnist_test/sign_mnist_test.csv'

train_data = pd.read_csv(train_dir).to_numpy()
test_data = pd.read_csv(test_dir).to_numpy()
train_labels, test_labels = train_data[:,0], test_data[:,0]
train_data, test_data = train_data[:,1:] / 255., test_data[:,1:] / 255.

train_data = np.expand_dims([i.reshape(28,28) for i in train_data], axis=-1)
test_data = np.expand_dims([i.reshape(28,28) for i in test_data], axis=-1)

#plt.imshow(data[0].reshape(28,28))
#plt.show()

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

model = make_model()
model.fit(train_data, train_labels, epochs=2)
model.evaluate(test_data, test_labels)