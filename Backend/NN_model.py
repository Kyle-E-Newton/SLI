import tensorflow.keras as K
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

batch_size = 32

path = 'asl-alphabet/'

train_dir = path + 'train/'
test_dir = path + 'test/'

def make_feature_extractor():
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_layer = K.layers.GlobalAveragePooling2D()(feature_extractor.output)
    convolutional_base = K.models.Model(feature_extractor.input, last_layer)
    return convolutional_base

def make_model():
    model = K.models.Sequential()
    model.add(K.layers.Dense(512, activation='relu', input_shape=(1280,)))
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dense(28, activation='softmax'))
    model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# create data generator
datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, brightness_range=(0.5, 1.5))
conv_base = make_feature_extractor()

# returns the output of the last layer of the MobileNetV2 convolutional neural network
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 1280), dtype=np.float32)
    labels = np.zeros(shape=(sample_count), dtype=np.float32)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=1,
        class_mode='sparse')
    i = 0
    for inputs_batch, labels_batch in generator:
        print('progress:', round(i/(93064.0/batch_size), 2))
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

data_features, data_labels = extract_features(train_dir, 93064)
data_features = np.reshape(data_features, (93064, 1280))

x_train, x_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.1, random_state=42)

if __name__ == "__main__":
    model = make_model()
    model.fit(x_train, y_train, epochs=80, batch_size=256, shuffle=True)
    print('Evaluation:')
    model.evaluate(x_test, y_test)
    model.save('SLI_Model3.h5')
