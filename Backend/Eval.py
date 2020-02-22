import tensorflow.keras as K
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import os

batch_size = 1

path = 'asl-alphabet/'

train_dir = path + 'train/'
test_dir = path + 'test/'

dir_list = os.listdir(train_dir)
print(dir_list)
test_images = np.asarray([np.array(Image.open(train_dir + i + '/' + i + '1.jpg').resize((224,224))).astype('float32') for i in dir_list])
test_images /= 255.
#print(test_images)

def make_feature_extractor():
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_layer = K.layers.GlobalAveragePooling2D()(feature_extractor.output)
    convolutional_base = K.models.Model(feature_extractor.input, last_layer)
    return convolutional_base


conv_base = make_feature_extractor()
model = K.models.load_model('SLI_Model2.h5')
print(test_images.shape)
conv_predictions = conv_base.predict(test_images)
predictions = model.predict(conv_predictions)

n = 28  # how many digits we will display
plt.figure(figsize=(18, 10))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(4, 10, i + 1, title=str(dir_list[np.argmax(predictions[i])]))
    plt.imshow(test_images[i])
    plt.gray()
    ax.get_yaxis().set_visible(False)
plt.show()