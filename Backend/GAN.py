import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Fetch MNIST Dataset using the supplied Tensorflow Utility Function
#mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The size of the noise vector
NOISE_SIZE = 100
HIDDEN_SIZE = 128
IMAGE_SIZE = 28*28
N_DIGITS = 10

# The input vector of noise
Z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE])
Y = tf.placeholder(tf.float32, shape=[None, N_DIGITS])
# 1st layer's weights and bias
G_W1 = tf.get_variable('G_W1', shape=[NOISE_SIZE + N_DIGITS, HIDDEN_SIZE])
G_b1 = tf.get_variable('G_b1', shape=[HIDDEN_SIZE])

# 2nd layer's weights and bias
G_W2 = tf.get_variable('G_W2', shape=[HIDDEN_SIZE, IMAGE_SIZE])
G_b2 = tf.get_variable('G_b2', shape=[IMAGE_SIZE])

# The trainable generator variables
theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z, y):
    ''' The generator net.
    '''
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

# The input image
X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])

# 1st layer's weights and bias
D_W1 = tf.get_variable('D_W1', shape=[IMAGE_SIZE + N_DIGITS, HIDDEN_SIZE])
D_b1 = tf.get_variable('D_b1', shape=[HIDDEN_SIZE])

# 2nd layer's weights and bias
D_W2 = tf.get_variable('D_W2', shape=[HIDDEN_SIZE, 1])
D_b2 = tf.get_variable('D_b2', shape=[1])

# The trainable discriminator variables
theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x, y):
    '''The discriminator net.
    '''
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

# Image created by the generator
G_sample = generator(Z, Y)

# Descriminator's output for the real MNIST image
D_real, D_logit_real = discriminator(X, Y)
# Descriminator's output for the generated MNIST image
D_fake, D_logit_fake = discriminator(G_sample, Y)

# Descriminator wants high probability for the real image
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_real,
        labels=tf.ones_like(D_logit_real)))
# Descriminator also wants low probability for the generated image
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake,
        labels=tf.zeros_like(D_logit_fake)))
# We sum these to get our total descriminator loss
D_loss = D_loss_real + D_loss_fake

# Generator wants high probability for the generated image
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))

def one_hot_encode(x):
    result = []
    for val in x:
        result.append([1 if i == val else 0 for i in range(10)])
    return np.array(result)

def sample_Z(m, n):
    '''Returns a uniform sample of values between
    -1 and 1 of size [m, n].
    '''
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    '''Plots a grid of 16 generated images.
    '''
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

# The optimizer for each net
D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

BATCH_SIZE = 128

image_input = np.identity(N_DIGITS)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Image counter
    i = 0
    #sample_arr = []
    for it in range(1000000):
        # Save out image of 16 generated digits
        if it % 1000 == 0:
            samples = sess.run(G_sample,
                               feed_dict={
                                   Z: sample_Z(N_DIGITS, NOISE_SIZE),
                                   Y: image_input
                               })
            fig = plot(samples)
            #plt.show()
            fig.savefig('output/'+'output_'+str(int(it/1000))+'.png')
            plt.close()

        # Get a batch of real MNIST images
        #X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, X_batch, _, Y_batch = train_test_split(x_train, y_train, test_size=BATCH_SIZE)
        X_batch = np.array([i.flatten() for i in X_batch]) / 255.
        Y_batch = one_hot_encode(Y_batch)

        # Run our optimizers
        _, D_loss_curr = sess.run([D_optimizer, D_loss],
                                  feed_dict={
                                      X: X_batch,
                                      Z: sample_Z(BATCH_SIZE, NOISE_SIZE),
                                      Y: Y_batch
                                  })
        _, G_loss_curr = sess.run([G_optimizer, G_loss],
                                  feed_dict={
                                      Z: sample_Z(BATCH_SIZE, NOISE_SIZE),
                                      Y: Y_batch
                                  })

        # Report loss
        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()

    #plt.show()
