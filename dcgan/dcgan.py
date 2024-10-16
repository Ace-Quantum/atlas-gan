#!/usr/bin/env python3
"""This contains the code for creating and training
a Deep Convolutional Generative Adversarial Network
It uses the following links as inspiration:
https://github.com/luisguiserrano/gans/blob/master/GANs_in_Slanted_Land.ipynb
https://www.tensorflow.org/tutorials/generative/dcgan"""


# I'll be honest, I only know what half of these are for (10/15)
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

# Loading the dataset
#   See, I knew that I could just load them in program
#   but my brain decided that I needed them directly in the root of my file
#   so I guess we'll really see what we need
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Prepping the dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 #normalizing

# Arbitrary buffer and batch size?
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# batch & shuffle
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='ReLU')) #I'm trying a different activation for funsies
    assert model.output_shape == (None, 28, 28, 1)

    return model
