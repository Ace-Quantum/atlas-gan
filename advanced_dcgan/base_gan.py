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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import pandas as pd
# import wandb
# from wandb.keras import WandCallback

from IPython import display

# /root/atlas-gan/Butterflies_Moths
# I need more monitors for my work.
# Something to consider for a work from home set up

# loading up and preprocessing the data
df = pd.read_csv('/root/atlas-gan/Butterflies_Moths/butterflies and moths.csv')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img

class ButterflyMothDataset(tf.data.Dataset):
    def __init__(self, df, base_dir, batch_size=32):

        self.df = df
        self.base_dir = base_dir
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
        # Construct full file path
        file_path = os.path.join(self.base_dir, row['filepaths'])
        
        # Load image
        img = load_image(file_path)
        
        # Convert label to categorical
        label = tf.one_hot(row['class_id'], num_classes=100)
        
        return img, label
    
    def __iter__(self):
        for _ in range(len(self)):
            yield next(iter(self))
    
    def __call__(self):

        return tf.data.Dataset.from_generator(
            lambda: self.__iter__(),
            output_types=(tf.float32, tf.int64),
            output_shapes=((224, 224, 3), (100,))
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

# Load CSV file
df = pd.read_csv('/Butterflies_Moths/butterflies and moths.csv')

print(df.columns)

# base_dir = '/root/atlas-gan/Butterflies_Moths'

# # Define base directories for train, validation, and test datasets
train_base_dir = '../Butterflies_Moths/train'
valid_base_dir = '../Butterflies_Moths/valid'
test_base_dir = '../Butterflies_Moths/train'

# Create dataset instances
# The problems start here
train_dataset = ButterflyMothDataset(df[df['data set'] == 'train'], train_base_dir)
valid_dataset = ButterflyMothDataset(df[df['data set'] == 'valid'], valid_base_dir)
test_dataset = ButterflyMothDataset(df[df['data set'] == 'test'], test_base_dir)

# Create ImageDataGenerator for data augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=15,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True)

# Apply data augmentation to training dataset
train_dataset = train_datagen.flow_from_dataframe(
    dataframe=df[df['data_set'] == 'train'],
    directory=train_base_dir,
    x_col='filepaths',
    y_col='class_id',
    target_size=(224, 224),
    batch_size=train_dataset.batch_size,
    class_mode='categorical'
)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(valid_dataset)}")
print(f"Test set size: {len(test_dataset)}")


# creation of the model
# Subject to change
# I'm not sure to what yet
def make_generator_model():
    model = tf.keras.Sequential()
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

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) #I'm trying a different activation for funsies
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_gnerate = 16

seed = tf.random.normal([num_examples_to_gnerate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Maybe make a way to log generated images

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # include wandb log here

        print('time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

train(train_dataset, EPOCHS)
