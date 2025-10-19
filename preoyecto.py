# GAN simple para generar imágenes (por ejemplo, carros o tralaleros)
# Guarda tus imágenes reales en una carpeta llamada "dataset/"

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from PIL import Image

def load_images(path, size=(128,128)):
    import glob
    from PIL import Image
    import numpy as np
    import os

    images = []
    formats = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.jfif', '*.webp']
    for ext in formats:
        for img_path in glob.glob(os.path.join(path, ext)):
            try:
                img = Image.open(img_path).convert('RGB').resize(size)
                img = np.array(img) / 255.0
                images.append(img)
            except Exception as e:
                print(f"Error al cargar {img_path}: {e}")

    print(f"Total de imágenes cargadas: {len(images)}")
    return np.array(images, dtype=np.float32)


dataset_pathe = 'Mazda-2000' 
images = load_images(dataset_pathe)
print(f"Total de imágenes cargadas: {len(images)}")

BUFFER_SIZE = len(images)
BATCH_SIZE = 16 
train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(16*16*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((16, 16, 256)),
        layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[128,128,3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

EPOCHS = 100
noise_dim = 100
num_examples = 16
seed = tf.random.normal([num_examples, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        img = (predictions[i] + 1) / 2
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(f"generated_epoch_{epoch:03d}.png")
    plt.close()

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Gen loss: {g_loss.numpy():.4f}, Disc loss: {d_loss.numpy():.4f}")
            generate_and_save_images(generator, epoch+1, seed)

train(train_dataset, EPOCHS)

print("Entrenamiento completado. Se generaron imágenes en la carpeta del proyecto.")
