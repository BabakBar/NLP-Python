# %%
msg = "Tensor with Mic"
print(msg)

# %% [markdown]
# Keras, is a higher-level and user-friendly API that is released as part of TensorFlow.
# Plans:
# load and prepare data to be used in machine learning.
# specify the architecture of a deep learning neural network.
# train a neural network.
# make a prediction using a neural network.

# %%
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import gzip
import numpy as np
# %% [markdown]
#the Fashion MNIST dataset. This dataset contains 70,000 grayscale 
# images of articles of clothing — 60,000 training and 10,000 for testing. 
# The images are square and contain 28 × 28 = 784 pixels, where each pixel is represented by a value between 0 and 255. 
# Each of these images is associated with a label, which is an integer 
# between 0 and 9 that classifies the article of clothing. 
# The following dictionary helps us understand the clothing categories corresponding to these integer labels:
#%%
labels_map = {
  0: 'T-Shirt',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle Boot',
}
#%%
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

def read_images(path: str, image_size: int, num_items: int) -> np.ndarray:
  with gzip.open(path, 'rb') as file:
    data = np.frombuffer(file.read(), np.uint8, offset=16)
    data = data.reshape(num_items, image_size, image_size)
  return data

def read_labels(path: str, num_items: int) -> np.ndarray:
  with gzip.open(path, 'rb') as file:
    data = np.frombuffer(file.read(num_items + 8), np.uint8, offset=8)
    data = data.astype(np.int64)
  return data

image_size = 28
num_train = 60000
num_test = 10000

training_images = read_images('data/FashionMNIST/raw/train-images-idx3-ubyte.gz', image_size, num_train)
test_images = read_images('data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz', image_size, num_test)
training_labels = read_labels('data/FashionMNIST/raw/train-labels-idx1-ubyte.gz', num_train)
test_labels = read_labels('data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz', num_test)
#%%
import random
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(7, 7))
cols = 3
rows = 3
for i in range(1, cols * rows + 1):
  sample_idx = random.randint(0, len(training_images))
  image = training_images[sample_idx]
  label = training_labels[sample_idx]
  figure.add_subplot(rows, cols, i)
  plt.title(labels_map[label])
  plt.axis('off')
  plt.imshow(image.squeeze(), cmap='gray')
plt.show()
# %%
#For such a small dataset, we could just use the NumPy 
# arrays given to us by Keras to train the neural network. 
# However, if we had a large dataset, 
# we would need to wrap it in a tf.data.Dataset instance, 
# which handles large data better by making it easy to keep 
# just a portion of it in memory. We've decided to wrap our 
# data in a Dataset in this sample, so you're prepared to 
# work with large data in the future.


# %%
train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
# %%
#You saw earlier that each pixel of the image is 
# represented by an unsigned int. In machine learning, 
# we generally want the pixel values of our training 
# data to be floating-point numbers between 0 and 1, 
# so we convert them in the following way: