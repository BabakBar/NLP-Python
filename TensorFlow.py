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
import numpy as np
import gzip
import pandas as pd
from tensorflow.keras import datasets
print("Tensorflow version: {}".format(tf.__version__))
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
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
print(len(train_images))

image_size = 28
num_train = 60000
num_test = 10000
#%%
import random
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(8, 8))
cols = 3
rows = 3
for i in range(1, cols * rows + 1):
  sample_idx = random.randint(0, len(train_images))
  image = train_images[sample_idx]
  label = train_labels[sample_idx]
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
#You saw earlier that each pixel of the image is 
# represented by an unsigned int. In machine learning, 
# we generally want the pixel values of our training 
# data to be floating-point numbers between 0 and 1, 
# so we convert them in the following way:

# %%
#You may have noticed that each value returned by the 
# Dataset is a tuple containing an image and a label. 
# We divide each value in the image by 255, and we keep 
# the label as is. Let's inspect the values of the same 
# image we inspected earlier, to see the difference.

# %%

# %%
#As expected, the pixel values are now floating-point numbers 
# between 0 and 1.

#Notice that now that we have a Dataset, 
# we can no longer index it the same way as a NumPy array. 
# Instead, we get an iterator by calling the as_numpy_iterator 
# method, and we advance it by calling its next method. 
# At this point, we have a tuple containing an image and 
# the corresponding label, so we can get the element at 
# index 0 to inspect the image.

#Finally, we tell the Dataset to give us batches of 
# data of size 64, and we shuffle the data:

# %%
batch_size = 64

# %%
#By specifying the batch size, we're telling the Dataset 
# that when we iterate over it, we want to receive not one, 
# but a batch of 64 items instead. If we print the length of 
# the first item returned by the iterator, we'll see that we 
# in fact get 64.


# %%

# %%
