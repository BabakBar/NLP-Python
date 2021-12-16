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