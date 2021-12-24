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
from typing import Tuple
import requests
from PIL import Image
import matplotlib.pyplot as plt
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
# data in a Dataset in this sample.

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# %%
#we saw earlier that each pixel of the image is 
# represented by an unsigned int. In machine learning, 
# we generally want the pixel values of our training 
# data to be floating-point numbers between 0 and 1, 
# so we convert them in the following way:
train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))
# %%
#we may have noticed that each value returned by the 
# Dataset is a tuple containing an image and a label. 
# We divide each value in the image by 255, and we keep 
# the label as is. Let's inspect the values of the same 
# image we inspected earlier, to see the difference.

train_dataset.as_numpy_iterator().next()[0]
# %%
#As expected, the pixel values are now floating-point numbers 
# between 0 and 1.

#Ntice that now thato we have a Dataset, 
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
train_dataset = train_dataset.batch(batch_size).shuffle(500)
test_dataset = test_dataset.batch(batch_size).shuffle(500)

# %%
#By specifying the batch size, we're telling the Dataset 
# that when we iterate over it, we want to receive not one, 
# but a batch of 64 items instead. If we print the length of 
# the first item returned by the iterator, we'll see that we 
# in fact get 64.
len(train_dataset.as_numpy_iterator().next()[0])

# %%
# Start NN architecture
#Our goal is to classify an input image into one of the 10 classes 
# of clothing, so we will define our neural network to take as 
# input a matrix of shape (28, 28) and output a vector of size 10, 
# where the index of the largest value in the output corresponds to
# the integer label for the class of clothing in the image.

# %%
#Because each image has 28 × 28 = 784 pixels, 
# we need 784 nodes in the input layer (one for each pixel value).
#  We decided to add one hidden layer with 20 nodes and a 
# ReLU (rectified linear unit) activation function. 
# We want the output of our network to be a vector of size 10, 
# therefore our output layer needs to have 10 nodes.

# Here's the Keras code that defines this neural network:
#%%
class NeuralNetwork(tf.keras.Model):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.sequence = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(20, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime
# %%
#The Flatten layer turns our input matrix of shape (28, 28) 
# into a vector of size 728. The Dense layers are also known as 
# "fully connected" or "linear" layers because they connect all 
# nodes from the previous layer with each of their own 
# nodes using a linear function. Notice that they specify "ReLU" 
# as the activation — that's because we want the results of the 
# linear mathematical operation to get passed as input to a 
# "Rectified Linear Unit" function, which adds non-linearity to 
# the calculations.

#It's important to have non-linear activation functions 
# (like the ReLU function) between linear layers, because
#  otherwise a sequence of linear layers would be mathematically 
# equivalent to just one layer. These activation functions 
# give our network more expressive power, allowing it to 
# approximate non-linear relationships between data.

#The Sequential class combines all the other layers. Lastly, 
# we define the call method, which supplies a tensor x as input 
# to the sequence of layers and produces the y_prime vector as 
# a result.

#We can print a description of our model using the summary method:
#%%
model = NeuralNetwork()
model.build((1, 28, 28))
model.summary()
# %%
#This is all the code needed to define our neural network. 
# Now that we have a neural network and some data, it's time
# to train the neural network using that data.
#%%
# Train & Test the NN
# visit https://docs.microsoft.com/en-us/learn/modules/intro-machine-learning-keras/4-train

from kintro import *
#%%
learning_rate = 0.1
batch_size = 64

model = NeuralNetwork()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate)
metrics = ['accuracy']
model.compile(optimizer, loss_fn, metrics)
# %%
#start over
# Now that we've configured our model with the parameters we need 
# for training, we can call fit to train the model. We specify the 
# number of epochs as 5, which means that we want to iterate over 
# the complete set of 60,000 training images five times while 
# training the neural network.

epochs = 5
print('\nFitting:')
model.fit(train_dataset, epochs=epochs)
# %%
# test model
print('\nEvaluating:')
(test_loss, test_accuracy) = model.evaluate(test_dataset)
print(f'\nTest accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')
# %%
#We've achieved pretty good test accuracy, considering that we used 
# such a simple network and only five epochs of training. 
# We're done with training and can now save the model.

model.save('outputs/model')
# %%
#Now that our neural network has appropriate values for its parameters, 
# we can use it to make a prediction.
#Assuming we achieved pretty good accuracy during the training and testing phases, we can now use the trained model for inference — in other words, to predict the
# classification of images that the network has never seen before.
#Making a prediction is easy — we simply call the model's predict method and pass one or more images. Let's start by geting 
# the image we'll use for prediction, and let's display it:
#%%
url = 'https://raw.githubusercontent.com/MicrosoftDocs/tensorflow-learning-path/main/intro-keras/predict-image.png'

with Image.open(requests.get(url, stream=True).raw) as image:
  X = np.asarray(image, dtype=np.float32).reshape((-1, 28, 28)) / 255.0

plt.figure()
plt.axis('off')
plt.imshow(X.squeeze(), cmap='gray')
plt.show()
# %%
#In the code below, we load the image, call predict to get its class index, 
# and map that index to the class name.

predicted_vector = model.predict(X)
predicted_index = np.argmax(predicted_vector)
predicted_name = labels_map[predicted_index]

print(f'Predicted class: {predicted_name}')
# %%
#Note that you can also get probabilities of the input image being of a certain class, in which case we need to normalize the output of our network using softmax 
# to get probabilities. Here are the predictions for our image:

probs = tf.nn.softmax(predicted_vector.reshape((-1,)))
for i,p in enumerate(probs):
    print(f'{labels_map[i]} -> {p:.3f}')
