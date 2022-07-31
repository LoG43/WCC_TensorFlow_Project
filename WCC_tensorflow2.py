# Module imports
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Version check
print("WCC Tensorflow Tutorial 2 Electic Boogaloo")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

# Dataset Upload 
train_data, validation_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)

# Training dataset sample
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)

# Represent data with pre-trained text embedder as first network layer
# Embedding refers to the connotation of sentences for our program (in this case determining whether the review is positive or negative)
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# Building the keras model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

# Adding necessary keras model settings
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

# Start keras model training
history = model.fit(train_data.shuffle(10000).batch(512),epochs=10,validation_data=validation_data.batch(512),verbose=1)

# Testing keras model
print("\nTesting")
results = model.evaluate(test_data.batch(512), verbose = 1)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))