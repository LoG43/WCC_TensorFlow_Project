# Module imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Version check
print("WCC Tensorflow Tutorial")
print("tf version:", tf.__version__)
print("np version:", np.__version__)
print("plt version: ", matplotlib.__version__)

# Dataset upload
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Image identifiers
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Unprocessed test image
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Downscaling pixel values so each pixel is between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

# 25 sample training images for network
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building keras model 
model = tf.keras.Sequential([
    # Data reformatting from 28x28 pixel array to 784 pixel array (2 -> 1 dimensions)  
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Adds 1st dense neural layer with 128 nodes 
    tf.keras.layers.Dense(128, activation='relu'),
    # Adds 2nd dense neural layer with 10 nodes (indicates possible outputs)
    tf.keras.layers.Dense(10)
])

# Adding necessary keras model settings
# Loss function measures how accurate the model is during training
# Optimizer determines keras model adjustments based on an algorithim reading loss function data
# Metrics is used to monitor the training and testing steps for the keras model
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# Start keras model training
model.fit(train_images, train_labels, epochs=5)

# Testing keras model
print("\nTesting: ")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('Test accuracy:', test_acc)

# Convert the model's linear output logits to probabilities, which should be easier to interpret.
# Logits are a functions that represent probability values from 0 to 1, and negative infinity to infinity.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# Creates array containing keras model predictions for each test image
predictions = probability_model.predict(test_images)

# Demonstration of keras model guesses with associated images
plt.figure()
plt.imshow(test_images[1], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
print('\nSAMPLE IMAGE GUESS 1: ', class_names[np.argmax(predictions_single[0])])

plt.figure()
plt.imshow(test_images[2], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
img = test_images[2]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
print('SAMPLE IMAGE GUESS 2: ', class_names[np.argmax(predictions_single[0])])

print("\nProgram Terminating\n")