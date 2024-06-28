import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Hey there! Welcome to the totally original Handwritten Digits Recognition Tool v2.0 (old model used 3 Epochs , updated it again to use 5 epochs as accuracy was a bit off)")

# Decide whether to load an existing model or train a fresh one
create_new_model = False

if create_new_model:
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the dataset
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Build a neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Loss: {loss}")
    print(f"Model Accuracy: {accuracy}")

    # Save the model
    model.save('handwritten_digits.model')
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model('handwritten_digits.model')

# Predict custom images
image_index = 1
while os.path.isfile(f'digits/digit{image_index}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_index}.png', cv2.IMREAD_GRAYSCALE)
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is most likely a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_index += 1
    except Exception as e:
        print(f"Oops! Couldn't read image. Moving on... {str(e)}")
        image_index += 1
