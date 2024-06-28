# Handwritten Digit Recognizer

Welcome to the Handwritten Digit Recognizer tool! This tool is designed to recognize handwritten digits using machine learning techniques. It leverages neural networks to accurately classify and identify digits from input images.

## Introduction

Hey there! Welcome to the Handwritten Digit Recognizer v2.0!

This tool offers the following functionalities:
- Train a new model for recognizing handwritten digits.
- Load an existing model for digit recognition.
- Predict custom images containing handwritten digits.

## Getting Started

To use this tool, follow these simple steps:

1. Clone this repository to your local machine.
2. Make sure you have the required dependencies installed (TensorFlow, OpenCV, NumPy, and Matplotlib).
3. Run the script `handwritten_digits_recognition.py` using Python.

## Usage

Upon running the script, you'll be prompted to choose whether to load an existing model or train a new one. Follow the on-screen instructions to proceed.

If you choose to train a new model:
- The script will load the MNIST dataset and normalize it.
- It will then build a neural network model and compile it.
- The model will be trained on the dataset for a specified number of epochs.
- After training, the model will be evaluated on a test dataset to measure its performance.
- Finally, the trained model will be saved for future use.

If you choose to load an existing model:
- The script will load the pre-trained model from the specified file.

## Once the model is loaded or trained, you can provide custom images containing handwritten digits for prediction(PUT YOUR CUSTOM IMAGES IN THE FOLDER NAMED "digits"(REMEMBER TO PROVIDE 28 x 28 PIXEL IMAGES). The script will display the predicted digit along with the input image.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib

## Notes

- Provide clear and well-defined images of handwritten digits for accurate predictions.
- For any issues or errors, refer to the documentation or contact the repository owner for assistance.

Happy digit recognition!

This project was made by Saumya Singh for the Code clause internship.
