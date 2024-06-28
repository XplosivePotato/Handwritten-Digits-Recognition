# Handwritten Digit Recognizer

Welcome to the Handwritten Digit Classifier! This innovative tool utilizes advanced neural networks to accurately interpret and classify handwritten digits from images. Whether you want to train a model from scratch or use a pre-trained one, this tool has got you covered.

## Introduction

Hello! You've arrived at the Handwritten Digit Classifier!

This program offers versatile features:

Train a fresh neural network model for recognizing digits.
Utilize a pre-trained model to recognize digits.
Analyze and predict custom digit images using the model.

## Getting Started

To start using this tool, follow these steps:

1.Download or clone this repository to your local system.
2.Ensure you have installed all necessary libraries (TensorFlow, OpenCV, NumPy, Matplotlib).
3.Execute the script handwritten_digits_recognition.py with Python.

## Usage

When you run the script, you’ll be prompted to either load an existing model or train a new one. Here’s how each option works:

Training a New Model:
- The script will load and prepare the MNIST dataset, a standard for digit recognition tasks.
- It will build and compile a neural network tailored for digit classification.
- The model will undergo training over several epochs to enhance accuracy.
- Post-training, the model’s performance will be assessed using a test dataset.
- The trained model will then be saved for future predictions.

Loading a Pre-trained Model:
- The script will directly load the saved model from a file, allowing you to bypass the training process.

# Custom Image Predictions
Once the model is either trained or loaded:

- You can input custom images of handwritten digits located in the digits directory. Ensure these images are 28x28 pixels in size for optimal prediction.
- The script will predict the digit and display the corresponding input image alongside its predicted label.

## Prerequisites

- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib

#  Tips for Best Results
Use clear, high-quality images of handwritten digits to ensure accurate recognition.
For troubleshooting or any inquiries, consult the documentation or reach out to the repository maintainer.
Enjoy exploring the world of digit recognition!

This project is a contribution by Archit Shukla
