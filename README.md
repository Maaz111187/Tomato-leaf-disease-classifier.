# Tomato Leaf Disease Classifier

A software-only computer vision project that classifies tomato leaf images into 10 categories using TensorFlow/Keras and a Streamlit web app.

## Project features

- Upload a tomato leaf image in the Streamlit app
- Get a predicted class and confidence score
- Train and evaluate the model in Google Colab
- View performance across all 10 classes

## Model results

On the project’s validation split, the model achieved 86.5% accuracy and a macro F1 score of 0.829. Results varied by class, and may be lower on real-world farm images.

## Run locally

1. Install Python 3.11.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```powershell
   pip install -r requirements.txt