# 🧠 Handwritten Digit Recognition with CNN | Week 8 - Let's Talk About Data

This is **Week 8** of the **Weekly Data Science Projects** series on the [Let's Talk About Data](https://www.youtube.com/@letstalkaboutdata) channel.

In this project, we build a **Convolutional Neural Network (CNN)** using Python and TensorFlow/Keras to recognize **handwritten digits** from the **MNIST dataset**.

---

## 📌 Project Overview

- 🎯 **Goal:** Train a deep learning model to classify digits (0–9) from grayscale images.
- 📂 **Dataset:** [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) – 28x28 grayscale images of handwritten digits.
- 🧰 **Tools Used:** Python, TensorFlow, Keras, Matplotlib
- 🧠 **Concepts Covered:**
  - CNN Architecture (Conv2D, MaxPooling2D, Flatten, Dense)
  - Image preprocessing & normalization
  - Model evaluation and visualization
  - Prediction on new data

---

## 📽️ Watch the Full Tutorial on YouTube

📺 **YouTube:** [Handwritten Digit Recognition with CNN | Project 8 – Data Science Series](https://www.youtube.com/@letstalkaboutdata-ltad)

---

## 🧱 Tech Stack

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn (for visualizations)  
- Scikit-learn (confusion matrix)

---

## 📊 Model Architecture

Input Layer (28x28 grayscale image)
→ Conv2D Layer
→ MaxPooling Layer
→ Conv2D Layer
→ MaxPooling Layer
→ Flatten
→ Dense (128 units, ReLU)
→ Output Dense Layer (10 units, Softmax)


---

## 🚀 How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/letstalkaboutdata.com/cnn-project
   cd cnn-project

2. **Install dependencies**
  pip install -r requirements.txt 
   
3. **Run the notebook or script**
  jupyter notebook 'Handwritten Digit Recognition using CNN.ipynb'


