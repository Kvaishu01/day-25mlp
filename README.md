# Day 25 — Multilayer Perceptron (MLP) Classifier

This project demonstrates a simple **Neural Network (MLP)** for classifying handwritten digits using the **Digits dataset** from scikit-learn.  
Built with **Streamlit** for an interactive demo. 🚀

---

## 📊 Algorithm
- **Multilayer Perceptron (MLP)**:  
  - Input layer: 64 features (8x8 image pixels)  
  - Hidden layers: (64, 32) neurons  
  - Output layer: 10 classes (digits 0–9)  

MLP is a **feed-forward neural network** trained with backpropagation.

---

## ⚙️ Features
- Displays dataset samples
- Trains an MLP classifier
- Shows accuracy, classification report, and confusion matrix
- Interactive demo for digit prediction

---

## 🚀 How to Run
```bash
pip install streamlit scikit-learn matplotlib
streamlit run MLP_Classifier_App.py
