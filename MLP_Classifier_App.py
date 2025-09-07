import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="MLP Classifier - Digits", layout="centered")
st.title("🔢 Day 25 — Multilayer Perceptron (MLP) Classifier")

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Sample Data Preview
st.subheader("📂 Sample Data Preview")
st.write("Each digit is represented as an 8x8 image (flattened into 64 features).")

# Display first 5 images with labels (side by side)
cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        col.image(digits.images[i], caption=f"Label: {y[i]}", width=80, clamp=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sidebar for hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")
hidden_layer_sizes = st.sidebar.slider("Hidden Layer Size", 10, 200, 100, 10)
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, 50)

# Train model
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, random_state=42)
mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)

# Show results
st.subheader("📊 Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🔎 Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_estimator(mlp, X_test, y_test, ax=ax, cmap="Blues")
st.pyplot(fig)

st.success("✅ MLP training complete — Neural Network classified handwritten digits!")
