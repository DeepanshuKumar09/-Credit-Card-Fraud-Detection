<div align="center">

# 💳 Credit Card Fraud Detection App

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)

**A deep learning-powered tool to identify fraudulent credit card transactions using an Autoencoder.**

[View Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📖 Project Overview

Credit card fraud detection is crucial for financial institutions to prevent monetary losses and maintain customer trust. This project implements an **Autoencoder**, a type of unsupervised neural network, to detect anomalous credit card transactions. The Autoencoder learns to reconstruct \"normal\" transaction patterns from the input data. Transactions that result in high reconstruction errors are flagged as potential anomalies (fraud).

The Streamlit application provides an interactive interface for users to upload transaction data, visualize the model's performance, and observe detected anomalies in real-time. It leverages pre-trained models and scalers for efficient inference.

---

## 🚀 Key Components

| Component | Description |
| :--- | :--- |
| **🧠 Autoencoder Model** | A pre-trained Keras sequential model (`autoencoder_model.keras`) designed to reconstruct normal transaction data. |
| **💻 Streamlit UI** | An interactive web dashboard (`streamlit_app.py`) for uploading data, performing anomaly detection, and visualizing results. |

---

## 🛠️ Features & Inputs

The Autoencoder model analyzes a set of **30 numerical features** derived from credit card transactions to identify anomalies. These features primarily consist of anonymized principal components (`V1` through `V28`), `Time`, and `Amount`.

<div align="center">

| Feature Category | Features |
| :--- | :--- |
| **Transaction Details** | `Time`, `Amount` |
| **Anonymized Features** | `V1`, `V2`, `V3`, `V4`, `V5`, `V6`, `V7`, `V8`, `V9`, `V10`, `V11`, `V12`, `V13`, `V14`, `V15`, `V16`, `V17`, `V18`, `V19`, `V20`, `V21`, `V22`, `V23`, `V24`, `V25`, `V26`, `V27`, `V28` |

</div>
