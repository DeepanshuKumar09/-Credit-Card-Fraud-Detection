
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def create_autoencoder(input_shape):
  model = models.Sequential()
  # Encoders Layers
  model.add(layers.InputLayer(input_shape=(input_shape,)))
  model.add(layers.Dense(64, activation="relu"))
  model.add(layers.Dense(32, activation="relu"))
  model.add(layers.Dense(16, activation="relu"))

  # Decoders layers
  model.add(layers.Dense(32, activation="relu"))
  model.add(layers.Dense(64, activation="relu"))
  model.add(layers.Dense(input_shape, activation="tanh"))
  return model

def main():
    st.title("Credit Card Fraud Detection using Autoencoder")

    # 1. Data Loading
    st.header("1. Data Loading")
    @st.cache_data
    def load_data():
        df = pd.read_csv("Detection/creditcard.csv")
        return df

    df = load_data()
    st.write("Dataset loaded successfully. Displaying first 5 rows:")
    st.dataframe(df.head())

    # 2. Data Preprocessing
    st.header("2. Data Preprocessing")
    with st.spinner('Performing data preprocessing...'):
        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        rows_after_dedup = df.shape[0]
        st.info(f"Dropped {initial_rows - rows_after_dedup} duplicate rows.")

        if 'Time' in df.columns:
            df = df.drop(["Time"], axis=1)
            st.info("Dropped 'Time' column.")

        scaler_amount = StandardScaler()
        df["Amount"] = scaler_amount.fit_transform(df[["Amount"]])
        st.info("Scaled 'Amount' column.")

        df["Class"] = df["Class"].astype(str)
        df = pd.get_dummies(df, columns=["Class"])
        st.info("One-hot encoded 'Class' column.")

        columns_to_scale = [col for col in df.columns if col not in ['Class_0', 'Class_1']]
        scaler_features = StandardScaler()
        df[columns_to_scale] = scaler_features.fit_transform(df[columns_to_scale])
        st.info("Scaled all other numerical features.")

    st.success("Data preprocessing complete!")
    st.write("DataFrame after preprocessing (first 5 rows):")
    st.dataframe(df.head())

    # 3. Data Splitting
    st.header("3. Data Splitting")
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train_data.drop(["Class_0","Class_1"],axis=1).values
    y_train = train_data[["Class_0","Class_1"]].values

    X_test = test_data.drop(["Class_0","Class_1"],axis=1).values
    y_test = test_data[["Class_0","Class_1"]].values
    st.success("Dataset split into training and testing sets.")
    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Testing data shape: {X_test.shape}")

    # 4. Model Definition
    st.header("4. Autoencoder Model Definition")
    input_shape = X_train.shape[1]
    autoencoder = create_autoencoder(input_shape)
    autoencoder.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    st.write("Autoencoder model defined and compiled.")
    st.text(autoencoder.summary())

    # 5. Model Training
    st.header("5. Model Training")
    if st.button('Train Autoencoder Model'):
        with st.spinner('Training Autoencoder model (this may take a while)...'):
            history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, validation_data=(X_test, X_test), shuffle=False, verbose=0)
        st.success("Autoencoder model training complete!")

        # 6. Display Training History Plots
        st.subheader("Training History")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].set_title('Training Loss and Accuracy')
        axes[0].legend()

        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Validation Loss and Accuracy')
        axes[1].legend()
        st.pyplot(fig)

        # 7. Anomaly Detection Logic
        st.header("6. Anomaly Detection and Evaluation")
        predictions = autoencoder.predict(X_test, verbose=0)
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)

        threshold = st.slider('Anomaly Threshold', min_value=0.1, max_value=2.0, value=0.6, step=0.05)
        anomalies = mse > threshold

        y_true = np.argmax(y_test, axis=1) # Convert one-hot to single label
        y_pred = anomalies.astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        st.write("### Evaluation Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label='Test Accuracy', value=f'{accuracy:.4f}')
        with col2:
            st.metric(label='Cohen\'s Kappa', value=f'{kappa:.4f}')

        # 8. Plot Anomaly Detection Visualization
        st.subheader("Anomaly Detection Visualization (MSE)")
        fig_mse = plt.figure(figsize=(10, 6))
        plt.scatter(range(len(mse)), mse, c=anomalies, cmap='coolwarm', s=4)
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.xlabel("Data Points")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Anomaly Detection based on Reconstruction Error")
        plt.legend()
        st.pyplot(fig_mse)

        # 9. Plot Data Reconstruction Visualizations
        st.subheader("Data Reconstruction Visualizations")

        # Using a subset for clearer visualization if X_test is very large
        sample_indices = np.random.choice(X_test.shape[0], min(X_test.shape[0], 500), replace=False)
        X_test_sample = X_test[sample_indices]
        predictions_sample = predictions[sample_indices]
        mse_sample = mse[sample_indices]
        anomalies_sample = anomalies[sample_indices]

        fig_reconstruction = plt.figure(figsize=(15, 15))
        plt.subplot(3, 1, 1)
        plt.imshow(X_test_sample.T, aspect='auto', cmap='viridis')
        plt.title('Input Data Sample')
        plt.xlabel('Data Points')
        plt.ylabel('Features')

        plt.subplot(3, 1, 2)
        plt.imshow(predictions_sample.T, aspect='auto', cmap='viridis')
        plt.title('Reconstruction Sample')
        plt.xlabel('Data Points')
        plt.ylabel('Features')

        plt.subplot(3, 1, 3)
        plt.plot(mse_sample, label='Reconstruction Error')
        plt.scatter(np.where(anomalies_sample)[0], mse_sample[anomalies_sample], color='red', label='Anomalies')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.title('Reconstruction Error and Anomalies Sample')
        plt.xlabel('Data Points')
        plt.ylabel('Mean Squared Error')
        plt.legend()

        plt.tight_layout()
        st.pyplot(fig_reconstruction)

if __name__ == '__main__':
    main()

