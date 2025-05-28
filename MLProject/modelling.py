import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

# Mengaktifkan autologging untuk scikit-learn
mlflow.sklearn.autolog()

if __name__ == "__main__":
    # --- 1. Memuat Dataset Hasil Preprocessing ---
    # Asumsi titanic_preprocessed.csv ada di subfolder namadataset_preprocessing
    data_path = os.path.join('titanic_preprocessing', 'titanic_preprocessed.csv')
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset '{data_path}' berhasil dimuat. Bentuk: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{data_path}' tidak ditemukan. Pastikan sudah ada.")
        exit()

    # --- 2. Memisahkan Fitur dan Target ---
    # Target: Survived
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # --- 3. Membagi Data Training dan Testing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data training shape: {X_train.shape}, Data testing shape: {X_test.shape}")

    # --- 4. Melatih Model (Contoh: RandomForestClassifier) ---
    # MLflow autologging akan secara otomatis mencatat parameter, metrik, dan model
    with mlflow.start_run():
        # Inisialisasi dan latih model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model RandomForestClassifier berhasil dilatih.")

        # Prediksi pada data test
        y_pred = model.predict(X_test)

        # --- 5. Evaluasi Model (MLflow autologging akan mencatatnya) ---
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Note: Karena autologging aktif, metrik ini seharusnya sudah dicatat secara otomatis.
        # Namun, kita bisa log manual juga jika mau, atau untuk metrik tambahan.
        # mlflow.log_metric("custom_accuracy", accuracy) # Contoh log manual

        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print("MLflow autologging telah mencatat parameter, metrik, dan model.")

    print("\nEksperimen MLflow selesai. Jalankan 'mlflow ui' di terminal untuk melihat hasilnya.")
