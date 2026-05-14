import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Enable autolog
mlflow.sklearn.autolog()


# Load dataset
df = pd.read_csv(
    "titanic_preprocessing/processed.csv"
)

# Split feature dan target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MLflow tracking
with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Akurasi: {accuracy}")