import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================
# NORMALIZATION FUNCTION
# =============================
def normalize(df):
    df = df.copy()

    for i in range(df.shape[0]):
        landmarks = df.iloc[i].values.astype(float)

        # Wrist is landmark 0
        wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]

        # Make wrist as origin
        landmarks[0::3] -= wrist_x
        landmarks[1::3] -= wrist_y
        landmarks[2::3] -= wrist_z

        # Scale
        max_val = np.max(np.abs(landmarks))
        if max_val != 0:
            landmarks = landmarks / max_val

        df.iloc[i] = landmarks

    return df

# =============================
# LOAD DATASET
# =============================
# Use the CSV created from data collection
df = pd.read_csv("hand_gesture_dataset.csv")

print("Dataset shape:", df.shape)
print("Labels:", df["label"].unique())

# =============================
# SPLIT FEATURES & LABELS
# =============================
X = df.drop(columns=["label"])
y = df["label"]

# =============================
# NORMALIZE FEATURES
# =============================
X = normalize(X)

# =============================
# TRAIN-TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# MODEL (ANTI-OVERFITTING)
# =============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# =============================
# EVALUATION
# =============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# =============================
# SAVE MODEL
# =============================
joblib.dump(model, "gesture_model.pkl")
print("✅ Model saved as gesture_model.pkl")
