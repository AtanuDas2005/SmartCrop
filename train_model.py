
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("crop_dataset_smartcrop.csv")

print("Dataset Loaded Successfully")
print("Total Records:", len(data))


# ==========================================
# 2. ENCODE CATEGORICAL FEATURES
# ==========================================

soil_map = {
    "loamy": 0,
    "sandy": 1,
    "clay": 2,
    "black": 3
}

season_map = {
    "summer": 0,
    "winter": 1,
    "monsoon": 2
}

data["soil_type"] = data["soil_type"].str.lower().map(soil_map)

data["season"] = data["season"].str.lower().map(season_map)


# ==========================================
# 3. CLEAN DATA
# ==========================================

data["temperature"] = pd.to_numeric(
    data["temperature"], errors="coerce"
)

data["rainfall"] = pd.to_numeric(
    data["rainfall"], errors="coerce"
)

data = data.dropna(
    subset=[
        "soil_type",
        "temperature",
        "rainfall",
        "season",
        "label"
    ]
)

# Temperature constraint: 0 to 50 degree Celsius
# Rainfall constraint: 0 to 500 mm

data = data[
    (data["temperature"] >= 0) &
    (data["temperature"] <= 50) &
    (data["rainfall"] >= 0) &
    (data["rainfall"] <= 500)
]

print("Cleaned Dataset Records:", len(data))


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = data[
    ["soil_type", "temperature", "rainfall", "season"]
]

y = data["label"]


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. RANDOM FOREST CLASSIFIER
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nRandom Forest Training Completed!")


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully as crop_model.pkl")
