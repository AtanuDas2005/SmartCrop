import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
data = pd.read_csv("crop_dataset_smartcrop.csv")

# Convert text values to numbers
soil_map = {
"loamy":0,
"sandy":1,
"clay":2,
"black":3
}

season_map = {
"summer":0,
"winter":1,
"monsoon":2
}

data["soil_type"] = data["soil_type"].map(soil_map)
data["season"] = data["season"].map(season_map)

# Features
X = data[["soil_type","temperature","rainfall","season"]]

# Label
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

# Save model
pickle.dump(model, open("crop_model.pkl","wb"))

print("Model trained successfully")