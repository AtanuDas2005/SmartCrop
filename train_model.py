import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("crop_dataset_smartcrop.csv")

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

X = data[["soil_type","temperature","rainfall","season"]]

y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

pickle.dump(model, open("crop_model.pkl","wb")) #saves the model to a file named "crop_model.pkl"

print("Model trained successfully")