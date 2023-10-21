import joblib
import numpy as np

loaded_model = joblib.load("models/knn_model.pkl")

new_data_point = np.array([39.0348,19.47,2.67,0.29,0.1648,36.09,6.75,10.419,81.0166,229.4088])  

prediction = loaded_model.predict([new_data_point])

print("Prediction:", prediction[0])
