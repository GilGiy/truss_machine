from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import operator
import numpy as np

df = pd.read_csv("truss_data_filtered.csv")

x = df[['TCPitch', 'BCPitch', 'Span', 'TCSize', 'BCSize', 'OverhangL','OverhangR', 'HeelL', 'HeelR', 'Plies', 'Spacing', 'LumberCost', 'TCLL','TCDL', 'BCLL', 'BCDL']]
y = df[['Price']]

training_data, validation_data, training_labels, validation_labels = train_test_split(x, y, test_size = .2, random_state = 100)

regressor = KNeighborsRegressor(n_neighbors = 22, weights="distance")
regressor.fit(training_data, training_labels)

score = regressor.score(validation_data, validation_labels)

print(score)

data = [float(item) for item in input(": ").split()] 
refined_data = [np.array(data)]

predictions = regressor.predict([refined_data])

print(predictions)
