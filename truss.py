import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from os import system, name 

def get_input():
	data = [float(item) for item in input(": ").split()] 
	refined_data = [np.array(data)]
	
	chord_bf, c_score = predict_chord_bf(refined_data)

	data.append(chord_bf[0,0])
	
	predicted_price, p_score = predict_price(data)

	display_data(chord_bf, c_score, predicted_price, p_score)


def display_data(cb, cs, pp, ps):
	system('cls')
	print("CB: ", cb[0,0])
	print("CS: ", cs)
	print("PS: ", ps)

	print ("---------------------------")
	print("Predicted price: $%.2f" % pp)
	print ("---------------------------")
	get_input()

def predict_chord_bf(data):
	x_chord = df[['TCPitch', 'BCPitch', 'Span', 'TCSize', 'BCSize', 'OverhangL','OverhangR', 'HeelL', 'HeelR', 'Plies', 'Spacing', 'LumberCost', 'TCLL','TCDL', 'BCLL', 'BCDL']]
	y_chord = df[['LumberBDFT']]

	x_train, x_test, y_train, y_test = train_test_split(x_chord, y_chord, train_size = .8, test_size = 0.2, random_state=6)
	chord_model = lm.fit(x_train, y_train)

	chord_score = lm.score(x_test,y_test)

	chord_prediction = chord_model.predict(data)
	return chord_prediction, chord_score


def predict_price(x):
	reshaped_data = [np.array(x)]
	x_price = df[['TCPitch', 'BCPitch', 'Span', 'TCSize', 'BCSize', 'OverhangL','OverhangR', 'HeelL', 'HeelR', 'Plies', 'Spacing', 'LumberCost', 'TCLL','TCDL', 'BCLL', 'BCDL','ChordBDFT']]
	y_price = df[['Price']]

	x_train, x_test, y_train, y_test = train_test_split(x_price, y_price, train_size = .8, test_size = 0.2, random_state=6)
	price_model = lm.fit(x_train, y_train)

	price_score = lm.score(x_test, y_test)

	price_prediction = price_model.predict(reshaped_data)
	return price_prediction, price_score



df = pd.read_csv("truss_data_filtered.csv")

lm = LinearRegression()

get_input()