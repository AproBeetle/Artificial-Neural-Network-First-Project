"""
	Artificial Neural Network(ANN) which trains
	using a csv file which contains 3 columns:
		- Color(Culoare);
		- Length(Lungime);
		- Width(Latime).
	Using these columns, the rows represent flowers
	that have the respective properties.
	The A.I. will predict if the flower is red
	or blue.
	
	Some information about me:
	• Author: Mark
	• Age: 16
	• Date of ending the project: 18 february 2024
"""

# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random, json

# importing the CSV
# getting the number of rows
# getting the number columns
# setting up the indexes
data = pd.read_csv('data.csv')
ROWS, COLUMNS = data.shape
indexes = [i for i in range(ROWS)]

# graph the points
for i in range(ROWS):
	color = "b"
	if data.iloc[i,0] == 1:
		color = "r"
	plt.scatter(data.iloc[i,1], data.iloc[i,2], c=color)

# activation function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

# train function (where the actual training is happening)
def train():
	# getting the activation function parameters from the input.json file
	with open('input.json', 'r') as file:
		tempData = json.load(file)
		w1 = tempData["w1"]
		w2 = tempData["w2"]
		b = tempData["b"]
	learning_rate = 0.2
	
	# main loop
	for i in range(50000):
		# pick a random point
		RI = random.choice(indexes)
		target = data.iloc[RI,0]
		
		# get prediction
		z = w1*data.iloc[RI,1] + w2*data.iloc[RI,2] + b
		pred = sigmoid(z)
		
		# get the derivativesof the functions
		dcost_dpred = 2 * (pred - target)
		dpred_dz = sigmoid(z) * (1-sigmoid(z))
		dz_dw1 = data.iloc[RI,1]
		dz_dw2 = data.iloc[RI,2]
		dz_db = 1
		
		# derivatives with respect to each paramteres
		# we can use the chain rule from calculus
		dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
		dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
		dcost_db = dcost_dpred * dpred_dz
		
		# modify the parameters
		w1 = w1 - learning_rate*dcost_dw1
		w2 = w2 - learning_rate*dcost_dw2
		b = b - learning_rate*dcost_db
	
	# updating the input file (the parameters)
	with open("input.json", "r+") as f:
	  oldData = json.load(f)
	  oldData["w1"] = w1
	  oldData["w2"] = w2
	  oldData["b"] = b
	  f.seek(0)
	  json.dump(oldData, f)
	  f.truncate()

# test function (where we can give test inputs)
def test(m1, m2):
	plt.scatter(m1, m2, c='black')
	with open("input.json", "r") as f:
		datax = json.load(f)
		z = datax["w1"]*m1 + datax["w2"]*m2 + datax["b"]
		pred = sigmoid(z)
		print(pred)
		if pred > 0.5:
			print("RED")
		else:
			print("BLUE")

# showing the graph containing the points
plt.show()