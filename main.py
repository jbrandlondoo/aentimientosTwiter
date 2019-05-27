from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np 

class modelo():
	data = []
	multi = MultinomialNB()

	def __init__(self, arg):
		super(modelo, self).__init__()
		self.arg = arg
		entrenar()
		

	def ClearData(text):
		text = text.replace(",","")
		text = text.replace(".","")
		text = text.replace("/","")
		text = text.replace(">","")
		text = text.replace("<","")
		text = text.replace('"',"")
		text = text.replace("'","")
		text = text.replace("1","")
		text = text.replace("2","")
		text = text.replace("3","")
		text = text.replace("4","")
		text = text.replace("5","")
		text = text.replace("6","")
		text = text.replace("7","")
		text = text.replace("8","")
		text = text.replace("9","")
		text = text.replace("0","")
		text = text.replace("*","")
		text = text.replace("!","")
		text = text.replace("(","")
		text = text.replace(")","")
		text = text.replace(";","")
		text = text.replace(":","")
		return text

	def setData(datset):
		dataH = []
		with open(datset,'r') as file:
			lineas = file.read().splitlines()
			i = 0
			for l in lineas:
				if i == 0:
					i += 1
				else:
					l = l.split(',')
					dataH.append([ClearData("".join(l)),l[len(l)-1]]);
		dataH = pd.DataFrame(dataH)
		data = dataH

	def entrenar():
		setData('dataset.csv')
		xTrain, xTest, yTrain, yTest = train_test_split(data[0], data[1], test_size = 0.008)
		hashingV = HashingVectorizer(stop_words='english',alternate_sign=False)
		xTrain = hashingV.transform(xTrain)
		xTest = hashingV.transform(xTest)
		multi.fit(xTrain,yTrain)
		score = accuracy_score(yTest,multi.predict(xTest))
		print(score)
	