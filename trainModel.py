from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cleanTxt as clean
import xgboost as xgb
import pandas as pd 
import os


dataTrain = []
dataPredit = []
resultPredit = []

def getDatos():
	dataH = []
	with open('dataset.csv','r') as file:
		lineas = file.read().splitlines()
		i = 0
		for l in lineas:
			if i == 0:
				i += 1
			else:
				l = l.split(',')
				dataPredit.append([clean.CleanData("".join(l)),l[len(l)-1]]);

def readDataTrain():
	path = [r".\neg",r".\pos"]
	i = 0
	for p in path:
		lista_de_archivos = os.listdir(p)
		for d in lista_de_archivos:
			try:
				with open(p+"\\"+d,'r') as file:
					lineas = file.read()
					dataTrain.append([clean.CleanData(lineas),i]);
			except Exception as e:
				pass
		i += 1

def getPredit():
	getDatos()
	readDataTrain()
	dataT = pd.DataFrame(dataTrain)
	dataP = pd.DataFrame(dataPredit)
	xTrain, xTest, yTrain, yTest = train_test_split(dataT[0], dataT[1], test_size = 0.008)
	hashingVT = HashingVectorizer(stop_words='english',alternate_sign=False,analyzer='word')
	hashingVP = HashingVectorizer(stop_words='english',alternate_sign=False,analyzer='word')
	xTrain, xTest, yTrain, yTest = train_test_split(dataT[0], dataT[1], test_size = 0.008)
	multi = MultinomialNB()
	xTrain = hashingVT.transform(xTrain)
	xTest =  hashingVP.transform(xTest)
	datos = hashingVT.transform(dataP[0])
	multi.fit(xTrain,yTrain)
	score = accuracy_score(yTest,multi.predict(xTest))
	modelo = xgb.XGBClassifier()
	modelo.fit(xTrain,yTrain)
	y_pred=modelo.predict(xTrain)
	prediccciones= [round(value) for value in y_pred]
	precision_train=accuracy_score(yTrain,prediccciones)
	if score > precision_train:
		return multi.predict(datos),score
	else:
		return modelo.predict(datos),precision_train


