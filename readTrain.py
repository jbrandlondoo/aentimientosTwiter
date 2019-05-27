from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np 
import os
import xgboost as xgb

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

dataH = []

from os import scandir
path = [r".\neg",r".\pos"]
i = 0
for p in path:
	lista_de_archivos = os.listdir(p)
	for d in lista_de_archivos:
		try:
			with open(p+r"\\"+d,'r') as file:
				lineas = file.read()
				dataH.append([ClearData(lineas),i]);
		except Exception as e:
			pass

	i += 1

dataH = pd.DataFrame(dataH)
multi = MultinomialNB()
xTrain, xTest, yTrain, yTest = train_test_split(dataH[0], dataH[1], test_size = 0.008)
hashingV = HashingVectorizer(stop_words='english',alternate_sign=False)
xTrain = hashingV.transform(xTrain)
xTest = hashingV.transform(xTest)
multi.fit(xTrain,yTrain)
score = accuracy_score(yTest,multi.predict(xTest))
print(score)

modelo= xgb.XGBClassifier()
modelo.fit(xTrain,yTrain)

#prediccion
y_pred=modelo.predict(xTrain)
prediccciones= [round(value) for value in y_pred]
precision_train=accuracy_score(yTrain,prediccciones)

print(precision_train)

