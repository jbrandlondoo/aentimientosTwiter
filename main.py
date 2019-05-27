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


from sklearn.naive_bayes import MultinomialNB
import pandas as pd 
import numpy as np 

data = []
multi = MultinomialNB()

with open('dataset.csv','r') as file:
	lineas = file.read().splitlines()
	i = 0
	for l in lineas:
		if i == 0:
			i += 1
		else:
			l = l.split(',')
			data.append([ClearData("".join(l)),l[len(l)-1]]);

data = pd.DataFrame(data)

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(data[0], data[1], test_size = 0.008)

from sklearn.feature_extraction.text import HashingVectorizer
hashingV = HashingVectorizer(stop_words='english',alternate_sign=False)
xTrain = hashingV.transform(xTrain)
xTest = hashingV.transform(xTest)


from sklearn.metrics import accuracy_score
multi.fit(xTrain,yTrain)
score = accuracy_score(yTest,multi.predict(xTest))
print(score)