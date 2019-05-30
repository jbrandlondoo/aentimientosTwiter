from flask import Flask,render_template
from sklearn.feature_extraction.text import HashingVectorizer
import trainModel as train
import pandas as pd 
import numpy


app = Flask(__name__)

@app.route("/")
def home():
	dataResultPredit,score = train.getPredit()
	dataPredit = train.dataPredit[0]
	dataResultPredit = numpy.array(dataResultPredit)
	dataPredit = numpy.array(dataPredit)

	return render_template('home.html',dataPredit = dataPredit[0:20].T,dataResultPredit = dataResultPredit[0:20].T, score = score)

if __name__ == '__main__':
	app.run(debug=True)


