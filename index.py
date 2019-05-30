from flask import Flask,render_template
from sklearn.feature_extraction.text import HashingVectorizer
import readDataPredit as read
import trainModel as train
import pandas as pd 


app = Flask(__name__)

@app.route("/")
def home():
	dataResultPredit,score = train.getPredit()
	dataPredit = train.dataPredit[0]
	return render_template('home.html', dataResultPredit=dataResultPredit, score = score, dataPredit=dataPredit)

if __name__ == '__main__':
	app.run(debug=True)


