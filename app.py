from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
import numpy as np
import random

# load the model from disk
def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data =[message]
        data=word_divide_char(data)
        vect = vectorizer.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction)
        return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
