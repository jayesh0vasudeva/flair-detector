from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource 
from werkzeug.utils import secure_filename
import pandas as pd
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import re
import pickle
import praw
import pprint
import json


nltk.download('punkt')
nltk.download('stopwords')

client_id = 'F9GtKcaO_sk9xw'
secret = 'hRtKBcb40jnXhQGUZFX8OHCpkxY'

reddit = praw.Reddit(user_agent='funky_cool_bruh', client_id=client_id, client_secret=secret,
                     username='jayesh0vasudeva', password='jaimatadi88JAI')

def preProcessData(dataa):
    stopwords_en = list(set(stopwords.words('english')))
    def split(word): 
        return [char for char in word]   
    punchList = split(punctuation)

    wordTokenList = [word_tokenize(sent) for sent in dataa]
    lowercasingList = [[word.lower() for word in sentence] for sentence in wordTokenList]
    noStopWordList = [[word for word in sentence if word not in stopwords_en] for sentence in lowercasingList]
    noPunchList = [[re.sub(r'([^\s\w]|_)+', '', word) for word in sentence] for sentence in noStopWordList]
    PP_data = [[word for word in sentence if word] for sentence in noPunchList]
    return PP_data

def text_extractor(text,text_type):
    title_list=[]
    for i in range(len(text)):
        title_list.append(text[text_type][i])
    return title_list
def joiner(data):
    input_corrected = [" ".join(i) for i in data]
    return input_corrected
def detect_flair(url,loaded_model):

    submission = reddit.submission(url=url)
    topics_dict = {"title":[], "comments":[]}
    topics_dict["title"].append(submission.title)
    submission.comments.replace_more(limit=None)
    comment = ''
    for top_level_comment in submission.comments:
        comment = comment + ' ' + top_level_comment.body
    topics_dict["comments"].append(comment)
    
    topics_data = pd.DataFrame(topics_dict)
    feature_combine = topics_data["title"] + topics_data["comments"]
    topics_data = topics_data.assign(feature_combine = feature_combine)
    feature=text_extractor(topics_data,'feature_combine')
    x=joiner(preProcessData(feature))
    return loaded_model.predict(x)

filename = 'rfr_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction", methods=['POST'])
def prediction():
     if request.method=='POST':
        link=request.form["email_name"]
        a=detect_flair(link,loaded_model)
        str_a=str(a[0])
        return render_template("prediction.html",str_a= str_a,link=link)
        
@app.route("/automated_testing", methods=['POST', 'GET'])
def automated_testing():

    if request.files:
        file = request.files["upload_file"]
        texts = file.read()
        texts = str(texts.decode('utf-8'))
        links = texts.split('\n')
        pred = {}
        for link in links:
            pred[link] =  str(prediction(str(link)))[2:-2]
        return jsonify(pred)
    else:
        return 400
    

if __name__=='__main__':
    app.debug=True
    app.run()
    
    

