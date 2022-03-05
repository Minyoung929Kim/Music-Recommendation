import os
import pickle
import pandas as pd
import json
from flask import Flask, request

from model import get_survey_model
from dataclass import DataClass

app = Flask(__name__) #opening up app
APP_ROOT = os.getenv('APP_ROOT', '/predict') #https://www, if we have /predict, send our data to the url


#preprocess the results into a vector --> preprocessing
#deep learning model to make it into actual vector --> deep learning model
#load the database and compare the database with all the values in the data base --> data base

#our full mode consists fo two parts 1, deep learning mode, and text embedding for music recommendation, so doing embeeddding now is wast eo f space so leave that later

#정리:

#1. we need to load the preprocessing code
with open('survey_data/data_mappings.json', 'r') as f:
    data_mappings = json.load(f)
data_class = DataClass('Gathered data.csv')

#2 we need to load the model (not the full mode, but survey model only)
survey_model = get_survey_model(128, 768)
survey_model.load_weights('checkpoints/survey_best')

#3 we need to load the database
with open('database.pkl', 'rb') as f:
    database = pickle.load(f)

#preprocessing function
@app.route(APP_ROOT, methods=["POST"])
def predict():
    data = request.json #send data into this link, stored in this file
    survey = data.get("survey") #format data so get survey

    #transform from json -> dict -> pandas dataframe so we can preprocess it using fucntion
    survey_df = pd.DataFrame.from_dict(eval(survey), orient='index').T
    survey_vector = data_class.process_test_data(survey_df, data_mappings).astype('float32') #required by tensorflow
    
    survey_vector = survey_model(survey_vector).numpy()
    songs = database.search(survey_vector, topk=5)
    #one detail required, can't just return songs
    return {"songs": [s.values.tolist() for s in songs]} #pandas dataframe into list
    #this is everything I need to knwo about for prediction code


#two files because we use heroku, looks at procfile