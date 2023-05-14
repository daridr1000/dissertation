from flask import Flask, render_template, request
from datetime import date
import os
import openai
import json
import requests
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np## for plotting
import matplotlib.pyplot as plt
import transformers
import re
import pickle, string
import json, re


key2 = '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'
openai.api_key = f"{key2}"
app = Flask(__name__)


model = keras.models.load_model("models/distilbert_model.h5", custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})

tokenizer_distilbert = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

VERDICT_MAP = {
    0:"Not The Asshole",
    1:"You The Asshole",
    2:"Everyone Sucks Here",
    3:"No Assholes Here"
}

INVERSE_VERDICT_MAP = {
    "Not The Asshole":0,
    "You The Asshole":1,
    "Everyone Sucks Here":2,
    "No Assholes Here":3
}

# Cleaning functions
def remove_links(text):
    return re.sub(r'http\S+', '', text)
def make_lower(text):
    return text.lower()
def remove_punctuation(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) 
    return re.sub(r'[^\w\s]', ' ', text)
def remove_extraspaces(text):
    return re.sub('\s\s+',' ', text)
def clean_text(text):
    text = remove_links(text)
    text = make_lower(text)
    text = remove_extraspaces(text)
    text = remove_punctuation(text)
    return text

# Encodes input into distilbert format
def get_bert_encoding(query, tokenizer_distilbert):  # This code has been adapted from the following blog https://resources.experfy.com/ai-ml/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert/
    maxlen = 512
    max_special_characters = int((maxlen-20)/2)
    query_tokenized = "[CLS] "+" ".join(tokenizer_distilbert.tokenize(clean_text(query))[:max_special_characters])+" [SEP] "
    
    masks = [[1]*len(query_tokenized.split(" ")) + [0]*(maxlen - len(query_tokenized.split(" ")))]
    txt2seq = query_tokenized + " [PAD]"*(maxlen-len(query_tokenized.split(" "))) if len(query_tokenized.split(" ")) != maxlen else query_tokenized
        
    idx = [tokenizer_distilbert.convert_tokens_to_ids(txt2seq.split(" "))]
    
    query = [np.asarray(idx, dtype='int64'), 
              np.asarray(masks, dtype='int64')]
    return query

# Returns the prediction probabilities
def get_prediction_probs(query, model, tokenizer_distil_bert):
    query_encoded = get_bert_encoding(query, tokenizer_distilbert)
    predicted_prob = model.predict(query_encoded)
    return predicted_prob



@app.route("/", methods=['POST','GET'])
def main_page():
    verdict = ""
    explanation = ""
    recommendation = ""
    story = ""
    title = "" 
    if request.method == 'POST':
        title = request.form["title"]
        story = request.form["story"]
        body = title + " " + story
        predicted_probs = get_prediction_probs(body, model, tokenizer_distilbert)
        predicted = VERDICT_MAP[np.argmax(predicted_probs)]
        verdict += predicted


        # Send requests to ChatGPT3.5 
        response_explanation = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user", "content":"A BERT machine learning model has been trained to classify moral judgement stories with the following four labels: You the asshole, Not the asshole, Everyone sucks here and No assholes here.  The verdict for the following scenario has been " +verdict+". Can you explain why the classifier made this decision? Do not put 'As an AI language model' in your answer. Here is the scenario: "+ story}])
        response_recommendation = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user", "content":"Do not put 'As an AI language model' in your answer for the following question. What would you recommend to the user in the following scenario: "+ body}])

        recommendation += response_recommendation.choices[0]["message"]["content"]
        explanation+= response_explanation.choices[0]["message"]["content"]
    return render_template("index.html", recommendation = recommendation, explanation = explanation, title = title, story = story, verdict = verdict)
