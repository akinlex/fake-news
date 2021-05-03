import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse

loaded_model = pickle.load(open('../fake_news_pred_model.h5', 'rb'))
loaded_vec = pickle.load(open('../vectorizer.pickle', 'rb'))

print(loaded_model)
print(loaded_vec)

app = FastAPI()

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

cv = CountVectorizer(analyzer='word', ngram_range=(1,3))
    
def clean_text(text):
    # Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    cleaned_text = re.sub('\[.*?\]', '', text)
    cleaned_text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    cleaned_text = re.sub('\w*\d\w*', '', text)
    cleaned_text = cleaned_text.lower()

    # Tokenize the text into word
    words = nltk.word_tokenize(cleaned_text)

    #Remove stop words and words with length less than equal to 3
    filtered_words = [word for word in words if not word in stop_words and len(word) > 3]

    #Lemmatize
    output_sentence = ''
    for word in filtered_words:
        output_sentence = output_sentence  + ' ' + str(lemmatizer.lemmatize(word))

    return output_sentence

def pipeline(text, cv):
    X = cv.transform(text)
    return X


@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <textarea name="text"></textarea>
        <input type="submit" />'''

@app.post('/predict')
def predict(text:str = Form(...)):
    text_cleaned = clean_text(text)
    df = pd.DataFrame()
    df.loc[0,'cleaned_text'] = text_cleaned
    clean_text_new = pipeline(df['cleaned_text'], loaded_vec)
    prediction = loaded_model.predict(clean_text_new)

    for i in range(len(prediction)):
        if prediction[i] == 0:
            news = 'Real news'
        else:
            news = 'Fake news'

    return { #return the dictionary for endpoint
         "ACTUAL STORY": text[:120],
         "PREDICTED": news,
         "PROBABILITY": loaded_model.predict_proba(clean_text_new)
    }

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}