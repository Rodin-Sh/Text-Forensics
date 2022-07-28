import streamlit as st
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
from afinn import Afinn

st.write('# Text Forensics')
st.write('Text Forensics is a website that utilizes natural language processing to perform analysis on the entered text. The result of the analysis will present the **sentiment**, **polarity score**, **factuality**, and **emotion** of the text.')

FakeNewsDetector = 'FakeNewsDetector.sav'
FakeNewsDetector = pickle.load(open(FakeNewsDetector, 'rb'))
FakeNewsCV = 'FakeNewsCV.pkl'
FakeNewsCV = joblib.load(FakeNewsCV)

EmotionDetector = 'EmotionDetector.sav'
EmotionDetector = pickle.load(open(EmotionDetector, 'rb'))
EmotionCV = 'EmotionCV.pkl'
EmotionCV = joblib.load(EmotionCV)

afn = Afinn()

text_input = st.text_area("Enter a text that you would like to have analyzed", max_chars=200)

def text_clean (text_input):
    text_input = re.sub('[^a-zA-Z]', ' ', text_input)
    text_input = text_input.lower()
    text_input = text_input.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text_input = [ps.stem(word) for word in text_input if not word in set(all_stopwords)]
    text_input = ' '.join(text_input)
    new_corpus = [text_input]
    X_FakeNews = FakeNewsCV.transform(new_corpus).toarray()
    X_Emotion = EmotionCV.transform(new_corpus).toarray()

    return X_FakeNews, X_Emotion

Emotions = {0: 'Sad',
            1: 'Joy',
            2: 'Love',
            3: 'Anger',
            4: 'Fear'}

News = {0: 'Most Likely Misinformation',
        1: 'Most Likely Valid News'}

if st.button("Perform Analysis"):
  text_input_cleaned = text_clean(text_input)
  
  y_FakeNews = FakeNewsDetector.predict(text_input_cleaned[0])
  y_Emotion = EmotionDetector.predict(text_input_cleaned[1])
  
  y_Sentiment = afn.score(text_input)
  if y_Sentiment >= 5:
    y_SentimentLabel = 'Very Positive'
  elif y_Sentiment <= -5:
    y_SentimentLabel = 'Very Negative'
  elif y_Sentiment > 0:
    y_SentimentLabel = 'Positive'
  elif y_Sentiment < 0:
    y_SentimentLabel = 'Negative'
  else:
    y_SentimentLabel = 'Neutral'

  
  st.markdown('### Analysis Results')
  st.markdown('##### **Sentiment Present:** ' + str(y_SentimentLabel))
  st.write("The mental attitute within the text determined by authors' feelings.")
  st.markdown('##### **Polarity Score:** ' + str(y_Sentiment))
  st.write('The numerical measurement of the overall sentiments within a text.')
  y_FakeNews = y_FakeNews[0]
  st.markdown('##### **Factuality:** ' + str(News[y_FakeNews]))
  st.write('The validity of the information of the text.')
  y_Emotion = y_Emotion[0]
  st.markdown('##### **Emotion:** ' + str(Emotions[y_Emotion]))
  st.write('The psychological state of the text.')
