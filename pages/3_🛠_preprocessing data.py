import streamlit as st

st.markdown('## Proses Filtering Data')
st.markdown('Pada halaman ini kita akan melakukan proses filtering data yang telah di scrapping.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sn
import pickle
import time

df0012 = pd.read_csv('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/MyIndi_scrape_data.csv')
st.dataframe(df0012.head(5))

#mengubah nilai skor menjadi positif atau negatif
sentimen = []
for index, row in df0012.iterrows():
    if row['score'] > 3:
        sentimen.append ('1') # positif
    else :
        sentimen.append ('-1') # negatif

df0012['sentiment'] = sentimen
st.markdown('Menambahkan Kolom sentimen.')
st.dataframe(df0012.head())

# menghilangkan variabel yang tidak dipakai
df0012_data = df0012.copy()
df0012_data = df0012.drop(columns = ['userName', 'score', 'at'])
st.markdown('Menghilangkan kolom yang tidak perlu')
st.dataframe(df0012_data.head())


x = df0012_data.iloc[:, 0].values # membuat variabel x berisi content dari dataset
y = df0012_data.iloc[:, -1].values # membuat variabel y berisi nilai sentiment dari dataset
# memecah data test 20% dari keseluruhan data
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
st.markdown('Memecah data menjadi test dan training dengan metode 80:20')
st.code('X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # stemming

st.markdown('Data sebelum di cleaning:')
st.dataframe(df0012_data.content)

# Cleaning
# remove url
df0012_data['content'] = df0012_data['content'].str.replace('https\S', ' ', case = False) # Tokenizing

# merubah teks menjadi huruf kecil
df0012_data['content'] = df0012_data['content'].str.lower() # case folding

# remove mention
df0012_data['content'] = df0012_data['content'].str.replace('@\S+', ' ', case = False) # Tokenizing

# remove hashtag
df0012_data['content'] = df0012_data['content'].str.replace('#\S+', ' ', case = False) # Tokenizing

# remove next character
df0012_data['content'] = df0012_data['content'].str.replace("\'\w+", ' ', case = False) # Tokenizing

# remove puntuation
df0012_data['content'] = df0012_data['content'].str.replace('[^\w\s]', ' ', case = False) # Tokenizing

# remove number
df0012_data['content'] = df0012_data['content'].str.replace(r'\w*\d+\w', ' ', case = False) # Tokenizing

# remove spasi berlebih
df0012_data['content'] = df0012_data['content'].str.replace('\s(2)', ' ', case = False) # Tokenizing

st.markdown('Data setelah di cleaning:')
st.dataframe(df0012_data.content)
df0012_data.to_csv("MyIndi_cleaning_data.csv", index = False)

st.markdown('data sentimen negatif')
#export csv data cleaning negatif
rv_list_negative = df0012_data[df0012_data['sentiment'] == '-1']
rv_list_negative.to_csv("negreview.csv", index = False)
st.dataframe(rv_list_negative)

st.markdown('data sentimen positif')
#export csv data cleaning positif
rv_list_positive = df0012_data[df0012_data['sentiment'] == '1']
rv_list_positive.to_csv("posreview.csv", index = False)
st.dataframe(rv_list_positive)


st.markdown('Contoh data yang sudah di tokenize')
# tokenizing
# testing
from nltk.tokenize import word_tokenize

x = df0012_data.iloc[0]
st.write(nltk.word_tokenize(x['content']))


def identify_tokens(row) :
    text = row ['content']
    tokens = nltk.word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df0012_data['content'] = df0012_data.apply(identify_tokens, axis = 1)
st.markdown('data setelah di tokenize:')
st.dataframe(df0012_data.content)


# stemming (pembentukan kata dasar)
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemming = PorterStemmer()

def stem_list(row) :
    text = row ['content']
    stem = [stemming.stem(word) for word in text]
    return(stem)

df0012_data['content'] = df0012_data.apply(stem_list, axis = 1)
st.markdown('data setelah di stemming:')
st.write(df0012_data.content)


# stopword (menghapus kata yang tidak penting)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
#from nltk.tokenize import word_tokenize
stops = set(stopwords.words('indonesian'))

st.markdown('Tampilkan data saat ini')
st.dataframe(df0012_data.head())

st.markdown('Jumlah data Sentimen')
st.dataframe(df0012_data['sentiment'].value_counts())

sn.countplot(df0012_data['sentiment'])
st.set_option('deprecation.showPyplotGlobalUse', False)
#plt.title("Plot Sentiment dari dataset")
st.markdown('Plot Sentiment dari dataset')
st.pyplot(plt.show())

st.markdown('Export clean data csv')
# df0012_data.to_csv("MyIndi_clean_data.csv", index = False)
st.code('df0012_data.to_csv("MyIndi_clean_data.csv", index = False)')