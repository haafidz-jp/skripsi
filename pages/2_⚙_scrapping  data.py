import streamlit as st

st.markdown('## Scrapping Data dari Google Play Store')

from google_play_scraper import app
import pandas as pd
import numpy as np

import seaborn as sn
import pickle
import time
# scrape desired number of reviews
from google_play_scraper import Sort, reviews

result, continuation_token = reviews(
    'com.telkom.indihome.external',
    lang = 'id',
    country = 'id',
    sort = Sort.MOST_RELEVANT,
    count = 3000,
    filter_score_with = None # default to None ( means all score) use 1 or 2 or 3 or 4 or 5 to select certain score
)

data0012 = pd.DataFrame(np.array(result), columns=['review'])
data0012 = data0012.join(pd.DataFrame(data0012.pop('review').tolist()))

st.write('Pada halaman ini kita akan melakukan scrapping data melalui Google Play Store denga jumlah', len(data0012.index),'data.')


# Menampilkan data teratas
# data0012.head()
st.markdown('Berikut 5 data teratas dari data yang telah di scrapping:')
st.dataframe(data0012.head())

#menampilkan dataframe yang sudah di urutkan
st.markdown('menampilkan data yang sudah diurutkan berdasarkan waktu terbaru:')
data0012_new = data0012[['userName', 'score', 'at', 'content']]
data0012_sorted = data0012_new.sort_values(by='at', ascending = False) #Sort by Newest, change to True if you want to sort by Oldest.
data0012_sorted.head()

data0012_scrape = data0012_sorted[['userName', 'score', 'at', 'content']] # get userName, rating, date-time, and reviews only

st.dataframe(data0012_scrape.head())

data0012_scrape.to_csv("MyIndi_scrape_data.csv", index = False)

st.markdown('## Proses Filtering Data')
st.markdown('Pada halaman ini kita akan melakukan proses filtering data yang telah di scrapping.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df0012 = pd.read_csv('MyIndi_scrape_data.csv')
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
st.markdown(df0012_data.content)


# stopword (menghapus kata yang tidak penting)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
#from nltk.tokenize import word_tokenize
stops = set(stopwords.words('indonesian'))

st.markdown('keadaan data saat ini')
st.dataframe(df0012_data.head())

st.markdown('Jumlah data Sentimen')
st.dataframe(df0012_data['sentiment'].value_counts())

sn.countplot(df0012_data['sentiment'])
st.set_option('deprecation.showPyplotGlobalUse', False)
#plt.title("Plot Sentiment dari dataset")
st.markdown('Plot Sentiment dari dataset')
st.pyplot(plt.show())