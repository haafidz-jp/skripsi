import streamlit as st

# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
import nltk
nltk.download('stopwords')
import pandas as pd
# from IPython.display import display
import re
import string

df0012 = pd.read_csv('Text_Preprocessing.csv')

st.markdown('#### Tampilan Word Cloud untuk semua kata yang ada di dataset')
st.image('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/wcall.png')
# tweet_list = pd.read_csv('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/MyIndi_cleaning_data.csv')
# st.dataframe(tweet_list.head(5))

# print(tweet_list["content"].shape)

# tw_list = pd.DataFrame(tweet_list)

# rv_list_positive = pd.read_csv('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/posreview.csv')
# rv_list_positive = rv_list_positive.drop(columns = ['sentiment'])

# rv_list_negative = pd.read_csv('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/negreview.csv')
# rv_list_negative = rv_list_negative.drop(columns = ['sentiment'])

# def create_wordcloud(content):
#     mask = np.array(Image.open("cloud.png"))
#     stopwords = set(STOPWORDS)
#     wc = WordCloud(background_color="white",
#                     mask = mask,
#                     max_words=3000,
#                     stopwords=stopwords,
#                     repeat=True)
#     wc.generate(str(content))
#     wc.to_file("wc.png")
#     print("Word Cloud Saved Successfully")
#     path="wc.png"
#     display(Image.open(path))


# st.write(print(create_wordcloud(tw_list["content"].values)))


st.markdown('#### Word Cloud kata Positif')
st.image('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/wcpos.png')
# st.write(create_wordcloud(rv_list_positive["content"].values))


st.markdown('#### Word Cloud kata Negatif')
st.image('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/wcneg.png')
# st.write(create_wordcloud(rv_list_negative["content"].values))

st.markdown('#### Kata yang sering muncul di dataset')

# mengubah menjadi vector term
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

#Cleaning Text
ps = nltk.PorterStemmer()
stopword = nltk.corpus.stopwords.words('indonesian')

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

#Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df0012['content'])
print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
countdf[:]

st.markdown('#### Most Used Positive Word')
countdf[5:11]

st.markdown('#### Most Used Negative Word')
countdf[10:20]

st.markdown('#### Pie Chart')
# Plotting matplotlib
import matplotlib.pyplot as plt

labels = ["Positif", "Negatif"]
sizes = [2056, 944]
explode = (0.1,0)

fig, ax = plt.subplots(figsize=(10,10))
ax.pie(sizes, explode=explode,labels=labels,shadow=True, autopct="%1.1f%%")
ax.axis("equal")

st.pyplot(fig)