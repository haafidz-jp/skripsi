import streamlit as st

# menghitung vector
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
content = df0012['content']

tfidfvectorizer = CountVectorizer(analyzer = 'word')
tfidf_wm = tfidfvectorizer.fit_transform(content)
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)

