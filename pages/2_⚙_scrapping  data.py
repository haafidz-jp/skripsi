import streamlit as st

st.markdown('## Scrapping Data dari Google Play Store')

from google_play_scraper import app
import pandas as pd
import numpy as np

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

