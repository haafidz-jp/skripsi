import streamlit as st
import pandas as pd
import time

#import csv
df0012 = pd.read_csv('https://raw.githubusercontent.com/haafidz-jp/skripsi/master/MyIndi_clean_data.csv')

# menghitung vector
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
content = df0012['content']

tfidfvectorizer = CountVectorizer(analyzer = 'word')
tfidf_wm = tfidfvectorizer.fit_transform(content)
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
st.markdown('## TD-IDF Vectorizer')
st.markdown('Term-weighting merupakan proses pemberian bobot term pada dokumen. Pembobotan ini digunakan nantinya oleh algoritma Machine Learning untuk klasifikasi dokumen. Ada beberapa metode yang dapat digunakan, salah satunya adalah TF-IDF (Term Frequency-Inverse Document Frequency).')
st.markdown('Berikut ini adalah hasil dari TF-IDF Vectorizer:')
st.write(df_tfidfvect)
st.write('Shape Array : ', df_tfidfvect.shape)

x = df0012.iloc[:, 0].values # membuat variabel x berisi content dari dataset
y = df0012.iloc[:, -1].values # membuat variabel y berisi nilai sentiment dari dataset
# memecah data test 20% dari keseluruhan data
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# mengubah menjadi vector term
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(x_test)

X_test_cv[1:1]

st.markdown('Membuat Model KNN dengan k = 7')
# membuat model knn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X_train_cv, y_train)

st.code('from sklearn.neighbors import KNeighborsClassifier \nmodel = KNeighborsClassifier(n_neighbors = 7) \nmodel.fit(X_train_cv, y_train)')

st.markdown('Kemudian kita memprediksi menggunakan dataset')
st.code('y_pred = model.predict(X_test_cv)')
# prediksi menggunakan dataset
y_pred = model.predict(X_test_cv)

st.markdown('selanjutnya kita melakukan perhitungan confusion matrix dan akurasi')
st.code('from sklearn.metrics import classification_report, confusion_matrix \ncm = confusion_matrix(y_test, y_pred) \ncr = classification_report(y_test, y_pred) \nprint(cm) \nprint(cr)')
# menghitung confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

start_time = time.time()

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


stop_time = time.time()
execution_time = stop_time - start_time
st.write(f'Confusing Matrix Complete')
st.write(f'Time Taken: {round(execution_time,3)} seconds \n')

st.write(cm)
st.write(cr)

