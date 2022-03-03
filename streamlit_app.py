import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
import pyarrow as pa
import seaborn as sns
from PIL import Image
from dateutil import parser
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
st.set_page_config(page_title='Skripsi')
image=Image.open('logo.png')
logo = st.columns((1, 1))

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#page layout
with logo[1]:
	st.image(image,width=200)

with logo[0]:
	st.markdown("""
# Skripsi
###### *Penggunaan Algoritma Stacking Ensemble Learning Dalam Memprediksi Pengguna Enroll.*
""")
	st.markdown("""
**Riyo Santo Yosep - 171402020**

""")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #16A2CB;">
  <a class="navbar-brand" href="https://www.youtube.com/channel/UCvWPAZKStHARokG5nkPklAQ" target="_blank">Riyo Santo Yosep</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="/">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#preprocessing">Preprocessing</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#training-testing">Training_Testing</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#predicting">Predicting</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

# def upload_dataset():
#   dataset = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
#   return dataset

@st.experimental_memo
def load_dataset():
  df = pd.read_csv(dataset)
  return df

@st.experimental_memo
def preprocessing():
  df=load_dataset()
  #menghitung ulang isi dari kolom screen_litst karena tidak pas di kolom num screens
  df['screen_list'] = df.screen_list.astype(str) + ','
  df['num_screens'] = df.screen_list.astype(str).str.count(',')
  return df

@st.experimental_memo
def preprocessing1():
  df1 = preprocessing()
  # df1=df.copy()
  #feature engineering
  #karena kolom hour ada spasinya, maka kita ambil huruf ke 1 sampai ke 3
  df1.hour=df1.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open dan enrolled_date itu adalah string, maka perlu diubah ke datetime
  df1.first_open=[parser.parse(i) for i in df1.first_open]
  #didalam dataset orang yg belum langganan itu NAN, maka jika i=string biarin, klo ga string diuah ke datetime kolom nan nya biarin tetap nat
  df1.enrolled_date=[parser.parse(i) if isinstance(i, str)else i for i in df1.enrolled_date]
  #membuat kolom selisih , yaitu menghitung berapa lama orang yg firs_open menjadi enrolled
  df1['selisih']=(df1.enrolled_date-df1.first_open).astype('timedelta64[h]')
  #karna digrafik menunjukkan orang kebanyakan enroll selama 24 jam pertama, maka kalau lebih dari 24 jam dianggap ga penting
  df1.loc[df1.selisih>24, 'enrolled'] = 0
  return df1

@st.experimental_memo
def preprocessing2():
  df2=preprocessing1()
  # b=df2['screen_list'].apply(pd.Series).stack()
  # c = b.tolist()
  # from collections import Counter
  # p = Counter(' '.join(b).split()).most_common(100)
  # rslt = pd.DataFrame(p)
  # rslt.to_csv('data/top_screens.csv', index=False)
  top_screens=pd.read_csv('data/top_screens.csv')
  # diubah ke numppy arry dan mengambil kolom ke2 saja karna kolom1 isinya nomor
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  df3 = df2.copy()
  #mengubah isi dari file top screen menjadi numerik
  for i in top_screens:
    df3[i]=df3.screen_list.str.contains(i).astype(int)
  #semua item yang ada di file top screen dihilangkan dari kolom screen list
  for i in top_screens:
    df3['screen_list']=df3.screen_list.astype(str).str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df3['lainnya']=df3.screen_list.astype(str).str.count(',')
  return df3

@st.experimental_memo
def preprocessing_pred():
  df = preprocessing()
  #menghapus kolom numsreens yng lama
  df.drop(columns=['numscreens'], inplace=True)
  #mengubah kolom hour
  df.hour=df.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open itu adalah string, maka perlu diubah ke datetime
  df.first_open=[parser.parse(i) for i in df.first_open]
  #import top_screen
  top_screens=pd.read_csv('top_screens.csv')
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  for i in top_screens:
      df[i]=df.screen_list.str.contains(i).astype(int)
  for i in top_screens:
      df['screen_list']=df.screen_list.str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df['lainnya']=df.screen_list.str.count(',')
  #menghapus double layar
  layar_loan = ['Loan','Loan2','Loan3','Loan4']
  df['jumlah_loan']=df[layar_loan].sum(axis=1)
  df.drop(columns=layar_loan, inplace=True)

  layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
  df['jumlah_loan']=df[layar_saving].sum(axis=1)
  df.drop(columns=layar_saving, inplace=True)

  layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
  df['jumlah_credit']=df[layar_credit].sum(axis=1)
  df.drop(columns=layar_credit, inplace=True)

  layar_cc = ['CC1','CC1Category','CC3']
  df['jumlah_cc']=df[layar_cc].sum(axis=1)
  df.drop(columns=layar_cc, inplace=True)
  #mendefenisikan variabel numerik
  pred_numerik=df.drop(columns=['first_open','screen_list','user'], inplace=False)
  scaler = joblib.load('data/standar.joblib')
  fitur = pd.read_csv('data/fitur_pilihan.csv')
  fitur = fitur['0'].tolist()
  pred_numerik = pred_numerik[fitur]
  pred_numerik = scaler.transform(pred_numerik)
  model = joblib.load('data/stack_model.pkl')
  prediksi = model.predict(pred_numerik)
  probabilitas = model.predict_proba(pred_numerik)
  user_id = df['user']
  prediksi_akhir = pd.Series(prediksi)
  hasil_akhir= pd.concat([user_id,prediksi_akhir], axis=1).dropna()
  return probabilitas, hasil_akhir

@st.experimental_memo
def funneling():
  df=preprocessing2()
  #menggabungkan item yang mirip mirip, seperti kredit 1 kredit 2 dan kredit 3
  #funneling = menggabungkan beberapa screen yang sama dan menghapus layar yang sama
  layar_loan = ['Loan','Loan2','Loan3','Loan4']
  df['jumlah_loan']=df[layar_loan].sum(axis=1)
  df.drop(columns=layar_loan, inplace=True)

  layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
  df['jumlah_loan']=df[layar_saving].sum(axis=1)
  df.drop(columns=layar_saving, inplace=True)

  layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
  df['jumlah_credit']=df[layar_credit].sum(axis=1)
  df.drop(columns=layar_credit, inplace=True)

  layar_cc = ['CC1','CC1Category','CC3']
  df['jumlah_cc']=df[layar_cc].sum(axis=1)
  df.drop(columns=layar_cc, inplace=True)
  #menghilangkan kolom yang ga relevan
  df_numerik=df.drop(columns=['user','first_open','screen_list','enrolled_date','selisih','numscreens'], inplace=False)
  return df_numerik

@st.experimental_memo
def choose_feature(df_numerik, jumlah_fitur):
  df=df_numerik.copy()
  mutual_info = mutual_info_classif(df.drop(columns=['enrolled']), df.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df.drop(columns=['enrolled']).columns
  mutual_info.sort_values(ascending=False)
  # from sklearn.feature_selection import SelectKBest
  fitur_terpilih = SelectKBest(mutual_info_classif, k = jumlah_fitur)
  fitur_terpilih.fit(df.drop(columns=['enrolled']), df.enrolled)
  pilhan_kolom = df.drop(columns=['enrolled']).columns[fitur_terpilih.get_support()]
  pd.Series(pilhan_kolom).to_csv('data/fitur_pilihan.csv',index=False)
  fitur = pilhan_kolom.tolist()
  fitur_baru = df[fitur]
  return fitur_baru

@st.experimental_memo
def standarization(fitur_baru):
  sc_X = StandardScaler()
  pilhan_kolom = sc_X.fit_transform(fitur_baru)
  joblib.dump(sc_X, 'data/standar.joblib')
  return pilhan_kolom

@st.experimental_memo
def split(df_numerik,pilhan_kolom, split_size):
  df=df_numerik.copy()
  X_train, X_test, y_train, y_test = train_test_split(pilhan_kolom, df['enrolled'],test_size=(100-split_size)/100, random_state=111)
  return X_train, X_test, y_train, y_test

@st.experimental_memo
def naive_bayes(X_train, X_test, y_train, y_test):
  nb = GaussianNB() # Define classifier)
  nb.fit(X_train, y_train)
  # Make predictions
  y_test_pred = nb.predict(X_test)
  matrik_nb = (classification_report(y_test, y_test_pred))
  cm_label_nb = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  return matrik_nb, cm_label_nb, nb

@st.experimental_memo
def random_forest(X_train, X_test, y_train, y_test, parameter_n_estimators):
  rf = RandomForestClassifier(n_estimators=parameter_n_estimators, max_depth=2, random_state=42) # Define classifier
  rf.fit(X_train, y_train)
  # Make predictions
  y_test_pred = rf.predict(X_test)
  matrik_rf = (classification_report(y_test, y_test_pred))
  cm_label_rf = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  return matrik_rf, cm_label_rf, rf

@st.cache
# @st.experimental_singleton
def stack_model(X_train, X_test, y_train, y_test, tetangga, nb, rf):
  # Build stack model
  estimator_list = [
      ('nb',nb),
      ('rf',rf)]
  stack_model = StackingClassifier(
      estimators=estimator_list, final_estimator=KNeighborsClassifier(tetangga),cv=5
  )
  # Train stacked model
  stack_model.fit(X_train, y_train)
  # Make predictions
  y_test_pred = stack_model.predict(X_test)
  # Evaluate model
  matrik_stack = (classification_report(y_test, y_test_pred))
  cm_label_stack = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
  joblib.dump(stack_model, 'data/stack_model.pkl')
  return matrik_stack, cm_label_stack,y_test_pred

#####################
st.markdown('''
## Preprocessing
''')
with st.sidebar.header('1. Preprocess'):
  dataset = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
  st.sidebar.write(" ")
  st.sidebar.write(" ")
  st.sidebar.write(" ")
  st.sidebar.write(" ")
if dataset is not None:
  st.write(load_dataset())
  df=preprocessing()
  container = st.columns((1.9, 1.1))
  df_types = df.dtypes.astype(str)
  
  with container[0]:
    st.write(df)
    st.markdown('''
    Merevisi kolom numscreens''')
    # st.text('Merevisi kolom numscreens')
  with container[1]:
    st.write(df_types)
    st.markdown('''
    Merevisi kolom numscreens''')
    # st.text('Tipe data setiap kolom')
  
  df1=preprocessing1()
  container1 = st.columns((1.9, 1.1))
  df1_types = df1.dtypes.astype(str)
  with container1[0]:
    st.write(df1)
    st.text('Merevisi kolom hour')
  with container1[1]:
    st.write(df1_types)
    st.text('Tipe data setiap kolom')

  df4=preprocessing2()
  st.write(df4)
  st.text('Mengubah isi screen_list menjadi kolom baru')

  df_numerik=funneling()
  st.write(df_numerik)
  #membuat plot korelasi tiap kolom dengan enrolled
  korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corrwith(df_numerik.enrolled)
  plot=korelasi.plot.bar(title='korelasi variabel')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()
  st.text('Membuat plot korelasi tiap koklom terhadap kelasnya(enrolled)')
  from sklearn.feature_selection import mutual_info_classif
  #determine the mutual information
  mutual_info = mutual_info_classif(df_numerik.drop(columns=['enrolled']), df_numerik.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df_numerik.drop(columns=['enrolled']).columns
  mutual_info.sort_values(ascending=False)
  mutual_info.sort_values(ascending=False).plot.bar(title='urutannya')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()
  st.text('mengurutkan korelasi setiap kolom terhadap kelasnya(enrolled)')
  

st.markdown('''
## Training_Testing
''')
with st.sidebar.header('2. Set Parameter'):
  split_size = st.sidebar.slider('Rasio Pembagian Data (% Untuk Data Latih)', 10, 90, 80, 5)
  jumlah_fitur = st.sidebar.slider('jumlah pilihan fitur (Untuk Data Latih)', 5, 47, 20, 5)
  parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 10, 100, 50, 10)
  tetangga = st.sidebar.slider('Jumlah K (KNN)', 11, 101, 55, 11)

if "load_state" not in st.session_state:
  st.session_state.load_state = False
if st.sidebar.button('Latih & Uji') or st.session_state.load_state:
  st.session_state.load_state = True
  df_numerik = funneling()
  fitur_baru = choose_feature(df_numerik, jumlah_fitur)
  pilhan_kolom=standarization(fitur_baru)
  X_train, X_test, y_train, y_test = split(df_numerik,pilhan_kolom, split_size)
  matrik_nb, cm_label_nb, nb = naive_bayes(X_train, X_test, y_train, y_test)
  matrik_rf, cm_label_rf, rf = random_forest(X_train, X_test, y_train, y_test, parameter_n_estimators)
  matrik_stack, cm_label_stack, y_test_pred = stack_model(X_train, X_test, y_train, y_test, tetangga, nb, rf)

  nb_container = st.columns((1.1, 0.9))
  #page layout
  with nb_container[0]:
    st.write("2a. Naive Bayes report using sklearn")
    st.text('Naive Bayes Report:\n ' + matrik_nb)
  st.write(" ")
  st.write(" ")
  st.write(" ")
  with nb_container[1]:
    cm_label_nb.index.name = 'Actual'
    cm_label_nb.columns.name = 'Predicted'
    sns.heatmap(cm_label_nb, annot=True, cmap='Blues', fmt='g')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
  st.write(" ")
  st.write(" ")

  # Evaluate model
  rf_container = st.columns((1.1, 0.9))
  #page layout
  with rf_container[0]:
    st.write("2b. Random Forest report using sklearn")
    st.text('Random Forest Report:\n ' + matrik_rf)
  st.write(" ")
  st.write(" ")
  st.write(" ")
  with rf_container[1]:
    cm_label_rf.index.name = 'Actual'
    cm_label_rf.columns.name = 'Predicted'
    sns.heatmap(cm_label_rf, annot=True, cmap='Blues', fmt='g')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
  st.write(" ")
  st.write(" ")

  stack_container = st.columns((1.1, 0.9))
  #page layout
  with stack_container[0]:
    st.write("2c. Stack report using sklearn")
    st.text('Stack Report:\n ' + matrik_stack)
  st.write(" ")
  st.write(" ")
  st.write(" ")

  with stack_container[1]:
    cm_label_stack.index.name = 'Actual'
    cm_label_stack.columns.name = 'Predicted'
    sns.heatmap(cm_label_stack, annot=True, cmap='Blues', fmt='g')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
  st.write(" ")
  st.write(" ")      

  #take df3 from apps/praproses.py
  df1 = preprocessing2()
  var_enrolled = df1['enrolled']
  # #membagi menjadi train dan test untuk mencari user id
  X_train, X_test, y_train, y_test = train_test_split(df1, df1['enrolled'], test_size=(100-split_size)/100, random_state=111)
  train_id = X_train['user']
  test_id = X_test['user']
  #menggabungkan semua
  y_pred_series = pd.Series(y_test).rename('Aktual',inplace=True)
  hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
  hasil_akhir['Prediksi']=y_test_pred
  hasil_akhir = hasil_akhir[['user','Aktual','Prediksi']].reset_index(drop=True)
  container_hasil_akhir = st.columns((0.9, 1.2, 0.9))
  with container_hasil_akhir[1]:
    st.text('Tabel Perbandingan Asli dan Prediksi:\n ')
    st.dataframe(hasil_akhir)

#####################
st.markdown('''
## Predicting
''')
with st.sidebar.header('3. Predict'):
  data_pred = st.sidebar.file_uploader("Unggah File CSV",type=['csv'])
if data_pred is not None:
  dataset = data_pred
  df = load_dataset()
  st.dataframe(df)
  if st.sidebar.button("predict Data"):
    hasil_akhir, probabilitas = preprocessing_pred()
    st.dataframe(df)
    layout = st.columns((1,1))
    with layout[0]:
      st.write(hasil_akhir)
    with layout[1]:
      st.write(probabilitas)

