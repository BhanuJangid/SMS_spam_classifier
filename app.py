import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer


def mess_tranform(mess):
    mess = mess.lower()  #to lower case
    mess = nltk.word_tokenize(mess)  #different words
    ps = PorterStemmer()
    
    y = []     #removing special char
    for i in mess:
        if i.isalnum():
            y.append(i)
    
    mess = y[:]
    
    # removing sentence forming words with no meaning
    y.clear()
    
    for i in mess:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    mess = y[:]
    y.clear()
    
    # removing stamming , means planging ->play
    
    for i in mess:
        y.append(ps.stem(i))
        
    return " ".join(y)
    
model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))

st.title("SMS SPAM CLASSIFIER")
input_sms = st.text_input("ENTER THE MESSAGE")

if st.button("PREDICT"):

    trans_mess = mess_tranform(input_sms)
    vector_input = tfidf.transform([trans_mess])

    result = model.predict(vector_input)

    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")


     