import pandas as pd
import numpy as np
import streamlit as st
import joblib
from label import encode

st.set_page_config(page_title='Mushroom Prediction')
st.title('Mushrooms prediction🍄')  
st.markdown('lets find out your mushrooms edible or not🥰')
upload_file=st.file_uploader('Upload your raw .csv file')
if upload_file:
    df=pd.read_csv(upload_file)
    st.dataframe(df)
    st.header('Your Encoding file has been done')
    data=encode(df)
    st.dataframe(data)
    model=joblib.load('mushroom.pkl')
    predict=pd.DataFrame(model.predict(data))
    predict.columns=['results']
    result=predict.replace({0:'Edible',1:'Poisons'})
    st.header('Your Prediction is done...')
    st.dataframe(result)
    csv = result.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="download data as csv",
        data=csv,
        file_name='mushrooms_result.csv',
        mime='text/csv',
    )


st.markdown('..........**``Created by Rajesh``**.............')