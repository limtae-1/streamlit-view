import streamlit as st
import pandas as pd


@st.cache
def load_data():
    # 데이터프레임을 여기에서 로드합니다
    df = pd.read_csv('result_final.csv')
    return df

st.header("jungtae homework")
df = load_data()
st.dataframe(df)
