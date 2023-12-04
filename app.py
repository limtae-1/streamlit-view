import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("Visit Counting")

@st.cache_data
def load_data():
    # 데이터프레임을 여기에서 로드합니다
    df = pd.read_csv('result_final.csv')
    return df

df = load_data()
tab1, tab2, tab3 = st.tabs(['raw data', 'scatter', 'bar'])

with tab1:
    st.dataframe(df)
    
with tab2:
    
    fig2 = plt.figure() # 여기부터 그래프영역
    plt.scatter(data = df, x = 'mday', y ='dis')
    plt.xticks(df['mday'].unique())    
    plt.yticks(np.arange(0,101,10))
    plt.title('Visit Counting Scatter')
    plt.xlabel('2023.11.27.-2023.12.01.')
    plt.ylabel('Visit Counting')
    st.pyplot(fig2)

    fig2 = plt.figure() # 여기부터 그래프영역
    plt.scatter(df['mday'],df['dis'])
    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("visit counting")
    plt.xticks(np.arange(0,32,1))     #x축 눈금 설정
    plt.yticks(np.arange(0,101,10))   #y축 눈금 설정
    plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')
    st.pyplot(fig2)

with tab3:
    fig3 = plt.figure() # 여기부터 그래프영역
    plt.bar(sorted(df["mday"].unique()), df.groupby('mday')['dis'].count(), color="green")
    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("visit counting")
    plt.xticks(np.arange(1,32,1))     #x축 눈금 설정
    plt.yticks(np.arange(0,101,10))   #y축 눈금 설정
    #x축의 데이터 설정 lenth는 눈금의 길이, pad는 눈금과 레이블의 거리, labelsize는 레이블의 크기
    plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')

    
    st.pyplot(fig3)
    plt.bar(sorted(df["mday"].unique()), df.groupby('mday')['dis'].count(), color="green")
    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("visit counting")
    plt.xticks(np.arange(1,32,1))     #x축 눈금 설정
    plt.yticks(np.arange(0,101,10))   #y축 눈금 설정

    #x축의 데이터 설정 lenth는 눈금의 길이, pad는 눈금과 레이블의 거리, labelsize는 레이블의 크기
    plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')

    x=sorted(df["mday"].unique())
    y=list(df.groupby('mday')['dis'].count())
    for i, v in enumerate(x):
        plt.text(v, y[i], y[i],
                 fontsize = 8,
                 fontweight = 'bold',
                 color = 'royalblue',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    st.pyplot(fig3)


