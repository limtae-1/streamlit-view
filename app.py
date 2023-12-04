import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("방문자 횟수 데이터 분석")

@st.cache_data
def load_data():
    # 데이터프레임을 여기에서 로드합니다
    df = pd.read_csv('result_final.csv')
    return df

df = load_data()
tab1, tab2, tab3, tab4, tab5 = st.tabs(['원본 데이터', '산점도 그래프', '막대 그래프','점 그래프','소스코드'])

with tab1:
    st.subheader("초음파 센서를 활용한 수집된 데이터")
    st.write("2023.11.27.-2023.12.01. 까지의 데이터")
    st.dataframe(df, height=400)
    st.divider()
    st.subheader("데이터 수집 방법")
    st.write("1. 초음파 센서의 최대 측정 거리를 1m 로 설정")
    st.write("2. 1m보다 작게 되면 그때의 거리와 시간을 csv파일에 기록")
    st.write("3. 일별로 정리하여 일별 방문자 카운트 종합 및 분석")

#산점도 그래프 부분 
with tab2:
    st.subheader("수집된 일자의 데이터만 표현")
    fig2 = plt.figure() # 여기부터 그래프영역
    plt.scatter(data = df, x = 'mday', y ='dis')
    plt.title('Visit Counting Scatter')
    plt.xticks(df['mday'].unique())    
    plt.yticks(np.arange(0,101,10))
    plt.xlabel('2023.11.27.-2023.12.01.')
    plt.ylabel('Visit Distance')
    st.pyplot(fig2)

    st.divider() #경계선 표시

    st.subheader("모든 기간을 파악할 수 있는 그래프")
    fig2 = plt.figure() # 여기부터 그래프영역
    plt.scatter(df['mday'],df['dis'])
    plt.title('Visit Counting Scatter')
    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("Visit Distance")
    plt.xticks(np.arange(0,32,1))     #x축 눈금 설정
    plt.yticks(np.arange(0,101,10))   #y축 눈금 설정
    plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')
    st.pyplot(fig2)

#막대 그래프 부분
with tab3:
    
    st.subheader("데이터 막대 그래프로 표현")
    fig3 = plt.figure() # 여기부터 그래프영역
    plt.bar(sorted(df["mday"].unique()), df.groupby('mday')['dis'].count(), color="green")
    plt.title('Visit Counting Bar')
    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("Visit Counting")
    plt.xticks(np.arange(1,32,1))     #x축 눈금 설정
    plt.yticks(np.arange(0,101,10))   #y축 눈금 설정
    #x축의 데이터 설정 lenth는 눈금의 길이, pad는 눈금과 레이블의 거리, labelsize는 레이블의 크기
    plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')

    st.pyplot(fig3)
    
    st.divider() #경계선 표시

    st.subheader("횟수를 볼 수 있는 막대그래프")
    plt.bar(sorted(df["mday"].unique()), df.groupby('mday')['dis'].count(), color="green")
    plt.title('Visit Counting Bar')
    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("Visit Counting")
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

#점그래프 (추후 업데이트)
with tab4:
    st.subheader("점 그래프 표현")
    fig4 = plt.figure() # 여기부터 그래프영역
    plt.plot(sorted(df["mday"].unique()), df.groupby('mday')['dis'].count(), "bs")
    plt.title('Visit Counting Dot')
    plt.xticks(np.arange(1,32,1))     #x축 눈금 설정
    plt.yticks(np.arange(0,101,10))   #y축 눈금 설정

    plt.xlabel("2023.11.27.-2023.12.01.")
    plt.ylabel("Visit Counting")

    plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')
    st.pyplot(fig4)

#소스코드 소개 부분
with tab5:
    code0='''
plt.scatter(df['mday'],df['dis'])
plt.xticks(df['mday'].unique())     #x축 눈금 설정
plt.yticks(np.arange(0,101,10))   #y축 눈금 설정

#x축의 데이터 설정 lenth는 눈금의 길이, pad는 눈금과 레이블의 거리, labelsize는 레이블의 크기
plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')
plt.show()

===경계선===
            
plt.scatter(df['mday'],df['dis'])
plt.xlabel("2023.11.27.-2023.12.01.")
plt.ylabel("visit counting")
plt.xticks(np.arange(0,32,1))     #x축 눈금 설정
plt.yticks(np.arange(0,101,10))   #y축 눈금 설정

plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')
plt.show()

===경계선===
            
plt.bar(sorted(df["mday"].unique()), df.groupby('mday')['dis'].count(), color="green")
plt.xlabel("2023.11.27.-2023.12.01.")
plt.ylabel("visit counting")
plt.xticks(np.arange(1,32,1))     #x축 눈금 설정
plt.yticks(np.arange(0,101,10))   #y축 눈금 설정

#x축의 데이터 설정 lenth는 눈금의 길이, pad는 눈금과 레이블의 거리, labelsize는 레이블의 크기
plt.tick_params(axis='x', direction='in', length=2, pad=2, labelsize=7, width=2, labelcolor='red')
plt.show()

===경계선===

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

plt.show()
'''
    st.code(code0, language='python')

st.divider() #경계선 표시


    code1='''
from picozero import DistanceSensor
from time import sleep, localtime

#localtime() 의 튜플 값은
#(연도, 달, 날, 시간, 분, 초, 요일[일~토/0~6], 1월1일부터 경과한 일 수)
start_time=localtime()
print(start_time)

ds = DistanceSensor(echo=20, trigger=21) #거리측정 코드

f = open('test030.csv', 'w') #피코 파일 경로랑 같이 있어야함
f.write(f'{start_time}\n')
#1시간 측정을 기준
for i in range(3600):
    dscm=ds.distance * 100
    print(dscm)
    sleep(1)
    if dscm<100:
        real_time=localtime()
        data=str(dscm)
        f.write("%s" f'{real_time}\n' %data) 
    else:
        continue

end_time=localtime()
f.write(f'{end_time}\n')
f.close()
'''
    st.code(code1, language='python')
