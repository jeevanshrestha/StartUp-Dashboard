import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


df = pd.read_csv('startup_funding.csv')

st.sidebar.title('Startup Funding Analysis')

option = st.sidebar.selectbox('Select One',['Overall Analysis','StartUp','Investor'])

if option == 'Overall Analysis':
    # load_overall_analysis()
    st.title(option)

elif option == 'StartUp':
    st.sidebar.selectbox('Select StartUp',sorted(df['startup'].unique().tolist()))
    btn1 = st.sidebar.button('Find StartUp Details')
    st.title('StartUp Analysis')
else:
    selected_investor = st.sidebar.selectbox('Select StartUp',sorted(set(df['investors'].str.split(',').sum())))
    st.title(selected_investor)
    # btn2 = st.sidebar.button('Find Investor Details')
    # if btn2:
    #     load_investor_details(selected_investor)
