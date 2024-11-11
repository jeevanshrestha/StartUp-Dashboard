import time

import streamlit as st
import numpy as np
import pandas as pd

st.title('Startup Dashboard')

st.header("I am learning Streamlit")

st.subheader('I am loving it.')

st.write ("This is a normal p txt")

st.markdown("""
### Markdown Header 3
- Race 3
- Humsakals
- Housefull
""")

st.code("""
def foo(input):
    return  foo**23
x = foo(2)
""")


st.latex('x^2 - y^2 + 2 = 0')

df = pd.DataFrame({

    'name':['Nitesh','Ankit'],
    'marks':[10,20]
})

st.dataframe(df)

st.metric('Revenue', 'Rs 3L', '3%')

st.json({

    'name':['Nitesh','Ankit'],
    'marks':[10,20]
})



st.sidebar.title('Sidebar Title')

col1, col2 = st.columns(2)
with col1:
    st.image('_CS_6586.jpg')

with col2:
    st.image('_CS_6586.jpg')


st.success('Success message')
st.error('error message')
st.info('info message')
st.warning('warning message')

progress_text = "Operation in progress. Please wait."
bar = st.progress(0, text=progress_text)

# for i in range(1,101):
#     time.sleep(0.1)
#     bar.progress(i, text=progress_text)

email = st.text_input('Enter Email :')
age = st.number_input('Enter age:', step =1)

date = st.date_input("Enter date of Birth:", format='DD/MM/YYYY')



username = st.text_input('Enter usernam')
password = st.text_input('Enter password')
gender = st.selectbox('Select Gender',['male','female','others'])

btn = st.button('Login')

if btn:
    if username =='jeevanshrestha' and password =='124':
        st.balloons()
        st.success('Login Successful')
        st.write(gender)
    else:
        st.error('Login Failed')



file = st.file_uploader('Upload a csv file')

if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df.describe())