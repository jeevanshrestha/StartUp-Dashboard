import re
import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(layout='wide')


def load_investor_details(investor):
    df_investor = df[df['investors'].str.contains(investor)].head()

    st.subheader('Latest 5 investments by ' + investor)
    st.dataframe(df_investor)

    big_investment = df[df['investors'].str.contains(investor)].groupby('startup')['amount'].sum().sort_values(
        ascending=False).head().reset_index(name='amount')
    st.subheader('Biggest 5 investments by ' + investor)
    # plot for india
    fig = px.bar(big_investment, x='startup', y='amount', color='startup')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Top 5 investments sectors by ' + investor)
    verticals = df[df['investors'].str.contains(investor)].groupby('vertical')['amount'].sum().sort_values(
        ascending=False).head().reset_index(name='amount')
    # plot for india
    fig = px.pie(verticals, color='vertical', values='amount')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Year on year investment by ' + investor)
    yoy = df[df['investors'].str.contains(investor)].groupby('year')['amount'].sum().reset_index(name='amount')
    # plot for india
    fig = px.area(yoy, x='year', y='amount', markers='0')
    st.plotly_chart(fig, use_container_width=False)

    X = df[['round', 'vertical', 'subvertical', 'city', 'amount']]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['amount']),
            ('cat', OneHotEncoder(), ['round', 'vertical', 'subvertical', 'city'])
        ])

    # Update the KNN pipeline to include preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=5, metric='euclidean'))
    ])

    st.write(pipeline)

    # Fit the pipeline on the entire dataset
    pipeline.fit(df.drop(columns=['investors']))

    # Find similar investors for a given investor
    # Set the target investor's name

    # Find the row corresponding to the target investor's profile in the dataset
    # Find the row corresponding to the target investor
    # Step 1: Fit the pipeline on the full dataset (excluding the 'investors' column)
    pipeline.fit(df.drop(columns=['investors']))

    # Step 2: Retrieve the profile for the selected investor
    investor_profile_row = df[df['investors'].str.contains(investor, case=False, na=False)]

    if not investor_profile_row.empty:
        # Selecting the first match
        investor_profile = investor_profile_row.drop(columns=['investors']).iloc[0:1]

        # Step 3: Transform the investor profile with the preprocessor
        investor_profile_transformed = pipeline['preprocessor'].transform(investor_profile)

        # Step 4: Find similar investors
        distances, indices = pipeline['knn'].kneighbors(investor_profile_transformed)

        # Step 5: Display the results
        similar_investors = df.iloc[indices[0]]
        st.write(f"Similar investors to {investor}:")

        for i, (index, distance) in enumerate(zip(indices[0], distances[0])):
            investor_info = similar_investors.iloc[i]
            st.write(f"Investor {i + 1}:")
            st.write(investor_info)
            st.write(f"Distance: {distance}")
            st.write("-" * 30)
    else:
        st.write(f"Investor '{investor}' not found in the dataset.")


def load_overall_analysis():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total = round(df['amount'].sum())
        st.metric('Total Funding ', 'Rs ' + str(total) + " Cr")

    with col2:
        st.write('Maximum Funding')
        maximum_per_startup = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).reset_index()
        st.metric( maximum_per_startup['startup'].loc[0], maximum_per_startup['amount'].loc[0])

    with col3:
        average_funding = round(df.groupby('startup')['amount'].sum().mean())
        st.metric('Average Funding ', 'Rs ' + str(average_funding) + " Cr")

    with col4:
        average_funding = df['startup'].nunique()
        st.metric('Total Startups ', str(average_funding))


    monthly_type = st.selectbox(' Select Monthly Analysis ', ['Total Amount', 'StartUp Count'])

    if monthly_type == 'Total Amount':
        monthly_analysis = df.groupby(['month_year'])['amount'].sum().reset_index()
        fig2 = px.line(monthly_analysis, x='month_year', y='amount', title='Startup funding trend analysis per Month',
                       labels={'month_year': 'Month', 'amount': 'Amount'}, markers='0')
        st.plotly_chart(fig2, use_container_width=True)
    elif monthly_type == 'StartUp Count':
        monthly_analysis = df.groupby(['month_year'])['amount'].count().reset_index(name = 'count')
        fig2 = px.line(monthly_analysis, x='month_year', y='count', title='Startup trend analysis per Month',
                       labels={'month_year': 'Month', 'count': 'Numbers of Startup'}, markers='0')
        st.plotly_chart(fig2, use_container_width=True)

    else:
        pass

    top_10_fund_raiser = df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(
        10).reset_index()
    fig3 = px.bar(top_10_fund_raiser, x='startup', y='amount', color='startup',
           labels={'startup': 'Startup Name', 'amount': 'Amount in Crores'},
           title='Top 10 Fund Raiser Company In India')
    st.plotly_chart(fig3, use_container_width=True)

    top_5_ecommerce_startup = df[df['vertical'] == 'E-Commerce'].groupby('startup')[
        'amount'].sum().sort_values(ascending=False).head().reset_index()

    fig4 = px.bar(top_5_ecommerce_startup, y='amount', x='startup',color='amount',
                  labels={'amount':'Amount in Crores','startup':'StartUp Company'},
           title='Top 5 E-commerce Fund Raiser Startups in India')
    st.plotly_chart(fig4, use_container_width=True)

    temp_df = df[~ df['city'].isnull()]
    # Top_5_Cities_by_Startups = df.groupby('City').count()
    Top_5_Cities_by_Startups = temp_df.groupby('city').size().sort_values(ascending=False).head().reset_index(
        name='Count')

    fig5 = px.bar(Top_5_Cities_by_Startups,x='city', y='Count', color='city', title='Top 5 Startup Friendly Cities in India', text_auto=True)

    st.plotly_chart(fig5, use_container_width=True)

    df['vertical'] = df['vertical'].replace('nan', np.nan)
    df.fillna({'vertical': 'Miscelleneous'}, inplace=True)
    Investment_sectors = df.groupby('vertical').size().sort_values(ascending=False).head(10).reset_index(
        name='Count')
    fig6 = px.bar(Investment_sectors, x='vertical', y='Count', color='Count', title='Top 10 Investors Choice Market',labels={'vertical':'StartUp Sector'})
    st.plotly_chart(fig6, use_container_width=True)


def load_startup_analysis(startup):
    st.subheader(startup)

    startup_df =  df[df['startup']==startup][['investors','amount']]
    total = startup_df['amount'].sum()
    st.metric('Total Funding', str(total)+ ' Cr')
    st.dataframe(startup_df)
    fig7 = px.pie(startup_df, values='amount', color='investors',title='Investment Shares Analysis')
    st.plotly_chart(fig7, use_container_width=True)

def clean_string(x):
    # Define the regex pattern
    pattern = r'\\\\n|\\\\xc2|\\\\xa0|\\\\xc3|\\\\xa9|\\\\xe2|\\\\x80|\\\\x99|\\\\xe2|\\\\x80|\\\\x93\\\\xc3|\\\\xa9|\\\\xe2|\\\\x80|\\\\x99|\\xe2|\\x80|\\x93'
    # Apply the pattern removal using re.sub
    return re.sub(pattern, '', str(x))



df = pd.read_csv('startup_funding.csv')

# Apply the function to each specified column
for col in ['date','startup','vertical','subvertical','city','investors','round','amount']:
    df[col] = df[col].apply(lambda x: clean_string(x))

df.loc[df['date'] == '01/07/015', 'date'] = '01/07/2015'
df.loc[df['date'] == '\\\\xc2\\\\xa010/7/2015', 'Date'] = '10/07/2015'
df.loc[df['date'] == '12/05.2015', 'date'] = '12/05/2015'
df.loc[df['date'] == '13/04.2015', 'date'] = '13/04/2015'
df.loc[df['date'] == '15/01.2015', 'date'] = '15/01/2015'
df.loc[df['date'] == '22/01//2015', 'date'] = '22/01/2015'
df.loc[df['date'] == '05/072018', 'date'] = '05/07/2018'

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_year'] = pd.to_datetime(df['month'].astype(str)+ '/' + df['year'].astype(str))

df.fillna({'amount':'0'}, inplace=True)
df['amount']=df['amount'].str.replace('\\\\xc2\\\\xa0','')
df['amount']=df['amount'].str.replace(',','')
df['amount']=df['amount'].str.replace('+','')
df['amount']=df['amount'].str.replace('N/A','0')
df['amount']=df['amount'].str.replace('Undisclosed','0')
df['amount']=df['amount'].str.replace('undisclosed','0')
df['amount']=df['amount'].str.replace('unknown','0')

df['amount'] = df['amount'].astype('float64')


df.loc[df['startup'].str.contains('Flipkart', case=False, na=False), 'startup'] = 'Flipkart'
df.loc[df['startup'].str.contains('Ola', case=False, na=False), 'startup'] = 'Ola'
df.loc[df['startup'].str.contains('Byju', case=False, na=False), 'startup'] = 'Byju'
df.loc[df['startup'].str.contains('Paytm', case=False, na=False), 'StartupName'] = 'Paytm'

df.replace(['ECommerce', 'eCommerce', 'Ecommerce', 'E-commerce'],'E-Commerce', inplace=True)
df.replace(['E-Tech', 'EdTech'],'Ed-Tech', inplace=True)
df.replace('FinTech','Fin-Tech', inplace=True)

df['city']=df['city'].str.replace('Mumbai/Bengaluru','Mumbai')
df.loc[df['city'].str.contains('Mumbai', case=False, na=False), 'city'] = 'Mumbai'
df.loc[df['city'].str.contains('Delhi', case=False, na=False), 'city'] = 'New Delhi'
df.loc[df['city'].str.contains('Pune', case=False, na=False), 'city'] = 'Pune'
df.loc[df['city'].str.contains('Bengaluru', case=False, na=False), 'city'] = 'Bangalore'
df.loc[df['city'].str.contains('Bangalore', case=False, na=False), 'city'] = 'Bangalore'
df.loc[df['city'].str.contains('India', case=False, na=False), 'city'] = 'India'
df.loc[df['city'].str.contains('Goa', case=False, na=False), 'city'] = 'Goa'
df.loc[df['city'].str.contains('Noida', case=False, na=False), 'city'] = 'Noida'
df.loc[df['city'].str.contains('Chennai', case=False, na=False), 'city'] = 'Chennai'
df.loc[df['city'].str.contains('Hyderabad', case=False, na=False), 'city'] = 'Hyderabad'
df.loc[df['city'].str.contains('Gurgaon', case=False, na=False), 'city'] = 'Gurgaon'


st.sidebar.title('Startup Funding Analysis')

option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'StartUp', 'Investor'])


if option == 'Overall Analysis':
    # load_overall_analysis()
    st.title(option)
    load_overall_analysis()

elif option == 'StartUp':
    selected_startup = st.sidebar.selectbox('Select StartUp', sorted(df['startup'].unique().tolist()))
    btn1 = st.sidebar.button('Find StartUp Details')
    st.title('StartUp Analysis')
    if btn1:
        load_startup_analysis(selected_startup)
else:
    selected_investor = st.sidebar.selectbox('Select StartUp', sorted(set(df['investors'].str.split(',').sum())))
    st.title('Investors Analysis')
    btn2 = st.sidebar.button('Find Investor Details')
    if btn2:
        load_investor_details(selected_investor)
