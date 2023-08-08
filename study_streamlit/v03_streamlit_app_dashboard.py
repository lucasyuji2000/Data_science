# __Libraries__ #
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px 


# __Page settings__ #
st.set_page_config(page_title='dashboard_kc_house_data', layout='wide')


# __Functions__ #
@st.cache( allow_output_mutation=True )
def get_data(path):
    df = pd.read_csv(path)
    return df


def set_attributes(df):
    df['price_m2'] = df['price'] / df['sqft_lot']
    return df


def data_overview(df):
    df_aux = df.copy()
    f_attributes = st.sidebar.multiselect('Enter columns', df.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', df['zipcode'].unique())

    st.title('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        df = df.loc[df['zipcode'].isin( f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        df = df.loc[df['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        df = df.loc[:, f_attributes]

    else:
        df = df.copy()

    st.dataframe(df)

    c1, c2 = st.columns((1, 1))

    # Average metrics
    df1 = df_aux[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df_aux[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = df_aux[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = df_aux[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(df3, df4, on='zipcode', how='inner')
    df1 = pd.merge(m1, m2, on='zipcode', how='inner')

    df1.columns = ['zipcode', 'total_houses', 'price_mean', 'sqft_living_mean', 'price/m2_mean']

    c1.header('Average Values')
    c1.dataframe(df1, height=620)

    # Statistic Descriptive
    num_attributes = df_aux.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df2 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df2.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    c2.header('Descriptive Analysis')
    c2.dataframe(df2, height=800)
    return None


def set_commercial(df):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # --Average Price per year built-- #
    # setup filters
    min_year_built = int(df['yr_built'].min())
    max_year_built = int(df['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

    st.header('Average price per year built')

    # get data
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    df1 = df.loc[df['yr_built'] < f_year_built]
    df1 = df1[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(df1, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # --Average Price per day-- #
    st.header('Average Price per day')
    st.sidebar.subheader('Select Max Date')

    # setup filters
    min_date = datetime.strptime(df['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(df['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # filter df
    df['date'] = pd.to_datetime(df['date'])
    df2 = df[df['date'] < f_date]
    df2 = df2[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(df2, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # --Histogram-- #
    st.header('Price Distribuition')
    st.sidebar.subheader('Select Max Price')

    # filters
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    avg_price = int(df['price'].mean())

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

    df3 = df[df['price'] < f_price]

    fig = px.histogram(df3, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    return None


if __name__ == "__main__":
    # ETL
    path = r'C:\Users\lucas\Estudos\_CDS\repos\python_zero_ds\data\kc_house_data.csv'

    # load data
    df = get_data(path)

    # transform data
    df = set_attributes(df)

    data_overview(df)

    set_commercial(df)
    



