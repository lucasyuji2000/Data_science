import pandas as pd
import numpy as np
import streamlit as st
import folium
import plotly.express as px

from datetime import time, datetime
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# __Page config__ #
st.set_page_config(page_title='App_house', layout='wide')


# Functions
@st.cache(allow_output_mutation=True)
def get_data(file_path):
    df = pd.read_csv(file_path)
    return df


def create_filter(df):
    attributes_filter = st.sidebar.multiselect('Enter columns', df.columns)
    zipcode_filter = st.sidebar.multiselect('Enter Zipcodes', df['zipcode'].unique())

    if zipcode_filter != [] and attributes_filter != []:
        df = df.loc[df['zipcode'].isin(zipcode_filter), attributes_filter]
    elif zipcode_filter != [] and attributes_filter == []:
        df = df.loc[df['zipcode'].isin(zipcode_filter), :]
    elif zipcode_filter == [] and attributes_filter != []:
        df = df.loc[:, attributes_filter]
    else:
        df = df.copy()
    return df


#
    # Average metrics
    df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(df3, df4, on='zipcode', how='inner')
    df_metrics = pd.merge(m1, m2, on='zipcode', how='inner')

    df_metrics.columns = ['zipcode', 'total_houses', 'price_mean', 'sqft_living_mean', 'price_m2_mean']

    # Descritive statistics
    num_attributes = df.select_dtypes(include=['int64', 'float64'])

    # Define decimals
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    # Central tendency - mean, median
    df_mean = pd.DataFrame(num_attributes.apply(np.mean, axis=0))
    df_median = pd.DataFrame(num_attributes.apply(np.median, axis=0))

    # Dispersion - std, min, max
    df_std = pd.DataFrame(num_attributes.apply(np.std, axis=0))
    df_min = pd.DataFrame(num_attributes.apply(np.min, axis=0))
    df_max = pd.DataFrame(num_attributes.apply(np.max, axis=0))

    df_stats = pd.concat([df_max, df_min, df_mean, df_median, df_std], axis=1).reset_index()
    df_stats.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    return df_stats, df_metrics


# __Extract data__ #
file_path = r'C:\Users\lucas\Estudos\_CDS\repos\python_zero_ds\data\kc_house_data.csv'
df = get_data(file_path)


# __Transform data__ #
df['price_m2'] = df['price'] / (df['sqft_lot'] * 0.093)
#df['date'] = pd.to_datetime(df['date'], '%Y-%m-%d')
df = create_filter(df)
#df1, df2 = data_analysis(df)


# __Load data__ #
st.title('Data Overview')
st.dataframe(df)
col1, col2 = st.columns(2)
#col1.dataframe(df1)
#col2.dataframe(df2)
