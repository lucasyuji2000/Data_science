# __Libraries__ #
import pandas as pd
import streamlit as st
import numpy as np
import folium

# __Functions__ #
@st.cache(allow_output_mutation=True) # le o arquivo da memoria, e nao do disco toda vez que rodar
def get_data(path):
    df = pd.read_csv(rf'{path}')
    return df


def show_shape(df):
    st.write(f'Number of rows: {df.shape[0]} \ Number of columns: {df.shape[1]}')
    return None


def descriptive_analysis(df):
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

    df1 = pd.concat([df_max, df_min, df_mean, df_median, df_std], axis=1).reset_index()
    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']
    return df1


def group_zip(df):

    # Average metrics
    df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge DataFrames
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(df3, df4, on='zipcode', how='inner')
    df = pd.merge(m1, m2, on='zipcode', how='inner')
    df.columns = ['zipcode', 'total_houses', 'price', 'sqft_living', 'price/M²']
    return df

def page1(df):
    st.title('Data Overview')
    c1, c2 = st.columns(2)

    # Creating filters
    with c1:
        zipcode_filter = st.multiselect('Enter Zipcodes', df['zipcode'].unique())
    with c2:
        attributes_filter = st.multiselect('Enter columns', df.columns)

    # Table conditions
    if zipcode_filter != [] and attributes_filter != []:
        df = df.loc[df['zipcode'].isin(zipcode_filter), attributes_filter]
    elif zipcode_filter != [] and attributes_filter == []:
        df = df.loc[df['zipcode'].isin(zipcode_filter), :]
    elif zipcode_filter == [] and attributes_filter != []:
        df = df.loc[:, attributes_filter]
    else:
        df = df.copy()

    # Show data
    st.write(df)
    show_shape(df)
    return df


def page2(df):
    st.title('Data Overview2')
    c1, c2 = st.columns((1, 2))

    # Get data
    df = group_zip(df)

    # Creating filters
    with c1:
        zipcode_filter = st.multiselect('Enter Zipcodes', df['zipcode'].unique())
        attributes_filter = st.multiselect('Enter columns', df.columns)

    # Table conditions
    if zipcode_filter != [] and attributes_filter != []:
        df = df.loc[df['zipcode'].isin(zipcode_filter), attributes_filter]
    elif zipcode_filter != [] and attributes_filter == []:
        df = df.loc[df['zipcode'].isin(zipcode_filter), :]
    elif zipcode_filter == [] and attributes_filter != []:
        df = df.loc[:, attributes_filter]
    else:
        df = df.copy()

    # Show data
    with c2:
        st.dataframe(df)
        show_shape(df)
    return df



def page3(df):
    st.title('Descriptive Analysis')
    c1, c2 = st.columns(2)

    # Get data
    df = descriptive_analysis(df)

    # Creating filters
    with c1:
        attr_filter = st.multiselect('Enter attributes', df['attributes'].unique())
    with c2:
        columns_filter = st.multiselect('Enter columns', df.columns)

    # Table conditions
    if attr_filter != [] and columns_filter != []:
        df = df.loc[df['attributes'].isin(attr_filter), columns_filter]
    elif attr_filter != [] and columns_filter == []:
        df = df.loc[df['attributes'].isin(attr_filter), :]
    elif attr_filter == [] and columns_filter != []:
        df = df.loc[:, columns_filter]
    else:
        df = df.copy()

    # Show data
    st.dataframe(df)
    show_shape(df)
    return df


# __page settings__ #
st.set_page_config(page_title='test1', layout='wide')


# __Extract data__ #

# __get data__ #
file_path = r'C:\Users\lucas\Estudos\_CDS\repos\python_zero_ds\data\kc_house_data.csv'
df = get_data(file_path)


# __Transform data__ #
df['price_m2'] = df['price'] / (df['sqft_lot'] * 0.093)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
# __page built__ #

# page 1
df_page1 = page1(df)
st.markdown('''---''')

# page 2
df_page2 = page2(df)
st.markdown('''---''')

# page 3
df_page3 = page3(df)





