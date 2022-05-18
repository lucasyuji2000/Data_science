# -- Libraries -- #
import pandas as pd

def get_data(path):
    df = pd.read_csv(rf'{path}')
    df['date'] = pd.to_datetime(df['date'])
    return df


# Load Data
df = get_data(r'C:\Users\lucas\Estudos\_CDS\repos\python_zero_ds\data\kc_house_data.csv')


print(df['id'].min())