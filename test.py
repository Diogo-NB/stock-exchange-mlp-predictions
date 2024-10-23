import pandas
import numpy as np

df_stocks_types = {
    'ABERTURA': np.float64,
    'FECHAMENTO': np.float64,
}

df_selic_types = {
    'valor': np.float64,
}

df_stocks = pandas.read_csv('IRB_Brasil_RE_IRBR3.csv', index_col='DATA' , dtype=df_stocks_types, decimal=',')

print(df_stocks)

df_selic = pandas.read_csv('bcdata_sgs_selic.csv', sep=';', index_col='data', dtype=df_selic_types , decimal=',')

print(df_selic)