import pandas as pd
import matplotlib.pyplot as plt

total_pop = 58.5 * 10 ** 6
expose_rate = 0.1
sigma = 0.1
df = pd.read_csv('data/time-series-19-covid-combined.csv')
df_hb = df[(df['Country/Region'] == 'China') & (df['Province/State'] == 'Hubei')]
del df_hb['Country/Region']
del df_hb['Province/State']
del df_hb['Lat']
del df_hb['Long']
df_hb['add_confirm'] = df['Confirmed'].diff(1)
df_hb['add_recover'] = df['Recovered'].diff(1)
df_hb['add_deaths'] = df['Deaths'].diff(1)
df_hb['exsit_exposed'] = df_hb['add_confirm'] / sigma
df_hb['add_exposed'] = df_hb['exsit_exposed'].diff(1) + df_hb['add_confirm']
df_hb['exsit_confirm'] = df['Confirmed'] - df['Recovered'] - df['Deaths']
df_hb['beta'] = df_hb['add_exposed'] / total_pop
df_hb['sigma'] = 0.1
df_hb['gamma'] = (df_hb['add_recover'] + df_hb['add_deaths']) / df_hb['exsit_confirm']
df_hb.to_csv('data/data_hb.csv', index=None)
# plt.plot(df_city['cases'])
# plt.plot(df_city['deaths'])
# plt.show()
