import pandas as pd
import numpy as np

Data = pd.read_csv("random_sample.csv")


Data['weekType'] = np.where((Data['weekday']=='sunday') | (Data['weekday'] == 'saturday'), 'weekend', 'not-weekend')


Data.loc[Data['hr'] <=5,'timeslot'] = 'time1'
Data.loc[(Data['hr'] >=6) & (Data['hr'] <= 11), 'timeslot'] = 'time2'
Data.loc[(Data['hr'] >=12) & (Data['hr'] <= 17), 'timeslot'] = 'time3'
Data.loc[(Data['hr'] >=18) & (Data['hr'] <= 23), 'timeslot'] = 'time4'





grouped = Data.groupby(['weekType','timeslot'])


for name, group in grouped:
    group.to_csv('{}.csv'.format(name))

