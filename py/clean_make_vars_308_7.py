#!/usr/bin/env python
# coding: utf-8

import calendar
import datetime as dt
from timeit import default_timer as timer

import numpy as np
import pandas as pd

start_time = timer()

df = pd.read_csv('applications data.csv')

df.date = pd.to_datetime(df.date, format='%Y%m%d')

# ========== Clean frivolous fields ==========

df.loc[df.ssn == 999999999, 'ssn'] = df.loc[df.ssn == 999999999, 'record']
df.loc[df.address == '123 MAIN ST', 'address'] = df.loc[df.address == '123 MAIN ST', 'record']
df.loc[df.homephone == 9999999999, 'homephone'] = df.loc[df.homephone == 9999999999, 'record']
df.loc[df.dob == 19070626, 'dob'] = df.loc[df.dob == 19070626, 'record']

df.ssn = df.ssn.astype(str)
df.zip5 = df.zip5.astype(str)
df.dob = df.dob.astype(str)
df.homephone = df.homephone.astype(str)
df.address = df.address.astype(str)

# add leading 0 to zips
df['zip5'] = df.zip5.apply(lambda x: x if len(x) == 5 else '0' * (5 - len(x)) + x)

# ========== Risk table for day of week ==========

df['dow'] = df['date'].apply(lambda x: calendar.day_name[x.weekday()])
train_test = df[df['date'] < '2016-11-01']

# do statistical smoothing
c = 4
nmid = 20
y_avg = train_test['fraud_label'].mean()
y_dow = train_test.groupby('dow')['fraud_label'].mean()
num = train_test.groupby('dow').size()
y_dow_smooth = y_avg + (y_dow - y_avg) / (1 + np.exp(-(num - nmid) / c))
df['dow_risk'] = df['dow'].map(y_dow_smooth)

# ========== Create entities ==========

df['name'] = df.firstname + df.lastname
df['fulladdress'] = df.address + df.zip5
df['name_dob'] = df.name + df.dob
df['name_fulladdress'] = df.name + df.fulladdress
df['name_homephone'] = df.name + df.homephone
df['fulladdress_dob'] = df.fulladdress + df.dob
df['fulladdress_homephone'] = df.fulladdress + df.homephone
df['dob_homephone'] = df.dob + df.homephone
df['homephone_name_dob'] = df.homephone + df.name_dob

for field in list(df.iloc[:, np.r_[3:9, 12:15]].columns):
    df['ssn_' + field] = df['ssn'] + df[field]

# ========== Velocity + Day since ==========

attributes = list(df.iloc[:, np.r_[2, 5, 7, 8, 12:30]].columns)

df1 = df.copy()
final = df.copy()
df1['check_date'] = df1.date
df1['check_record'] = df1.record

start = timer()

for entity in attributes:
    st = timer()

    df_l = df1[['record', 'date', entity]]
    df_r = df1[['check_record', 'check_date', entity]]
    temp = pd.merge(df_l, df_r, left_on=entity, right_on=entity)

    # day since
    day_since_df = temp[temp.record > temp.check_record][['record', 'date', 'check_date']] \
        .groupby('record')[['date', 'check_date']].last()
    mapper = (day_since_df.date - day_since_df.check_date).dt.days
    final[entity + '_day_since'] = final.record.map(mapper)
    final[entity + '_day_since'].fillna((final.date - pd.to_datetime('2016-01-01')).dt.days,
                                        inplace=True)
    print(f'\n{entity}_day_since ---> Done')

    # velocity
    for offset_t in [0, 1, 3, 7, 14, 30]:
        count_day_df = temp[(temp.check_date >= (temp.date - dt.timedelta(offset_t)))
                            & (temp.record >= temp.check_record)]
        col_name = f'{entity}_count_{offset_t}'
        mapper2 = count_day_df.groupby('record')[entity].count()
        final[col_name] = final.record.map(mapper2)

        print(f'{entity}_count_{str(offset_t)} ---> Done')

    print(f'Run time for the last entity ----------------- {timer() - st}s')

print(f'Total run time: {(timer() - start) / 60}min')

# ========== Relative Velocity ==========

start = timer()
for att in attributes:
    for d in ['0', '1']:
        for dd in ['3', '7', '14', '30']:
            final[att + '_count_' + d + '_by_' + dd] \
                = final[att + '_count_' + d] / (final[att + '_count_' + dd] / float(dd))
print(f'Total run time: {timer() - start}s')

# ========== Keep desired variables ==========

final.set_index('record', inplace=True)

final = final.iloc[:, np.r_[8, 10, 29:337]]

final.to_csv('vars_308.csv')

print('Duration: ', (timer() - start_time) / 60, ' min')
