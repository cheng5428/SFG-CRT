import numpy as np
import pandas as pd
import os
import sys
import zipfile
import subprocess

from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from tqdm.notebook import tqdm
from copy import deepcopy

import json

DATASET = 'ml-1m' 
RAW_PATH = os.path.join('./', DATASET)

RANDOM_SEED = 0
NEG_ITEMS = 99

if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH, exist_ok=True)
    
if not os.path.exists(os.path.join(RAW_PATH, DATASET + '.zip')):
    print('Downloading data into ' + RAW_PATH)
    import urllib.request
    url = f'http://files.grouplens.org/datasets/movielens/{DATASET}.zip'
    zip_path = os.path.join(RAW_PATH, DATASET + '.zip')
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print('Download completed!')
    except Exception as e:
        print(f'Download failed: {e}')
        raise
    
    print('Unzip files...')
    f = zipfile.ZipFile(zip_path, 'r') 
    for file in f.namelist():
        print("Extract %s" % (file))
        f.extract(file, '.') # 修改此处：解压到当前目录，避免双重嵌套
    f.close()
    print('Extraction completed!')
else:
    print('Data already exists')

# read interaction data
interactions = []
user_freq, item_freq = dict(), dict()
file = os.path.join(RAW_PATH,"ratings.dat")
with open(file) as F:
    header = 0
    for line in tqdm(F):
        if header == 1:
            header = 0
            continue
        line = line.strip().split("::")
        uid, iid, rating, time = line[0], line[1], float(line[2]), float(line[3])
        if rating >= 4:
            label = 1
        else:
            label = 0
        interactions.append([uid,time,iid,label])
        if int(label)==1:
            user_freq[uid] = user_freq.get(uid,0)+1
            item_freq[iid] = item_freq.get(iid,0)+1

# 5-core filtering
select_uid, select_iid = [],[]
while len(select_uid)<len(user_freq) or len(select_iid)<len(item_freq):
    select_uid, select_iid = [],[]
    for u in user_freq:
        if user_freq[u]>=5:
            select_uid.append(u)
    for i in item_freq:
        if item_freq[i]>=5:
            select_iid.append(i)
    print("User: %d/%d, Item: %d/%d"%(len(select_uid),len(user_freq),len(select_iid),len(item_freq)))

    select_uid = set(select_uid)
    select_iid = set(select_iid)
    user_freq, item_freq = dict(), dict()
    interactions_5core = []
    for line in tqdm(interactions):
        uid, iid, label = line[0], line[2], line[-1]
        if uid in select_uid and iid in select_iid:
            interactions_5core.append(line)
            if int(label)==1:
                user_freq[uid] = user_freq.get(uid,0)+1
                item_freq[iid] = item_freq.get(iid,0)+1
    interactions = interactions_5core

print("Selected Interactions: %d, Users: %d, Items: %d"%(len(interactions),len(select_uid),len(select_iid)))

# Get timestamp
ts = []
for i in tqdm(range(len(interactions))):
    ts.append(datetime.fromtimestamp(interactions[i][1]))

# Construct and Save 5 core results with situation context
interaction_df = pd.DataFrame(interactions,columns = ["user_id","time","news_id","label"])
interaction_df['timestamp'] = ts
interaction_df['hour'] = interaction_df['timestamp'].apply(lambda x: x.hour)
interaction_df['weekday'] = interaction_df['timestamp'].apply(lambda x: x.weekday())
interaction_df['date'] = interaction_df['timestamp'].apply(lambda x: x.date())

def get_time_range(hour): # according to the Britannica dictionary
    # https://www.britannica.com/dictionary/eb/qa/parts-of-the-day-early-morning-late-morning-etc
    if hour>=5 and hour<=8:
        return 0
    if hour>8 and hour<11:
        return 1
    if hour>=11 and hour<=12:
        return 2
    if hour>12 and hour<=15:
        return 3
    if hour>15 and hour<=17:
        return 4
    if hour>=18 and hour<=19:
        return 5
    if hour>19 and hour<=21:
        return 6
    if hour>21:
        return 7
    return 8 # 0-4 am

interaction_df['period'] = interaction_df.hour.apply(lambda x: get_time_range(x))
min_date = interaction_df.date.min()
interaction_df['day'] = (interaction_df.date - min_date).apply(lambda x: x.days)

interaction_df.to_csv("interaction_5core.csv",index=False)
interaction_df["user_id"] = interaction_df["user_id"].astype(int)
interaction_df["news_id"] = interaction_df["news_id"].astype(int)

CTR_PATH='./ML_1MCTR/'
os.makedirs(CTR_PATH,exist_ok=True)

# copy interaction file, rename and re-id all features
interaction_ctr = interaction_df.copy()
interaction_ctr.rename(columns={'hour':'c_hour_c','weekday':'c_weekday_c','period':'c_period_c','day':'c_day_f',
                              'user_id':'original_user_id'},
                     inplace=True)
user2newid_ctr = dict(zip(sorted(interaction_ctr.original_user_id.unique()), 
                      range(1,interaction_ctr.original_user_id.nunique()+1)))
interaction_ctr['user_id'] = interaction_ctr.original_user_id.apply(lambda x: user2newid_ctr[x])

item2newid_ctr = dict(zip(sorted(interaction_ctr.news_id.unique()), 
                      range(1,interaction_ctr.news_id.nunique()+1)))
interaction_ctr['item_id'] = interaction_ctr['news_id'].apply(lambda x: item2newid_ctr[x])
interaction_ctr.sort_values(by=['user_id','time'],inplace=True)
interaction_ctr = interaction_ctr.reset_index(drop=True)

nu2nid = dict()
ni2nid = dict()
for i in user2newid_ctr.keys():
    oi = int(i)
    nu2nid[oi] = user2newid_ctr[i]

for i in item2newid_ctr.keys():
    oi = int(i)
    ni2nid[oi] = item2newid_ctr[i]

json.dump(nu2nid,open(os.path.join(CTR_PATH,"user2newid.json"),'w'))
json.dump(ni2nid,open(os.path.join(CTR_PATH,"item2newid.json"),'w'))

# Count statistics
for col in interaction_ctr.columns:
    if col in ['user_id','item_id'] or col.startswith('c_'):
        print(col, interaction_ctr[col].nunique())

# split training, validation, and test sets.
split_time1 = interaction_ctr.c_day_f.max() * 0.8
train = interaction_ctr.loc[interaction_ctr.c_day_f<=split_time1].copy()
val_test = interaction_ctr.loc[(interaction_ctr.c_day_f>split_time1)].copy()
split_time2 = interaction_ctr.c_day_f.max() * 0.9
val = val_test.loc[val_test.c_day_f<=split_time2].copy()
test = val_test.loc[val_test.c_day_f>split_time2].copy()

# Delete user&item in validation&test sets that not exist in training set
train_u, train_i = set(train.user_id.unique()), set(train.item_id.unique())
val_sel = val.loc[(val.user_id.isin(train_u))&(val.item_id.isin(train_i))].copy()
test_sel = test.loc[(test.user_id.isin(train_u))&(test.item_id.isin(train_i))].copy()
print("Train user: %d, item: %d"%(len(train_u),len(train_i)))
print("Validation user: %d, item:%d"%(val_sel.user_id.nunique(),val_sel.item_id.nunique()))
print("Test user: %d, item:%d"%(test_sel.user_id.nunique(),test_sel.item_id.nunique()))
train.label.sum(),train.label.mean(),val_sel.label.sum(),val_sel.label.mean(),test_sel.label.sum(),test_sel.label.mean()

# Assign impression ids
for interaction_partial in [train,val_sel,test_sel]:
    interaction_partial['last_user_id'] = interaction_partial['user_id'].shift(1)
    impression_ids = []
    impression_len = 0
    current_impid = 0
    max_imp_len = 20
    for uid, last_uid in tqdm(interaction_partial[['user_id','last_user_id']].to_numpy()):
        if uid == last_uid:
            if impression_len >= max_imp_len:
                current_impid += 1
                impression_len = 1
            else:
                impression_len += 1
            impression_ids.append(current_impid)
        else:
            current_impid += 1
            impression_len = 1
            impression_ids.append(current_impid)
    interaction_partial.loc[:,'impression_id'] = impression_ids

# Save interaction data
select_columns = ['user_id','item_id','time','label','c_hour_c','c_weekday_c','c_period_c','c_day_f','impression_id']
train[select_columns].to_csv(os.path.join(CTR_PATH,'train.csv'),sep="\t",index=False)
val_sel[select_columns].to_csv(os.path.join(CTR_PATH,'dev.csv'),sep="\t",index=False)
test_sel[select_columns].to_csv(os.path.join(CTR_PATH,'test.csv'),sep="\t",index=False)

# organize & save item metadata
item_meta = pd.read_csv(os.path.join(DATASET, "movies.dat"),
            sep='::',names=['movieId','title','genres'],encoding='latin-1',engine='python') # columns: movieId,title,genres
item_select = item_meta.loc[item_meta.movieId.isin(interaction_ctr.news_id.unique())].copy()
# item_select['item_id'] = item_select.movieId.apply(lambda x: item2newid_ctr[str(x)])
item_select['item_id'] = item_select.movieId.apply(lambda x: item2newid_ctr[x])
genres2id = dict(zip(sorted(item_select.genres.unique()),range(1,item_select.genres.nunique()+1)))
item_select['i_genre_c'] = item_select['genres'].apply(lambda x: genres2id[x])
title2id = dict(zip(sorted(item_select.title.unique()),range(1,item_select.title.nunique()+1)))
item_select['i_title_c'] = item_select['title'].apply(lambda x: title2id[x])

item_select[['item_id','i_genre_c','i_title_c']].to_csv(
    os.path.join(CTR_PATH,'item_meta.csv'),sep="\t",index=False)