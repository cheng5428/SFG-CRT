import numpy as np
import pandas as pd
import os
import sys
import zipfile
import json
import gc
from tqdm import tqdm
from datetime import datetime

# --- 配置 ---
DATASET = 'MIND_large' 
RAW_PATH = os.path.join('./', DATASET)
CTR_PATH = './MINDCTR/'

RANDOM_SEED = 0
NEG_ITEMS = 99

def extract_data():
    """解压数据文件"""
    print('Unzip files...')
    if os.path.exists(os.path.join(RAW_PATH,'MINDlarge_train.zip')):
        f = zipfile.ZipFile(os.path.join(RAW_PATH,'MINDlarge_train.zip'),'r') 
        for file in f.namelist():
            print("Extract %s"%(file))
            f.extract(file, RAW_PATH)
        f.close()

    if os.path.exists(os.path.join(RAW_PATH,'MINDlarge_dev.zip')):
        f = zipfile.ZipFile(os.path.join(RAW_PATH,'MINDlarge_dev.zip'),'r') 
        for file in f.namelist():
            print("Extract %s"%(file))
            f.extract(file, RAW_PATH)
        f.close()

def read_and_filter_interactions():
    """读取交互数据并进行5-core过滤"""
    print("Reading interaction data...")
    interactions = []
    user_freq, item_freq = dict(), dict()
    
    for d in [os.path.join(RAW_PATH,'MINDlarge_train'), os.path.join(RAW_PATH,'MINDlarge_dev')]:
        file = os.path.join(d, "behaviors.tsv")
        if not os.path.exists(file):
            continue
        with open(file, 'r', encoding='utf-8') as F:
            for line in tqdm(F, desc=f"Reading {d}"):
                line = line.strip().split("\t")
                sid, uid, time = line[0], line[1], line[2]
                impressions = line[4].split(" ")
                for imp in impressions:
                    iid, label = imp.split("-")
                    interactions.append([sid, uid, time, iid, label])
                    if int(label) == 1:
                        user_freq[uid] = user_freq.get(uid, 0) + 1
                        item_freq[iid] = item_freq.get(iid, 0) + 1

    # 5-core filtering
    print("Starting 5-core filtering...")
    select_uid, select_iid = [], []
    for u in user_freq:
        if user_freq[u] >= 5:
            select_uid.append(u)
    for i in item_freq:
        if item_freq[i] >= 5:
            select_iid.append(i)
    print("User: %d/%d, Item: %d/%d" % (len(select_uid), len(user_freq), len(select_iid), len(item_freq)))

    while len(select_uid) < len(user_freq) or len(select_iid) < len(item_freq):
        select_uid = set(select_uid)
        select_iid = set(select_iid)
        user_freq, item_freq = dict(), dict()
        interactions_5core = []
        for line in tqdm(interactions, desc="5-core filtering"):
            uid, iid, label = line[1], line[3], line[-1]
            if uid in select_uid and iid in select_iid:
                interactions_5core.append(line)
                if int(label) == 1:
                    user_freq[uid] = user_freq.get(uid, 0) + 1
                    item_freq[iid] = item_freq.get(iid, 0) + 1
        interactions = interactions_5core
        select_uid, select_iid = [], []
        for u in user_freq:
            if user_freq[u] >= 5:
                select_uid.append(u)
        for i in item_freq:
            if item_freq[i] >= 5:
                select_iid.append(i)
        print("User: %d/%d, Item: %d/%d" % (len(select_uid), len(user_freq), len(select_iid), len(item_freq)))

    print("Selected Interactions: %d, Users: %d, Items: %d" % (len(interactions), len(select_uid), len(select_iid)))

    # exclude illegal interactions
    for i in range(len(interactions)):
        if len(interactions[i]) > 5:
            interactions[i] = interactions[i][:-1]

    return interactions

def create_features(interactions):
    """构建DataFrame并生成时间特征"""
    print("Constructing DataFrame...")
    interaction_df = pd.DataFrame(interactions, columns=["session_id", "user_id", "time_str", "news_id", "label"])

    # 释放原始列表内存
    print("Releasing raw list memory...")
    del interactions
    gc.collect()

    # 解析时间戳
    print("Parsing timestamps...")
    format_t = '%m/%d/%Y %I:%M:%S %p'
    interaction_df['timestamp'] = pd.to_datetime(interaction_df['time_str'], format=format_t)

    # 生成特征
    print("Generating features...")
    interaction_df['time'] = interaction_df['timestamp'].astype(np.int64) // 10**9
    interaction_df['hour'] = interaction_df['timestamp'].dt.hour
    interaction_df['weekday'] = interaction_df['timestamp'].dt.weekday
    interaction_df['date'] = interaction_df['timestamp'].dt.date

    # 优化 period 计算
    h = interaction_df['hour']
    conditions = [
        (h >= 5) & (h <= 8),
        (h > 8) & (h < 11),
        (h >= 11) & (h <= 12),
        (h > 12) & (h <= 15),
        (h > 15) & (h <= 17),
        (h >= 18) & (h <= 19),
        (h > 19) & (h <= 21),
        (h > 21)
    ]
    choices = [0, 1, 2, 3, 4, 5, 6, 7]
    interaction_df['period'] = np.select(conditions, choices, default=8)

    # 计算 day
    min_date_ts = interaction_df['timestamp'].min().normalize()
    interaction_df['day'] = (interaction_df['timestamp'].dt.normalize() - min_date_ts).dt.days

    return interaction_df

def process_ctr_task(interaction_df):
    """处理CTR任务数据"""
    print(f"\nCreating output directory: {CTR_PATH}")
    os.makedirs(CTR_PATH, exist_ok=True)

    # 重命名与 ID 映射
    print("Renaming columns...")
    interaction_ctr = interaction_df.copy()
    interaction_ctr.rename(columns={
        'hour': 'c_hour_c',
        'weekday': 'c_weekday_c',
        'period': 'c_period_c',
        'day': 'c_day_f',
        'user_id': 'original_user_id'
    }, inplace=True)

    # 确保 ID 是字符串类型
    interaction_ctr['original_user_id'] = interaction_ctr['original_user_id'].astype(str)
    interaction_ctr['news_id'] = interaction_ctr['news_id'].astype(str)

    print("Generating User IDs...")
    unique_users = sorted(interaction_ctr.original_user_id.unique())
    user2newid_ctr = {u: i+1 for i, u in enumerate(unique_users)}
    interaction_ctr['user_id'] = interaction_ctr['original_user_id'].map(user2newid_ctr).astype('int32')

    print("Generating Item IDs...")
    unique_items = sorted(interaction_ctr.news_id.unique())
    item2newid_ctr = {i: k+1 for k, i in enumerate(unique_items)}
    interaction_ctr['item_id'] = interaction_ctr['news_id'].map(item2newid_ctr).astype('int32')

    # 关键修正：显式排序
    print("Sorting data by User ID and Time (Critical Step)...")
    interaction_ctr.sort_values(by=['user_id', 'time'], inplace=True)
    interaction_ctr.reset_index(drop=True, inplace=True)

    # 保存映射字典
    print("Saving ID mappings...")
    with open(os.path.join(CTR_PATH, "user2newid.json"), 'w') as f:
        json.dump(user2newid_ctr, f)
    with open(os.path.join(CTR_PATH, "item2newid.json"), 'w') as f:
        json.dump(item2newid_ctr, f)

    # 划分数据集
    print("\nSplitting datasets...")
    split_time1 = 5
    
    train = interaction_ctr.loc[interaction_ctr.c_day_f <= split_time1].copy()
    val_test = interaction_ctr.loc[interaction_ctr.c_day_f > split_time1].copy()
    
    del interaction_ctr
    gc.collect()

    val_test.sort_values(by='time', inplace=True)

    print("Splitting Val/Test by session...")
    unique_sessions = val_test['session_id'].unique() 
    split_idx = len(unique_sessions) // 2
    
    val_sessions_set = set(unique_sessions[:split_idx])
    
    val = val_test.loc[val_test.session_id.isin(val_sessions_set)].copy()
    test = val_test.loc[~val_test.session_id.isin(val_sessions_set)].copy()

    del val_test
    gc.collect()

    # 过滤 Cold-Start
    print("Filtering cold-start users/items...")
    train_u = set(train.user_id.unique())
    train_i = set(train.item_id.unique())

    val_sel = val.loc[(val.user_id.isin(train_u)) & (val.item_id.isin(train_i))].copy()
    test_sel = test.loc[(test.user_id.isin(train_u)) & (test.item_id.isin(train_i))].copy()

    print(f"Train user: {len(train_u)}, item: {len(train_i)}")
    print(f"Validation user: {val_sel.user_id.nunique()}, item: {val_sel.item_id.nunique()}")
    print(f"Test user: {test_sel.user_id.nunique()}, item: {test_sel.item_id.nunique()}")

    # 保存结果
    print("\nSaving datasets...")
    select_columns = ['user_id', 'item_id', 'time', 'label', 'c_hour_c', 'c_weekday_c', 'c_period_c', 'c_day_f']
    
    train[select_columns].to_csv(os.path.join(CTR_PATH, 'train.csv'), sep="\t", index=False)
    val_sel[select_columns].to_csv(os.path.join(CTR_PATH, 'dev.csv'), sep="\t", index=False)
    test_sel[select_columns].to_csv(os.path.join(CTR_PATH, 'test.csv'), sep="\t", index=False)

    del train, val, test, val_sel, test_sel
    gc.collect()

    # 处理物品元数据
    print("\nProcessing Item Metadata...")
    possible_paths = [
        os.path.join(RAW_PATH, 'MINDlarge_train', "news.tsv"),
        os.path.join(RAW_PATH, 'train', "news.tsv"),
        os.path.join(RAW_PATH, "news.tsv")
    ]
    
    news_file = None
    for p in possible_paths:
        if os.path.exists(p):
            news_file = p
            break
            
    if news_file:
        print(f"Reading {news_file}...")
        item_meta_train = pd.read_csv(news_file, sep="\t", header=None, quoting=3) 
        item_meta_train.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entitiy', 'abstract_entity']
        
        valid_news_ids = set(item2newid_ctr.keys())
        item_select = item_meta_train.loc[item_meta_train.news_id.isin(valid_news_ids)].copy()
        
        item_select['item_id'] = item_select.news_id.map(item2newid_ctr)
        
        category2id = dict(zip(sorted(item_select.category.unique()), range(1, item_select.category.nunique() + 1)))
        subcategory2id = dict(zip(sorted(item_select.subcategory.unique()), range(1, item_select.subcategory.nunique() + 1)))
        
        item_select['i_category_c'] = item_select['category'].map(category2id)
        item_select['i_subcategory_c'] = item_select['subcategory'].map(subcategory2id)
        
        item_select[['item_id', 'i_category_c', 'i_subcategory_c']].to_csv(
            os.path.join(CTR_PATH, 'item_meta.csv'), sep="\t", index=False)
        print("Item metadata saved.")
    else:
        print("Warning: news.tsv not found!")

def main():
    """主函数"""
    print("=" * 60)
    print("MIND Large Dataset CTR Processing Pipeline")
    print("=" * 60)
    
    # Step 1: 解压数据
    extract_data()
    
    # Step 2: 读取并过滤交互数据
    interactions = read_and_filter_interactions()
    
    # Step 3: 构建特征
    interaction_df = create_features(interactions)
    
    # Step 4: 保存5-core交互数据
    print("\nSaving 5-core interactions...")
    interaction_df.to_csv("interaction_5core.csv", index=False)
    print("Saved to interaction_5core.csv")
    
    # Step 5: 处理CTR任务
    process_ctr_task(interaction_df)
    
    print("\n" + "=" * 60)
    print("All Processing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()