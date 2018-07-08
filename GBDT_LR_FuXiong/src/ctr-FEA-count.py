
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3

import argparse, csv, sys, pickle, collections, time

from common import *

src='~/train_sample10w.csv'
FIELDS = ['id','click','hour','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
NEW_FIELDS = FIELDS+['device_id_count','device_ip_count','user_count']

id_cnt = collections.defaultdict(int)
ip_cnt = collections.defaultdict(int)
user_cnt = collections.defaultdict(int)

start = time.time()

def scan(src):
    for i, row in enumerate(csv.DictReader(open(src)), start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))

        user = def_user(row)
        id_cnt[row['device_id']] += 1
        ip_cnt[row['device_ip']] += 1
        user_cnt[user] += 1

history = collections.defaultdict(lambda: {'history': '', 'buffer': '', 'prev_hour': ''})

def gen_data(src):
    reader = csv.DictReader(open(src))
    writer = csv.DictWriter(open(src, 'w'), NEW_FIELDS)
    writer.writeheader()

    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))
        
        new_row = {}
        for field in FIELDS:
            new_row[field] = row[field]

        new_row['device_id_count'] = id_cnt[row['device_id']]
        new_row['device_ip_count'] = ip_cnt[row['device_ip']]
        user= def_user(row)
        new_row['user_count'] = user_cnt[user]
        writer.writerow(new_row)

print('======================scan complete======================')

