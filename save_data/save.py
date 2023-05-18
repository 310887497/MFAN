import csv
import json
import os
import pickle
import re

import torch
import clip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graph_part import config_file


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


def load_dataset():
    pre = os.path.dirname('C:/Users/31088/Desktop/MFAN-main/dataset/weibo/weibo_files/')
    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    with open(pre + '/new_id_dic.json', 'r', encoding='utf-8', errors='ignore') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    content_path = 'C:/Users/31088/Desktop/MFAN-main/dataset/weibo/weibocontentwithimage'
    with open(content_path + '/weibo_content.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        result = list(reader)[1:]
        mid2num = {}
        for line in result:
            mid2num[line[1]] = line[0]

    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj, newid2mid, mid2num, result


config = process_config(config_file.config)

X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj, newid2mid, mid2num, result = load_dataset()

train_data = []  # 训练集数据
valid_data = []  # 验证集数据
test_data = []  # 测试集数据

for id in X_train_tid:
    clean_text = ""
    dic_data = {}
    newid = newid2mid[id]
    img_id = mid2num[newid]
    target_text = [d[2] for d in result if d[1] == newid][0]
    sentence = target_text.split('http')[0].strip()
    clean_text = re.sub(r'[^\w\s]', '', sentence)
    target_label = [d[3] for d in result if d[1] == newid][0]
    train_data.append({'text': clean_text, 'img': img_id, 'label': target_label})


for id in X_dev_tid:
    dic_data = {}
    newid = newid2mid[id]
    img_id = mid2num[newid]
    target_text = [d[2] for d in result if d[1] == newid][0]
    sentence = target_text.split('http')[0].strip()
    target_label = [d[3] for d in result if d[1] == newid][0]
    valid_data.append({'text': sentence, 'img': img_id, 'label': target_label})

for id in X_test_tid:
    dic_data = {}
    newid = newid2mid[id]
    img_id = mid2num[newid]
    target_text = [d[2] for d in result if d[1] == newid][0]
    sentence = target_text.split('http')[0].strip()
    target_label = [d[3] for d in result if d[1] == newid][0]
    test_data.append({'text': sentence, 'img': img_id, 'label': target_label})


with open('train_CLIP_mix_weibo.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # 获取所有需要写入的字段名
    fieldnames = train_data[0].keys()

    # 创建一个csv.DictWriter对象，用于写入字典行
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 循环写入数据到CSV文件中
    for row in train_data:
        writer.writerow(row)

with open('valid_CLIP_mix_weibo.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # 获取所有需要写入的字段名
    fieldnames = valid_data[0].keys()

    # 创建一个csv.DictWriter对象，用于写入字典行
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 循环写入数据到CSV文件中
    for row in valid_data:
        writer.writerow(row)

with open('test_CLIP_mix_weibo.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # 获取所有需要写入的字段名
    fieldnames = test_data[0].keys()

    # 创建一个csv.DictWriter对象，用于写入字典行
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 循环写入数据到CSV文件中
    for row in test_data:
        writer.writerow(row)
