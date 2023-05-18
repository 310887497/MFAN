import csv
import json
import os
import pickle

content_path = 'C:/Users/31088/Desktop/谣言检测数据集/twitter'

with open(content_path + '/test_posts.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    result = list(reader)[1:]

with open("test_twitter.csv", "w", newline="", encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["text", "img", "label"])
    writer.writeheader()

    # 写入每一行数据
    for row in result:
        if row[6] == "real":
            row[6] = 1
        else:
            row[6] = 0
        writer.writerow({"text": row[1], "img": row[4], "label": row[6]})
