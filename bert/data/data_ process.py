#准备10000行的csv作为样例，进行拆分
import csv
import pandas as pd
file_name1='comments.csv'
file_name2='./weiboData/usual_train.csv'
with open(file_name2, 'r',encoding='UTF-8') as infile:
    reader = csv.reader(infile)
    output_rows = []
    for row in reader:
        output_rows.append([row[0], int(row[1])])
        if len(output_rows)==12500:
          break
    with open('test.csv', 'w', newline='',encoding='UTF-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(output_rows)
print('保存成功')
