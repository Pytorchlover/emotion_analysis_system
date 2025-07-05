import csv
import pandas as pd
import random
from torch import nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def read_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        comments_data = [[line[0], int(line[1])] for line in reader if len(line[0]) > 0]
    data = pd.DataFrame(comments_data)
    f.close()
    return data

class BERTClassifier(nn.Module):
    def __init__(self, output_dim, pretrained_name='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.mlp = nn.Linear(768, output_dim)
    def forward(self, tokens_X):
        res = self.bert(**tokens_X)
        return self.mlp(res[1])

def evaluate(net, comments_data, labels_data, batch_size):
    sum_correct, i = 0, 0
    while i < len(comments_data):
        comments = comments_data[i: min(i + batch_size, len(comments_data))]
        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)
        res = net(tokens_X)
        y = torch.tensor(labels_data[i: min(i + batch_size, len(comments_data))]).reshape(-1).to(device=device)
        sum_correct += (res.argmax(axis=1) == y).sum()
        i += batch_size
    return sum_correct / len(comments_data)

def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels, device, batch_size, epochs):
    max_acc = 0.5
    print('开始训练...........')
    for epoch in tqdm(range(epochs)):
        i, sum_loss = 0, 0
        while i < len(train_comments):
            comments = train_comments[i: min(i + batch_size, len(train_comments))]
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)
            res = net(tokens_X)
            y = torch.tensor(train_labels[i: min(i + batch_size, len(train_comments))]).reshape(-1).to(device=device)
            optimizer.zero_grad()
            l = loss(res, y)
            l.backward()
            optimizer.step()
            sum_loss += l.detach()
            i += batch_size
        train_acc = evaluate(net, train_comments, train_labels, batch_size)
        test_acc = evaluate(net, test_comments, test_labels, batch_size)
        print('\n--epoch', epoch + 1, '\t--loss:', sum_loss / (len(train_comments) / batch_size), '\t--train_acc:', train_acc, '\t--test_acc', test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            model_name = str(len(train_comments)) + '-' + str(epochs) + '-' + str(batch_size) + '-' + 'bert' + '.parameters'
            torch.save(net.state_dict(), './model/' + model_name)
            print('精度大于之前保存的最大精度，已更新模型参数')
        else:
            print('精度小于之前保存的最大精度，未更新模型参数')

def predict(net, test_comments, test_labels, train_len, epochs, batch_size, device):
    model_name = str(train_len) + '-' + str(epochs) + '-' + str(batch_size) + '-' + 'bert' + '.parameters'
    net.load_state_dict(torch.load('./model/' + model_name))
    start = 0
    while start < 20:
        comment = test_comments[start]
        token_X = tokenizer(comment, padding=True, truncation=True, return_tensors='pt').to(device)
        label = test_labels[start]
        result = net(token_X).argmax(axis=1).item()
        print(comment)
        if result == 0:
            print('预测结果: ', 0, '----》负向', end='\t')
        elif result == 1:
            print('预测结果: ', 1, '----》中性', end='\t')
        else:
            print('预测结果: ', 2, '----》正向', end='\t')
        if label == 0:
            print('实际结果: ', 0, '----》负向', end='\t')
        elif label == 1:
            print('实际结果: ', 1, '----》中性', end='\t')
        else:
            print('实际结果: ', 2, '----》正向', end='\t')
        if result == label:
            print('预测正确')
