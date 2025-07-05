# -*- coding: utf-8 -*-
"""
测试模块
基于LSTM的情感分析模型测试和评估
"""

import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import csv
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import pandas as pd

# 超参数设置
vocab_path = './data/vocab.pkl'  # 词表
save_path = './saved_dict/lstm.ckpt'  # 模型训练结果
data_path = 'output_file.txt'  # 测试数据路径

# 模型参数
try:
    embedding_pretrained = torch.tensor(
        np.load('./data/embedding_Tencent.npz')["embeddings"].astype('float32')
    )
    embed = embedding_pretrained.size(1)  # 词向量维度
except FileNotFoundError:
    print("预训练词向量文件未找到，使用随机初始化")
    embed = 200
    embedding_pretrained = None

# 超参数
dropout = 0.5  # 随机丢弃
num_classes = 2  # 类别数
num_epochs = 30  # epoch数
batch_size = 128  # mini-batch大小
pad_size = 50  # 每句话处理成的长度(短填长切)
learning_rate = 1e-3  # 学习率
hidden_size = 128  # lstm隐藏层
num_layers = 2  # lstm层数
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
tokenizer = lambda x: [y for y in x]  # 字级别

# 设备配置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载词表
try:
    vocab = pkl.load(open(vocab_path, 'rb'))
except FileNotFoundError:
    print("词表文件未找到")
    vocab = {}


def get_data():
    """获取测试数据"""
    print(f"词表大小: {len(vocab)}")
    test = load_dataset(data_path, pad_size, tokenizer, vocab)
    return test


def load_dataset(path, pad_size, tokenizer, vocab):
    """
    加载数据集
    
    Args:
        path: 文件路径
        pad_size: 序列长度
        tokenizer: 分词器
        vocab: 词表
        
    Returns:
        处理后的数据列表
    """
    contents = []
    n = 0
    
    try:
        with open(path, 'r', encoding='utf8') as f:
            for line in tqdm(f, desc="加载数据"):
                lin = line.strip()
                if not lin:
                    continue
                    
                try:
                    content, label = lin.split('\t####\t')
                except ValueError:
                    print(f"数据格式错误: {lin}")
                    continue
                    
                words_line = []
                token = tokenizer(content)
                
                # 填充或截断
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD, 0)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        
                # 将词映射为ID
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK, 0)))
                    
                n += 1
                contents.append((words_line, int(label)))
                
    except FileNotFoundError:
        print(f"文件未找到: {path}")
        return []
        
    print(f"加载了 {n} 条数据")
    return contents


class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Model(nn.Module):
    """LSTM情感分析模型"""
    
    def __init__(self):
        super(Model, self).__init__()
        
        if embedding_pretrained is not None:
            # 使用预训练词向量
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            # 随机初始化词向量
            self.embedding = nn.Embedding(len(vocab), embed)
            
        # 双向LSTM
        self.lstm = nn.LSTM(
            embed, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


def get_time_dif(start_time):
    """计算时间差"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def init_network(model, method='xavier', exclude='embedding'):
    """初始化网络权重"""
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)


def eval_one(model, texts):
    """单个文本评估"""
    model.eval()
    all_result = []
    
    with torch.no_grad():
        for content in texts:
            words_line = []
            token = tokenizer(content)
            
            # 填充或截断
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD, 0)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    
            # 词映射为ID
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK, 0)))
                
            # 转为tensor
            input_tensor = torch.LongTensor([words_line]).to(device)
            outputs = model(input_tensor)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
            
            label_map = {0: '消极', 1: '积极'}
            all_result.append(label_map.get(predic, '未知'))
            
    return all_result


def dev_eval(model, data_iter, test=False):
    """模型评估"""
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss_total += loss.item()
            
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    acc = accuracy_score(labels_all, predict_all)
    
    if test:
        # 详细评估报告
        report = metrics.classification_report(labels_all, predict_all, 
                                             target_names=['消极', '积极'], digits=4)
        confusion = confusion_matrix(labels_all, predict_all)
        
        print(f"测试集准确率: {acc:.4f}")
        print("分类报告:")
        print(report)
        print("混淆矩阵:")
        print(confusion)
        
        # 保存混淆矩阵图
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/confusion_matrix.png')
        plt.close()
        
    return acc, loss_total / len(data_iter)


def test_csv(csv_path):
    """测试CSV文件中的数据"""
    try:
        df = pd.read_csv(csv_path)
        
        # 检查是否有content列
        if 'content' not in df.columns:
            print("CSV文件中没有找到'content'列")
            return
            
        # 提取内容
        content_df = df[['content']].copy()
        content_df['label'] = -1  # 默认标签
        
        # 保存为测试格式
        with open('output_file.txt', 'w', encoding='utf-8') as f:
            for index, row in content_df.iterrows():
                f.write(f"{row['content']}\t####\t{row['label']}\n")
                
        # 加载模型并测试
        model = Model()
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        
        test_data = get_data()
        test_iter = DataLoader(TextDataset(test_data), batch_size, shuffle=False)
        
        acc, loss = dev_eval(model, test_iter, test=True)
        print(f"测试完成 - 准确率: {acc:.4f}, 损失: {loss:.4f}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")


if __name__ == "__main__":
    print("LSTM情感分析测试模块")
    print(f"设备: {device}")
    print(f"词表大小: {len(vocab)}")
    
    # 示例测试
    test_texts = ["这部电影太好看了", "服务态度很差", "还可以吧"]
    
    try:
        model = Model()
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        
        results = eval_one(model, test_texts)
        
        for text, result in zip(test_texts, results):
            print(f"文本: {text} -> 情感: {result}")
            
    except FileNotFoundError:
        print("模型文件未找到，请先训练模型")
    except Exception as e:
        print(f"测试出错: {e}")