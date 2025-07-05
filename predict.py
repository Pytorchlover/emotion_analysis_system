"""
情感分析预测模块
基于BERT模型进行中文情感分析
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


class SentimentClassifier(nn.Module):
    """情感分析模型"""
    
    def __init__(self, n_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_output.pooler_output)
        return self.out(output)


def load_model(model_path='bert/model.pth'):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(n_classes=3)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        print("模型文件未找到，请先训练模型")
        return None, device


def preprocess_text(text, tokenizer, max_len=128):
    """文本预处理"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding


def predict_sentiment(text, model=None, tokenizer=None, device=None):
    """预测单条文本的情感"""
    if model is None or tokenizer is None:
        # 如果没有传入模型，尝试加载
        model, device = load_model()
        if model is None:
            return "模型加载失败"
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 文本预处理
    encoding = preprocess_text(text, tokenizer)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(prediction, dim=1).cpu().numpy()[0]
        confidence = prediction.max().cpu().numpy()
    
    # 映射标签
    label_map = {0: '消极', 1: '中性', 2: '积极'}
    
    return {
        'sentiment': label_map[predicted_class],
        'confidence': float(confidence),
        'all_scores': {
            '消极': float(prediction[0][0]),
            '中性': float(prediction[0][1]),
            '积极': float(prediction[0][2])
        }
    }


def batch_predict(texts, model=None, tokenizer=None, device=None):
    """批量预测文本情感"""
    if model is None or tokenizer is None:
        model, device = load_model()
        if model is None:
            return ["模型加载失败"] * len(texts)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    results = []
    for text in texts:
        result = predict_sentiment(text, model, tokenizer, device)
        results.append(result)
    
    return results


def evaluate_model(test_data_path, model=None, tokenizer=None):
    """评估模型性能"""
    if model is None or tokenizer is None:
        model, device = load_model()
        if model is None:
            return "模型加载失败"
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 加载测试数据
    df = pd.read_csv(test_data_path)
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    
    # 预测
    predictions = batch_predict(texts, model, tokenizer, device)
    predicted_labels = [pred['sentiment'] for pred in predictions]
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }


def bert_predict(text):
    """简化的预测函数，用于外部调用"""
    try:
        model, device = load_model()
        if model is None:
            return "模型未加载"
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        result = predict_sentiment(text, model, tokenizer, device)
        return result['sentiment']
    except Exception as e:
        return f"预测失败: {str(e)}"


if __name__ == "__main__":
    # 测试代码
    test_texts = [
        "今天天气真好！",
        "这部电影太糟糕了",
        "还可以吧，一般般",
        "非常喜欢这个产品",
        "服务态度很差"
    ]
    
    print("开始情感分析测试...")
    for text in test_texts:
        result = bert_predict(text)
        print(f"文本: {text}")
        print(f"情感: {result}")
        print("-" * 50)