import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from pathlib import Path
import os
local_path = os.getcwd()
local_path = local_path.replace(os.sep, '/')

class BERTClassifier(nn.Module):
    def __init__(self, output_dim, pretrained_name='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(local_path+'/bert-base-chinese')
        self.mlp = nn.Linear(768, output_dim)
    def forward(self, tokens_X):
        res = self.bert(**tokens_X)
        return self.mlp(res[1])

tokenizer = BertTokenizer.from_pretrained(local_path+'/bert-base-chinese')
device = torch.device("cpu")
net = BERTClassifier(output_dim=3)
net = net.to(device)
net.load_state_dict(torch.load(local_path + '/bert/21642-10-32-bert.parameters', map_location='cpu'), False)

def bert_predict(text_comments):
    token_X = tokenizer(text_comments, padding=True, truncation=True, return_tensors='pt',max_length=512).to(device)
    result = net(token_X).argmax(axis=1).item()
    if result == 0:
        emotion_prefer = '负向'
    elif result == 1:
        emotion_prefer = '中性'
    else:
        emotion_prefer = '正向'
    return emotion_prefer
