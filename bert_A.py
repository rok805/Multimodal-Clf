#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

# torch, nlp를 위한 모듈 로드
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import re
import gc
import os

# kobert 모듈 로드
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# transformer 모듈 로드
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

# kobert 모듈 로드
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


#GPU 사용 시
device = torch.device("cuda:0")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"  # Set the GPU 2 to use

from textrank import KeysentenceSummarizer

from konlpy.tag import Okt

import pickle



class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))



# 학습 데이터셋 구축 클래스
class BERTDataset(Dataset):
    
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

    
# BERT 분류기 클래스
class BERTClassifier(nn.Module):
    
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 2, 
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
#         print(attention_mask)
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out), pooler    
    

# 학습 평가 지표인 accuracy 계산 -> 얼마나 타겟값을 많이 맞추었는가
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()

device = torch.device("cuda:0") #GPU 사용 시
#device = torch.device("cpu") #CPU 사용 시
torch.cuda.is_available()

# 토크나이저 생성
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

model = BERTClassifier(bertmodel, dr_rate=0.2).to(device)
# model = nn.DataParallel(model, device_ids=[0,1])
model.cuda()
model = nn.DataParallel(model).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

max_len = 128
batch_size = 64
warmup_ratio = 0.1
num_epochs = 3
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

    
    
    
    
    

if __name__=='__main__':

    print('data_load')
    with open('dataA.pkl', 'rb') as p:
        data_list = pickle.load(p)
        
    print('data_setting')
    # 학습
    # Train / Test set 분리
    #dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, random_state=0)
    
    dataset_train = data_list

    # 학습을 위한 데이터셋 구축
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    #data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)


    # pytorch용 DataLoader 사용
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    #test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    # 옵티마이저 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss() # softmax용 Loss Function 정하기 <- binary classification도 해당 loss function 사용 가능

    # warm_up
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    # 스케쥴러
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    print('learning')
    for e in range(num_epochs):
        train_acc = 0.0
        #test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out, hidden = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))

        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))


#         model.eval()
#         for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader)):
#             token_ids = token_ids.long().to(device)
#             segment_ids = segment_ids.long().to(device)
#             valid_length= valid_length
#             label = label.long().to(device)
#             out, hidden = model(token_ids, valid_length, segment_ids)
#             test_acc += calc_accuracy(out, label)
#         print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))


        # GPU 사용시 학습 전 캐시제거
        gc.collect()
        torch.cuda.empty_cache()
        
        
    # 모델 로컬 저장
    torch.save(model, f'./result_m_A.pt')
    torch.save(model.state_dict(), f'./result_state_dict_A.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f'./result_A.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능



