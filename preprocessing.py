from textrank import KeysentenceSummarizer
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
from konlpy.tag import Okt
import pickle
import re

#문장분리
from kiwipiepy import Kiwi

# Summarization(뉴스 요약)

def okt_tokenize(sent):

    okt = Okt()
    words = okt.nouns(sent)

    return words

def getsummarize(txt, k=5):

    summarizer = KeysentenceSummarizer(
        tokenize = okt_tokenize,
        min_sim = 0.5,
        verbose = True
        )
    keysents = summarizer.summarize(txt, topk=k)

    result_dict = {}
    for i in keysents:
        result_dict[i[0]] = i[1]

    return result_dict

def make_data_list(data):
    
    data_list = []
    
    for ques, label in zip(data['overview'], data['cat3']):
        tmp = []   
        tmp.append(ques)
        tmp.append(str(label))

        data_list.append(tmp)
        
    return data_list


def make_dataA(data_list):
    
    dataA = []
    for i in data_list:
        if i[1] in ['한식', '야영장,오토캠핑장']:
            dataA.append([i[0], 1])
        else:
            dataA.append([i[0], 0])
            
    with open('dataA.pkl', 'wb') as p:
        pickle.dump(dataA, p)
    
    return dataA


def make_dataB(data_list):
    
    dataB = []
    for i in data_list:
        if i[1] == '한식':
            dataB.append([i[0], 1])
        elif i[1] == '야영장,오토캠핑장':
            dataB.append([i[0], 0])
        else:
            pass

    with open('dataB.pkl', 'wb') as p:
        pickle.dump(dataB, p)

    return dataB
    
    
def make_dataC(data_list):
    
    dataC = []
    for i in data_list:
        if i[1] not in ['한식', '야영장,오토캠핑장']:
            dataC.append(i)
        else:
            pass
        
    data = pd.DataFrame(dataC, columns=['content', 'cat3'])
    
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'species'.
    data['cat3']= label_encoder.fit_transform(data['cat3'])

    # 저장용
    num_to_chr = {}
    for i in data['cat3'].unique():
        num_to_chr[i] = label_encoder.inverse_transform(np.array([i]))[0]
    dataC=[]
    for idx, row in tqdm(data.iterrows()):
        dataC.append([row[0], row[1]])

    with open('dataC.pkl', 'wb') as p:
        pickle.dump(dataC, p)
    with open('dataC_dict.pkl', 'wb') as p:
        pickle.dump(num_to_chr, p)
        
        
    return dataC
    
    
if __name__=="__main__":

    data = pd.read_csv('./train.csv')[['overview', 'cat3']]


    # 문장분리 (3분)
    print('문장분리')
    kiwi = Kiwi()
    sent_split = []
    for sent in tqdm(data['overview']):
        sents = []

        for s in kiwi.split_into_sents(sent):
            sents.append(s[0])
        sent_split.append(sents)


    #주요문장순서대로 정렬(10분)
    print('주요문장')
    main_sentences = []
    err_cnt = 0 #문장추출 에러 카운트
    for sent in tqdm(sent_split):
        sent_prep = []

        # 전처리
        for s in sent:
            sent_prep.append(' '.join(re.sub('[^가-힣]', ' ', s).split()))

        sent_prep = [i for i in sent_prep if len(i)>0]

        # 주요문장 추출
        try:
            text_rank = getsummarize(sent_prep, k=99)
            main_sentence_idx = list(text_rank.keys())
            main_sents = [sent_prep[i] for i in main_sentence_idx]
        except:
            err_cnt += 1
            main_sents = sent_prep

        main_sentences.append(main_sents)
    print(f'error:{err_cnt}')
        
    data['overview'] = [' '.join(i) for i in main_sentences]
    
    data_list = make_data_list(data)
    
    #data_A = make_dataA(data_list) # 한식 or 캠핑장: 1, 그외: 0
    #data_B = make_dataB(data_list) # 한식: 1, 캠핑장: 0
    print("C")
    data_C = make_dataC(data_list) # 그외1: 1, 그외2: 2, ... , 그외n: n