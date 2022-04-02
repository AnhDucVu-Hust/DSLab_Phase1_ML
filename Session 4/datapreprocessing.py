import numpy as np
import os
import re
from genericpath import isfile
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os
def gen_data_and_vocab():
    def collect_data_from(path,new_group,word_count=None):
        data=[]
        for group_id, group in enumerate(new_group):
            dir_path=path+'/'+ group +'/'
            files=[(filename,dir_path+filename) for filename in os.listdir(dir_path) if isfile(dir_path+filename)]
            files.sort()
            label=group_id
            print('Processing: {} - {}'.format(group_id,group))
            for filename,filepath in files:
                with open(filepath) as f:
                    text=f.read().lower()
                    words=re.split("\W+", text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content=' '.join(words)
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content )
        return data
    word_count=defaultdict(int)
    path="D:/20news-bydate0"
    train_path="D:/20news-bydate/20news-bydate-train"
    test_path="D:/20news-bydate/20news-bydate-test"
    group_list=[group for group in os.listdir(train_path)]
    group_list.sort()
    train_data=collect_data_from(path=train_path,new_group=group_list,word_count=word_count)
    vocab=[word for word, freq in zip(word_count.keys(),word_count.values()) if freq > 10 ]
    vocab.sort()
    vocab_raw='\n'.join(vocab)
    with open('D:/20news-bydate/vocab_raw.txt','w') as f:
        f.write(vocab_raw)
    test_data=collect_data_from(path=test_path,new_group=group_list)
    with open("D:/20news-bydate/20news-train-raw.txt","w") as f:
        f.write('\n'.join(train_data))
    with open("D:/20news-bydate/20news-test-raw.txt","w") as f:
        f.write('\n'.join(test_data))
def encode_data(data_path,vocab_path):
    MAX_DOC_LENGTH=500
    with open(vocab_path) as f:
        vocab=dict([word,word_ID+2] for word_ID,word in enumerate(f.read().splitlines()))
    with open(data_path) as f:
        documents=[]
        for line in f.read().splitlines():
            content=line.split("<fff>")
            documents.append((content[0],content[1],content[2]))
    encoded_data=[]
    for document in documents:
        label,doc_id,text=document
        words=text.split()[:MAX_DOC_LENGTH]
        sentence_length=len(words)
        encoded_text=[]
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(1))
        if len(words) < MAX_DOC_LENGTH:
            num_padding=MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(0))
        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' + str(sentence_length) + '<fff>' + ' '.join(encoded_text))
    dir_name='/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + '/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))
gen_data_and_vocab()
MAX_DOC_LENGTH=400
encode_data(data_path='D:/20news-bydate/20news-train-raw.txt',vocab_path='D:/20news-bydate/vocab_raw.txt')
encode_data(data_path='D:/20news-bydate/20news-test-raw.txt',vocab_path='D:/20news-bydate/vocab_raw.txt')

