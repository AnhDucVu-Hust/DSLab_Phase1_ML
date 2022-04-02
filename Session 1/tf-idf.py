from genericpath import isfile
import numpy as np
import os
from nltk.stem.porter import PorterStemmer
import re
from collections import defaultdict
def gather_data():
    path="D:/20news-bydate/"
    train_dir=path+"20news-bydate-train/"
    test_dir = path + "20news-bydate-test/"
    list_group= [group for group in os.listdir(train_dir) if not isfile(train_dir+group)]
    list_group.sort()
    stemmer=PorterStemmer()
    def collect_data(dir,list_group):
        data=[]
        for id,group in enumerate(list_group):
            label=id
            dir_path=dir + "/" + group + "/"
            files=[(filename,dir_path+filename) for filename in os.listdir(dir_path)]
            for filename, filepath in files:
                with open(filepath,"r") as f:
                    text=f.read().lower()
                    words=[stemmer.stem(word) for word in re.split("\W+",text)]
                    content=' '.join(words)
                    data.append(str(label)+'<fff>'+filename+'<fff>'+content)
        return data
    train_data=collect_data(train_dir,list_group)
    test_data=collect_data(test_dir,list_group)
    full_data=train_data+test_data
    with open("D:/20news-bydate/train_data",'w') as f:
        f.write('\n'.join(train_data))
    with open("D:/20news-bydate/test_data",'w') as f:
        f.write('\n'.join(test_data))
    with open("D:/20news-bydate/full_data",'w') as f:
        f.write('\n'.join(full_data))
def gather_library(data_path):
    def compute_idf(df,corpus_size):
        assert df >0
        return np.log10(corpus_size/df)
    with open(data_path,"r") as f:
        lines=f.readlines()
    lines_data=[]
    doc_count=defaultdict(int)
    corpus_size=len(lines)
    for line in lines:
        text=line.strip('\n').split("<fff>")
        word=text[-1]
        words=list(set(word.split()))
        for a in words:
            doc_count[a] +=1
    word_idfs=[(word,compute_idf(document_freq,corpus_size)) for word,document_freq in zip(doc_count.keys(),doc_count.values()) if document_freq>10 and not word.isdigit()]
    print("vocabulary size: {}",format(len(word_idfs)))
    with open("D:/20news-bydate/word_idfs.txt","w") as f:
        f.write("\n".join(word + '<fff>' + str(idf) for word,idf in word_idfs))
def get_tfidf(data_path):
    with open("D:/20news-bydate/word_idfs.txt","r") as f:
        word_idfs= [(line.split('<fff>')[0],float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        word_ids= dict([(word,index) for index, (word,idf) in enumerate(word_idfs)])
        idfs=dict(word_ids)
    with open(data_path,"r") as f:
        documents=[(int(line.split("<fff>")[0]),int(line.split("<fff>")[1]),line.split("<fff>")[2]) for line in f.read().splitlines()]
    data_tf_idf=[]
    for document in documents:
        label,doc_id,text=document
        words=[word for word in text.split() if word in idfs]
        word_set=list(set(words))
        max_term_freq=max([words.count(word) for word in word_set])
        word_tfidfs=[]
        sum_square=0.0
        for word in word_set:
            term_freq=words.count(word)
            tf_idf_value=term_freq/max_term_freq *idfs[word]
            word_tfidfs.append((word_ids[word],tf_idf_value))
            sum_square += tf_idf_value ** 2
        word_tfidfs_normalized = [str(index) + ":" + str(tf_idf_value/np.sqrt(sum_square)) for index, tf_idf_value in word_tfidfs ]
        rep=' '.join(word_tfidfs_normalized)
        data_tf_idf.append((label,doc_id,rep))
    data_tf_idf_normalize=''
    for label,doc_id,rep in data_tf_idf:
        data_tf_idf_normalize += str(label) + "<fff>" + str(doc_id) + "<fff>" + rep +"\n"
    with open("D:/20news-bydate/tf_idf_"+data_path.split("/")[-1].split("_")[0],"w") as f:
        f.write(data_tf_idf_normalize)

#gather_data()
gather_library("D:/20news-bydate/full_data")
get_tfidf("D:/20news-bydate/train_data")
get_tfidf("D:/20news-bydate/test_data")