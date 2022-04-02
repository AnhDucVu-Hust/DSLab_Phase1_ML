import numpy as np
from collections import defaultdict
class member:
    def __init__(self,r_d,label=None,doc_id=None):
        self._r_d=r_d
        self._label=label
        self._doc_id=doc_id
class cluster:
    def __init__(self):
        self._centroid=None
        self._members=[]
    def reset_member(self):
        self._members=[]
    def add_member(self,member):
        self._members.append(member)
class K_mean:
    def __init__(self,num_cluster):
        self._num_cluster=num_cluster
        self._cluster=[cluster() for _ in range(self._num_cluster)]
        self._cen=[]
        self._S=0
    def load_data(self,datapath):
        def sparse_to_dense(sparse_r_d,vocab_size):
            r_d=[0.0 for _ in range(vocab_size)]
            indice_tfidfs=sparse_r_d.split()
            for index_tfidfs in indice_tfidfs:
                index=int(index_tfidfs.split(":")[0])
                tfidf=float(index_tfidfs.split(":")[1])
                r_d[index]=tfidf
            return np.array(r_d)

        with open(datapath,"r") as f:
            d_line=f.read().splitlines()
        with open("D:/20news-bydate/word_idfs.txt") as f:
            vocab_size=len(f.read().splitlines())
        self._data=[]
        self._vocabsize=vocab_size
        self._labelcount=defaultdict(int)
        for data_id,d in enumerate(d_line):
            label,doc_id=int(d.split("<fff>")[0]), int(d.split("<fff>")[1])
            self._labelcount[label] += 1
            r_d=sparse_to_dense(sparse_r_d=d.split("<fff>")[2],vocab_size=vocab_size)
            self._data.append(member(r_d=r_d,doc_id=doc_id,label=label))
    def _random_init(self,seed_value):
        for cluster in self._cluster:
            cluster._centroid = np.random.rand(self._vocabsize)
            self._cen.append(cluster._centroid)
    def compute_similarity(self,member,centroid):
        return np.sqrt(np.sum((member._r_d-centroid)**2))
    def select_cluster_for(self,member):
        best_cluster=None
        max_similar=-2
        for cluster in self._cluster:
            similar=self.compute_similarity(member,cluster._centroid)
            if similar > max_similar:
                best_cluster=cluster
                max_similar=similar
        best_cluster.add_member(member)
        return max_similar
    def update_centroid_of(self,cluster):
        member_r_ds=np.array([member._r_d for member in cluster._members]).reshape(-1,self._vocabsize)
        average_r_d=np.mean(member_r_ds,axis=0)
        sqrt_sum=np.sqrt(np.sum(average_r_d**2))
        new_centroid=np.array([a/sqrt_sum for a in average_r_d])
        cluster._centroid=new_centroid
    def run(self,seed_value,criterion,threshold):
        self._random_init(seed_value)
        self._iteration=0
        while True:
            for cluster in self._cluster:
                cluster.reset_member()
            self._new_S=0
            for member in self._data:
                max_s=self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._cluster:
                self.update_centroid_of(cluster)
            self._iteration +=1
            if self._stopping_condition(criterion,threshold):
                break
    def _stopping_condition(self,criterion,threshold):
        criteria=['centroid','similarity','max_iters']
        assert criterion in criteria
        if criterion=='max_iters':
            if self._iteration > threshold:
                return True
            else:
                return False
        elif criterion=='centroid':
            Enew=[list(cluster._centroid) for cluster in self._cluster]
            Enew_minus_E=[centroid for centroid in Enew if centroid not in self._E]
            self._E=Enew
            if len(Enew_minus_E) <= threshold:
                return True
            else:
                return False
        elif criterion=='similarity':
            newS_minus_S=self._new_S-self._S
            self._S=self._new_S
            if newS_minus_S <= threshold:
                return True
            else:
                return False
            self._new_S=0
            for member in self._data:
                max_S=self.select_cluster_for(member)
                self._new_S += max_S
    def purity(self):
        major_sum=0
        for cluster in self._cluster:
            member_labels=[member._label for member in cluster._members]
            max_count=max([member_labels.count(label) for label in range(20)])
            major_sum += max_count
        return major_sum/len(self._data)




