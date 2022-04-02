import numpy as np
from tensorflow import keras
from keras import layers
def get_data(path):
    with open(path) as f:
        raw_data=f.read().splitlines()
        padded_data=[]
        padded_label=[]
        for line in raw_data:
            content=line.split("<fff>")[3].split()
            padded_data.append(content)
            label=line.split("<fff>")[0]
            padded_label.append(int(label))
        padded_data=[[int(ind) for ind in content] for content in padded_data]
    return padded_data,padded_label
train_data,train_label=get_data('D:/20news-bydate/20news-train-encoded.txt')
test_data,test_label=get_data('D:/20news-bydate/20news-test-encoded.txt')
with open("D:/20news-bydate/vocab_raw.txt") as f:
    vocab=f.read().splitlines()
LENGTH=400
embedding_size=500
LSTM_size=100
NUM_CLASSES=20
vocab_size=len(vocab)
model = keras.Sequential(
    [
        layers.Input(shape=(LENGTH)),
        layers.Embedding(vocab_size + 2,embedding_size, input_length=LENGTH,
                         mask_zero=True),
        layers.LSTM(LSTM_size),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_data,train_label, epochs=40,batch_size=50,validation_data=(test_data,test_label),verbose=1)
