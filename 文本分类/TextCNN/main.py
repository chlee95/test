import codecs
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from model import TextCNN
embedding_file = 'h:/wiki_news.vec'
train_data_file = 'train.txt'
test_data_file = 'test.txt'
max_len = 12
filter_sizes = [3,4,5]
num_filters = 100

def get_data(file):
    labels = []
    sentences = []
    with open(file,encoding='latin-1') as f:
        for lines in f:
            line = lines.strip().split()
            labels.append(int(line[0]))
            sentences.append(line[1:])
    return labels,sentences

train_labels,train_sentences = get_data(train_data_file)
test_labels, test_sentences = get_data(test_data_file)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentences+test_sentences)
train_X = tokenizer.texts_to_sequences(train_sentences)
test_X = tokenizer.texts_to_sequences(test_sentences)
train_X = pad_sequences(train_X,maxlen=max_len)
test_X = pad_sequences(test_X,maxlen=max_len)
train_y = to_categorical(train_labels,num_classes=6)
test_y = to_categorical(test_labels,num_classes=6)

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split( )) for o in open(embedding_file, encoding='utf-8') if len(o) > 100)
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index)+1, 300))
for word,i in word_index.items():
    if word in embedding_index:
        embedding_matrix[i] = embedding_index[word]
    else:
        vec = np.random.uniform(-0.25,0.25,300)
        embedding_matrix[i] = vec

model = TextCNN(max_len, filter_sizes, num_filters, word_index,embedding_matrix).get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_X,y=train_y,batch_size=128,epochs=10,verbose=2)