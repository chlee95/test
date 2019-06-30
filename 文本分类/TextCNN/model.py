from keras.models import Model
from keras.layers import Embedding,Conv2D,Input,Reshape,MaxPool2D,Concatenate,Flatten,Dense,Dropout,SpatialDropout1D

class TextCNN(object):
    def __init__(self, max_len, filter_sizes, num_filters, word_index,embedding_matrix):
        self.max_len = max_len
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix

    def get_model(self):
        inp = Input(shape=(self.max_len,))
        x = Embedding(input_dim=len(self.word_index)+1, output_dim=300, weights=[self.embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Reshape((self.max_len, 300, 1))(x)

        convs = []
        for fs in self.filter_sizes:
            c = Conv2D(self.num_filters, kernel_size=(fs, 300),kernel_initializer='he_normal',activation='relu')(x)
            c = MaxPool2D(pool_size=(self.max_len - fs + 1, 1))(c)
            convs.append(c)
        z = Concatenate(axis=1)(convs)
        z = Flatten()(z)
        z = Dropout(0.5)(z)
        out = Dense(6, activation='softmax')(z)
        model = Model(inputs=inp, outputs=out)
        return model