import random
import os
import data_helpers as dhrt
from tensorflow.contrib import learn
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sequence_helpers import get_alignments, get_vocab, class_to_onehot


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, LSTM, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Input, Reshape, GlobalMaxPooling1D, dot, multiply
from keras.layers import RepeatVector, concatenate, Permute
from keras.optimizers import SGD, Adam
from keras.layers.merge import Dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from keras.preprocessing.sequence import skipgrams
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt
from Bio import pairwise2


# Network Parameters
learning_rate = 0.01
num_features = 372
word_length = 6
vec_length = 4
batch_size = 4
nb_epoch = 16
hidden_size = 100
sequences_per_family = 1000
num_sequences = 10
steps_per_epoch = 10
num_classes = 10
num_filters = [16, 4]


def split_alignments(x, max_len):
    s1 = []
    s2 = []
    for i in range(len(x)):
        proc1 = ''
        proc2 = ''
        for j in range(max_len):
            if j < len(x[i][0]):
                proc1 += (x[i][0][j]).replace('-', 'x')
                proc2 += (x[i][1][j]).replace('-', 'x')
            else:
                proc1 += 'x'
                proc2 += 'x'
        proc1 = ' '.join([proc1[i:i + word_length] for i in range(0, len(proc1))])
        proc2 = ' '.join([proc2[i:i + word_length] for i in range(0, len(proc2))])
        s1 = np.append(s1, proc1)
        s2 = np.append(s2, proc2)
    return s1, s2


def generate_vec_batch(x_train, y_train, batch_size, tokenizer, SkipGram):
    indices_i = np.arange(0, len(x_train) - 1 - batch_size)
    indices_j = np.arange(0, len(x_train) - 1 - batch_size)
    while True:
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x_train) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x_train) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        if (i + batch_size) < len(x_train) and (j + batch_size) < len(x_train):
            align_x, align_y, score, max_length = (get_alignments(x_train, y_train, i, j, batch_size))
        elif (i + batch_size) >= len(x_train):
            print('End of training set, temp batch:', len(x_train[i:]))
            align_x, align_y, score, max_length = (get_alignments(x_train, y_train, i, j, len(x_train[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, score, max_length = (get_alignments(x_train, y_train, i, j, len(x_train[j:])))
        s1, s2 = split_alignments(align_x, max_length)
        for _, doc in enumerate(pad_sequences(tokenizer.texts_to_sequences(np.append(s1, s2)), maxlen=max_length, padding='post')):
            data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=word_length, negative_samples=5.)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                yield x, y


def alignments2vec(x, y, V, tokenizer):
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(V, vec_length)(w_inputs)

    # context
    c_inputs = Input(shape=(1,), dtype='int32')
    c = Embedding(V, vec_length)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.summary()
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

    history = SkipGram.fit_generator(generate_vec_batch(x, y, batch_size, tokenizer, SkipGram),
                           steps_per_epoch=steps_per_epoch,
                           epochs=300,#len(x_train)//batch_size//steps_per_epoch,
                           validation_data=generate_vec_batch(x, y, batch_size, tokenizer, SkipGram),
                           validation_steps=steps_per_epoch)

    print(history.history.keys())
    # summarize history for accuracy
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    f = open('alignment_vec.txt', 'w')
    f.write('{} {}\n'.format(V - 1, vec_length))
    vectors = SkipGram.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()

    w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)
    print(w2v.most_similar(positive=['a'*word_length]))


def get_list_of_word2vec(x, w2v, max_length, n_samples):
    word_vec = []
    for alignment in x:
        vec = []
        word_list = alignment.split(' ')
        if len(word_list[-1]) != word_length:
            word_list = word_list[:-1]
        for word in word_list:
            try:
                if len(vec) == 0:
                    vec = w2v.word_vec(word)
                else:
                    vec = np.vstack((vec, w2v.word_vec(word)))
            except Exception as e:
                print('Word', word, 'not in vocab')
        vec = np.reshape(vec, (len(word_list), -1))
        if len(word_vec) == 0:
            word_vec = vec
        else:
            word_vec = np.dstack((word_vec, vec))
    word_vec = np.reshape(word_vec, (n_samples, -1, vec_length))
    return word_vec


def generate_word2vec_batch(x, y):
    indices_i = np.arange(0, len(x) - 1 - batch_size)
    indices_j = np.arange(0, len(x) - 1 - batch_size)
    while True:
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)
        if (i + batch_size) < len(x) and (j + batch_size) < len(x):
            align_x, align_y, score, max_length = (get_alignments(x, y, i, j, batch_size))
        elif (i + batch_size) >= len(x):
            print('End of training set, temp batch:', len(x[i:]))
            align_x, align_y, score, max_length = (get_alignments(x, y, i, j, len(x[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, score, max_length = (get_alignments(x, y, i, j, len(x[j:])))
        #text = zip_alignments(align_x, max_length)
        s1, s2 = split_alignments(align_x, max_length)
        word2vec1 = get_list_of_word2vec(s1, w2v, max_length, align_y.shape[0])
        word2vec2 = get_list_of_word2vec(s2, w2v, max_length, align_y.shape[0])
        align_y = np_utils.to_categorical(align_y)
        if align_y.shape[1] == 2:
            yield [word2vec1, word2vec2], align_y


def generate_batch(x, y, tokenizer):
    indices_i = np.arange(0, len(x) - 1 - batch_size)
    indices_j = np.arange(0, len(x) - 1 - batch_size)
    while True:
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        if (i + batch_size) < len(x) and (j + batch_size) < len(x):
            align_x, align_y, score, max_length = (get_alignments(x, y, i, j, batch_size))
        elif (i + batch_size) >= len(x):
            print('End of training set, temp batch:', len(x[i:]))
            align_x, align_y, score, max_length = (get_alignments(x, y, i, j, len(x[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, score, max_length = (get_alignments(x, y, i, j, len(x[j:])))
        a1, a2 = split_alignments(align_x, max_length)
        s1, s2 = [], []
        for alignment in a1:
            s1 = np.append(s1, alignment.split(' '))
        for alignment in a2:
            s2 = np.append(s2, alignment.split(' '))
        s1 = np.array(tokenizer.texts_to_sequences(a1))
        s2 = np.array(tokenizer.texts_to_sequences(a2))
        s1 = np.reshape(s1, (a1.shape[0], -1))
        s2 = np.reshape(s2, (a2.shape[0], -1))
        y1 = np.array(np.hsplit(align_y, 2)[0].T[0])
        y2 = np.array(np.hsplit(align_y, 2)[1].T[0])
        align_y = 1*np.equal(y1, y2)
        y1 = class_to_onehot(y1, num_classes)
        y2 = class_to_onehot(y2, num_classes)
        align_y = np_utils.to_categorical(align_y, num_classes=2)
        yield [s1, s2, score], [y1, y2, align_y]


# load data
dir = os.getcwd() + '/histone_data/'

x0, y0 = dhrt.load_data_and_labels_pos(dir + 'pos/h3.pos', pos=0, sequences_per_family=sequences_per_family)
x1, y1 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me1.pos', pos=1, sequences_per_family=sequences_per_family)
x2, y2 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me2.pos', pos=2, sequences_per_family=sequences_per_family)
x3, y3 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me3.pos', pos=3, sequences_per_family=sequences_per_family)
x4, y4 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k9ac.pos', pos=4, sequences_per_family=sequences_per_family)
x5, y5 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k14ac.pos', pos=5, sequences_per_family=sequences_per_family)
x6, y6 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k36me3.pos', pos=6, sequences_per_family=sequences_per_family)
x7, y7 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k79me3.pos', pos=7, sequences_per_family=sequences_per_family)
x8, y8 = dhrt.load_data_and_labels_pos(dir + 'pos/h4.pos', pos=8, sequences_per_family=sequences_per_family)
x9, y9 = dhrt.load_data_and_labels_pos(dir + 'pos/h4ac.pos', pos=9, sequences_per_family=sequences_per_family)


x_rt = np.concatenate((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9))
y_rt = np.concatenate((y0, y1, y2, y3, y4, y5, y6, y7, y8, y9))


#x_rt, y_rt = dhrt.load_data_and_labels('cami.pos', 'cami.neg')

x_rt = np.array([seq.replace(' ', '') for seq in x_rt])
y_rt = np.array(list(y_rt))
shuffled_rt = np.random.permutation(range(len(x_rt)))
x_shuffle = x_rt[shuffled_rt]
y_shuffle = y_rt[shuffled_rt]
#print('X:', x_shuffle)
#print('Y:', y_shuffle)
print(pairwise2.align.localxx(x_shuffle[0], x_shuffle[1], one_alignment_only=True))

x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                      y_shuffle,
                                                      stratify=y_shuffle,
                                                      test_size=0.2)

print('x shape:', x_train.shape)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(get_vocab('atcgx', word_length))
V = len(tokenizer.word_index) + 1
print('Num Words:', V)

#alignments2vec(x_train, y_train, V, tokenizer) #uncomment to train word2vec representation

alignment_batch = batch_size * batch_size - 2 * batch_size + 2
encoder_a = Input(shape=(None,))
layer_a = Embedding(V, hidden_size)(encoder_a)
layer_a = Dropout(0.2)(layer_a)
#layer_a = (LSTM(hidden_size, return_sequences=True))(layer_a)
out_a = Conv1D(128, word_length)(layer_a)
out_a = MaxPooling1D(5)(out_a)
out_a = Conv1D(256, 3)(out_a)
out_a = MaxPooling1D(5)(out_a)
out_a = LSTM(100)(out_a)
#out_a = GlobalMaxPooling1D()(out_a)
out_a = Dense(512, activation='relu')(out_a)
out_a = Dense(num_classes, activation='softmax')(out_a)

encoder_b = Input(shape=(None,))
layer_b = Embedding(V, hidden_size)(encoder_b)
layer_b = Dropout(0.2)(layer_b)
#layer_b = (LSTM(hidden_size, return_sequences=True))(layer_b)
out_b = Conv1D(128, word_length)(layer_b)
out_b = MaxPooling1D(5)(out_b)
out_b = Conv1D(256, 3)(out_b)
out_b = MaxPooling1D(5)(out_b)
out_b = LSTM(100)(out_b)
#out_b = GlobalMaxPooling1D()(out_b)
out_b = Dense(512, activation='relu')(out_b)
out_b = Dense(num_classes, activation='softmax')(out_b)

align_score = Input(shape=(1,))
score = RepeatVector(hidden_size)(align_score)
score = Permute((2, 1)) (score)
#score_a = multiply([layer_a, align_score])
#score_b = multiply([layer_b, align_score])

decoder = concatenate([layer_a, layer_b], axis=1)

bias = concatenate([decoder, score], axis=1)

#decoder = multiply([decoder, align_score])

#dense_1 = Dense(2048, activation='relu')(pool_2)
#dense_2 = Dense(1024, activation='relu')(dense_1)
conv_1 = Conv1D(128, word_length)(bias)
pool_1 = MaxPooling1D(5)(conv_1)
conv_2 = Conv1D(256, 3)(pool_1)
pool_2 = MaxPooling1D(5)(conv_2)
output = LSTM(100)(pool_2)
output = Dense(512, activation='relu')(output)
output = Dense(2, activation='softmax')(output)
model = Model(inputs=[encoder_a, encoder_b, align_score],
              outputs=[out_a, out_b, output])

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc'],
              loss_weights=[0.1, 0.1, 0.8])
print('Training shapes:', x_train.shape, y_train.shape)
print('Valid shapes:', x_valid.shape, y_valid.shape)
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)

history = model.fit_generator(generate_batch(x_train, y_train, tokenizer),
                              steps_per_epoch=steps_per_epoch,
                              epochs=16,#2 * len(x_train)//batch_size//steps_per_epoch,
                              validation_data=generate_batch(x_valid, y_valid, tokenizer),
                              validation_steps=steps_per_epoch)
# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['dense_2_acc'])
plt.plot(history.history['dense_4_acc'])
plt.plot(history.history['dense_6_acc'])
plt.plot(history.history['val_dense_2_acc'])
plt.plot(history.history['val_dense_4_acc'])
plt.plot(history.history['val_dense_6_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['seq1', 'seq2', 'same', 'val_seq1', 'val_seq2', 'val_same'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['dense_2_loss'])
plt.plot(history.history['dense_4_loss'])
plt.plot(history.history['dense_6_loss'])
plt.plot(history.history['val_dense_2_loss'])
plt.plot(history.history['val_dense_4_loss'])
plt.plot(history.history['val_dense_6_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['total_loss', 'total_val_loss', 'seq1', 'seq2', 'same', 'val_seq1', 'val_seq2', 'val_same'], loc='upper left')
plt.show()

